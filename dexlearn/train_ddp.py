"""Single-node DDP entry point for the BODex tabletop normalizing-flow baseline.

algo.batch_size is PER GPU. Use 32 on eight GPUs or 64 on four for global batch 256.
The upstream dataset randomly samples records, so epochs are loader cycles.
"""
import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import trange

from dexlearn.dataset import create_dataset, minkowski_collate_fn
from dexlearn.network.models import BaseModel
from dexlearn.utils.logger import Logger
from dexlearn.utils.util import set_seed


class DistributedLoader:
    def __init__(self, cfg, mode, rank, world):
        dataset = create_dataset(cfg, mode)
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                          shuffle=mode == 'train', seed=cfg.seed)
        generator = torch.Generator().manual_seed(cfg.seed + rank + (10000 if mode == 'eval' else 0))
        workers = cfg.data.num_workers
        self.loader = DataLoader(dataset, batch_size=cfg.algo.batch_size,
                                 sampler=self.sampler, drop_last=True,
                                 num_workers=workers, collate_fn=minkowski_collate_fn,
                                 generator=generator, persistent_workers=workers > 0,
                                 **({'multiprocessing_context': 'spawn'} if workers else {}))
        if len(self.loader) == 0:
            raise ValueError(f'{mode} loader has no full batch; reduce per-GPU batch size')
        self.dataset_size = len(dataset)
        self.device = cfg.device
        self.epoch = 0
        self.iterator = None

    def get(self):
        if self.iterator is None:
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
        try:
            data = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self.sampler.set_epoch(self.epoch)
            self.iterator = iter(self.loader)
            data = next(self.iterator)
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = (v.float() if v.is_floating_point() else v).to(self.device)
        return data


def require_finite(value, label, device):
    bad = torch.tensor(int(not bool(torch.isfinite(value).all())), device=device)
    dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(f'Non-finite {label} on at least one rank')


def average_metrics(result, device, world):
    keys = sorted(result)
    values = torch.stack([torch.as_tensor(result[k], device=device).detach().float().mean() for k in keys])
    dist.all_reduce(values)
    return dict(zip(keys, (values / world).cpu().tolist()))


def check_parameter_sync(model, device, world):
    # Only enabled for readiness tests: compare every parameter with rank zero.
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    error = (flat-reference).abs().max()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    assert error.item() == 0, f'Parameters diverged: {error.item()}'
    return float(error)


@hydra.main(config_path='config', config_name='base', version_base=None)
def main(cfg):
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    torch.cuda.set_device(local_rank)
    cfg.device = f'cuda:{local_rank}'
    dist.init_process_group('nccl')
    try:
        if HydraConfig.get().runtime.choices.algo != 'nflow':
            raise ValueError('This DDP entry currently supports algo=nflow. Use the tested single-GPU entry for diffusion.')
        set_seed(cfg.seed)
        # Only rank zero writes Hydra snapshots, logs and checkpoints.
        logger = Logger(cfg) if rank == 0 else None
        ckpt_path = [cfg.ckpt if rank == 0 else None]
        dist.broadcast_object_list(ckpt_path, src=0)
        cfg.ckpt = ckpt_path[0]
        if rank == 0:
            folder = Path(cfg.output_folder) / cfg.wandb.id
            (folder / '.hydra').mkdir(parents=True, exist_ok=True)
            OmegaConf.save(cfg, folder / '.hydra/config.yaml')
            OmegaConf.save(HydraConfig.get().overrides.task, folder / '.hydra/overrides.yaml')

        model = BaseModel(cfg.algo.model).to(cfg.device)
        set_seed(cfg.seed + rank)
        train = DistributedLoader(cfg, 'train', rank, world)
        val = DistributedLoader(cfg, 'eval', rank, world)
        checkpoint = torch.load(cfg.ckpt, map_location='cpu') if cfg.ckpt else None
        hand_identity = {}
        if cfg.data.get('hand_model_id') is not None:
            hand_identity = dict(hand_model_id=cfg.data.hand_model_id,
                                 joint_names=list(cfg.data.joint_names))
            if checkpoint:
                for key, expected in hand_identity.items():
                    if checkpoint.get(key) != expected:
                        raise ValueError(f'Checkpoint {key} does not match the configured hand')
        cur_iter = checkpoint['iter'] if checkpoint else 0
        if checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            # ActNorm modifies parameters on its first forward. Initialize BEFORE
            # DDP synchronizes parameters/buffers, rather than independently after it.
            model.train()
            with torch.no_grad():
                initial = model(train.get())
                require_finite(torch.stack([v.mean() for v in initial.values()]), 'initialization', cfg.device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.algo.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.algo.max_iter, eta_min=cfg.algo.lr_min)
        if checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['scheduler']['T_max'] != cfg.algo.max_iter:
                raise ValueError('Resume with the original algo.max_iter to preserve the learning-rate schedule')
            scheduler.load_state_dict(checkpoint['scheduler'])
            if rank == 0:
                print(f'Loaded checkpoint {cfg.ckpt} at step {cur_iter}', flush=True)
            del checkpoint
        steps_per_epoch = len(train.loader)
        info = dict(world_size=world, per_gpu_batch=cfg.algo.batch_size,
                    global_batch=cfg.algo.batch_size * world, train_scenes=train.dataset_size,
                    steps_per_loader_epoch=steps_per_epoch, max_iter=cfg.algo.max_iter,
                    loader_epochs=cfg.algo.max_iter / steps_per_epoch,
                    sampling='random object/scene/grasp/view with replacement; not an exhaustive grasp pass')
        if rank == 0:
            (folder / 'distributed.json').write_text(json.dumps(info, indent=2)+'\n')
            print(json.dumps(info), flush=True)
        verify = cfg.get('verify_ddp', False)
        if verify:
            check_parameter_sync(model, cfg.device, world)
        ddp.train()
        for it in trange(cur_iter, cfg.algo.max_iter, disable=rank != 0):
            optimizer.zero_grad(set_to_none=True)
            result = ddp(train.get())
            loss = sum(v * cfg.algo.loss_weight[k] for k, v in result.items() if k in cfg.algo.loss_weight)
            require_finite(loss, 'training loss', cfg.device)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.algo.grad_clip)
            require_finite(norm, 'gradient norm', cfg.device)
            optimizer.step()
            scheduler.step()
            if (it+1) % cfg.algo.log_every == 0:
                result['lr'] = torch.tensor(scheduler.get_last_lr()[0], device=cfg.device)
                metrics = average_metrics(result, cfg.device, world)
                if rank == 0:
                    logger.log(metrics, 'train', it)
            if (it+1) % cfg.algo.val_every == 0:
                ddp.eval()
                with torch.no_grad():
                    results = [ddp(val.get()) for _ in range(cfg.algo.val_num)]
                    metrics = average_metrics({k: torch.stack([r[k].mean() for r in results]).mean()
                                               for k in results[0]}, cfg.device, world)
                if rank == 0:
                    logger.log(metrics, 'eval', it)
                ddp.train()
            if (it+1) % cfg.algo.save_every == 0 or it+1 == cfg.algo.max_iter:
                if rank == 0:
                    target = Path(logger.save_ckpt_dir) / f'step_{it+1:06d}.pth'
                    temporary = target.with_suffix('.pth.tmp')
                    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scheduler=scheduler.state_dict(), iter=it+1, distributed=info,
                                    **hand_identity), temporary)
                    os.replace(temporary, target)
                dist.barrier()
        if verify:
            error = check_parameter_sync(model, cfg.device, world)
            if rank == 0:
                (folder/'ddp_verification.json').write_text(json.dumps(dict(world_size=world,
                    parameter_max_difference=error, completed_steps=cfg.algo.max_iter,
                    resumed_from_step=cur_iter, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2)+'\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
