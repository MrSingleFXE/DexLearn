"""Audit every collected label and split before training (run from repo root)."""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--objects', type=Path, default=Path('assets/object/DGN_2k'))
    args = parser.parse_args()
    cfg = yaml.safe_load(Path('dexlearn/config/data/bodex_tabletop_sharpa_right.yaml').read_text())
    splits = {k: set(json.loads((args.objects/'valid_split'/f'{k}.json').read_text())) for k in ('train', 'test')}
    if splits['train'] & splits['test'] or {k: len(v) for k,v in splits.items()} != {'train':1918,'test':479}:
        raise ValueError('Expected official disjoint 1918/479 object split')
    audit = json.loads((args.labels.parent/'coverage.json').read_text())
    if not audit['coverage_passed'] or audit.get('collected_rows') != audit['successes']:
        raise ValueError('Run Bench coverage and collection first')
    counts = {'train':0,'test':0}; rows = 0
    for p in args.labels.rglob('*.npy'):
        obj = p.relative_to(args.labels).parts[0]
        split = next((k for k,v in splits.items() if obj in v), None)
        if split is None: raise ValueError(f'Unknown object: {obj}')
        data = np.load(p, allow_pickle=True).item(); n = len(data['grasp_qpos'])
        if not np.all(data['hand_model_id'] == cfg['hand_model_id']) or not np.array_equal(data['joint_names'], np.tile(cfg['joint_names'], (n,1))):
            raise ValueError(f'Hand identity mismatch: {p}')
        for key in ('pregrasp_qpos','grasp_qpos','squeeze_qpos'):
            q = data[key]
            if q.shape != (n,29) or not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q[:,3:7],axis=1),1,atol=1e-4):
                raise ValueError(f'Invalid poses: {p}')
        if not Path(str(data['scene_path'])).is_file(): raise FileNotFoundError(data['scene_path'])
        counts[split] += 1; rows += n
    if rows != audit['successes'] or not all(counts.values()): raise ValueError('Incomplete labels')
    # The real loaders verify scene/view matching and all required sample fields.
    from hydra import compose, initialize_config_dir
    from dexlearn.dataset import create_dataset
    with initialize_config_dir(config_dir=str(Path('dexlearn/config').resolve()), version_base=None):
        cfg = compose(config_name='base', overrides=['data=bodex_tabletop_sharpa_right', f'data.grasp_path={args.labels.resolve()}', f'data.object_path={args.objects.resolve()}'])
    for mode in ('train','eval','test'):
        ds = create_dataset(cfg, mode)
        if len(ds) <= 0: raise ValueError(f'Empty loader: {mode}')
        for i in range(min(4,len(ds))):
            sample = ds[i]
            if sample['point_clouds'].shape != (1024,3) or not np.isfinite(sample['point_clouds']).all(): raise ValueError('Invalid point clouds')
    print(json.dumps(dict(label_scenes=counts, successful_rows=rows, loader_smoke_passed=True),indent=2))


if __name__ == '__main__': main()
