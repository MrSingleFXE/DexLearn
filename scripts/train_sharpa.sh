#!/usr/bin/env bash
# Run from the repository root with the DexLearn Python environment activated.
set -euo pipefail
LABELS="$(cd "$1" && pwd)"
EXPERIMENT="${2:-sharpa_v3_s34_kp12_alpha1}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_LIST <<< "$GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ -e "output/bodex_tabletop_sharpa_right_nflow_${EXPERIMENT}" ]]; then
  echo 'Choose a fresh experiment name; refusing to overwrite a run.' >&2; exit 1
fi
python scripts/audit_sharpa_dataset.py --labels "$LABELS"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#GPU_LIST[@]}" \
  --module dexlearn.train_ddp data=bodex_tabletop_sharpa_right algo=nflow \
  "exp_name=$EXPERIMENT" "data.grasp_path=$LABELS" resume=false ckpt=null wandb.mode=disabled \
  algo.batch_size=32 data.num_workers=8 algo.max_iter=50000 algo.save_every=10000 \
  algo.val_every=1000 algo.log_every=10 +verify_ddp=true
