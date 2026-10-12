# Sharpa nflow learning

This fork adds the 22-joint Sharpa dataset, identity-aware checkpoints and samples, offline metrics, single-node DDP training and CPU all-candidate evaluation. `scripts/train_sharpa.sh` gates training on collected-label audits; `scripts/evaluate_sharpa.py` evaluates the five checkpoints through the companion Bench fork.

See the [complete Chinese reproduction guide](https://github.com/MrSingleFXE/BODex/blob/main/docs/SHARPA_PIPELINE.md) and [model provenance, results and development history](https://github.com/MrSingleFXE/BODex/blob/main/docs/SHARPA.md). Follow the original installation instructions below; datasets, generated labels and trained weights are not bundled.
