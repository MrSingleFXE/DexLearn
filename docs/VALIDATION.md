# Release validation

The portable release was checked in isolated checkouts on the existing Ubuntu/A800 server. Original experimental inputs, labels and weights were read without modifying the completed experiment.

- All Python sources parsed; training shell entry passed `bash -n`.
- Official 23 mesh hashes, 22 compiled joint names/ranges, uniform kp12/damping0.9 and enabled joint torque caps passed checks with MuJoCo 3.3.2.
- Re-evaluated 40 archived candidates through the released conversion, physics and collection wrapper: all 40 success flags matched the original full-run evaluation, with exact coverage and collected-row agreement.
- Released deterministic synthesis entry generated two new scenes (40 candidates); the downstream conversion/screening/collection smoke test used these actual outputs.
- Eight-GPU nflow smoke test completed two training steps, validation and checkpoint saves, using batch 2 per GPU for the smoke test. DDP parameter maximum difference was 0.0. This is not a repeat of the 50k-step scientific experiment.
- Dataset audit smoke fixture exercised train/eval/test loaders; the official test dataset enumerated 10,892 scenes.
- Existing 50k checkpoint CPU sampling on a two-scene fixture produced all 200 candidates; released Bench wrapper and ranking analysis completed with exact coverage.

Environment used: BODex PyTorch 2.2.2 / trimesh 5.1.0; Bench MuJoCo 3.3.2; DexLearn PyTorch 2.0.1+cu117 / MinkowskiEngine 0.5.4. Python 3.10. The isolated BODex smoke reused the existing compiled CUDA extensions; a clean compiler/toolchain installation was not re-tested. Follow upstream installation requirements and activate each environment (including its executable PATH).

The previously reported full-run success rates are historical experiment results. This release validation is a small integration check, not a new million-candidate run or a new full training run. Large test artifacts/checkpoints remain external to Git.
