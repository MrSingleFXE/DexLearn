# Sharpa nflow learning

This fork adds the 22-joint Sharpa dataset, identity-aware checkpoints and samples, offline metrics, single-node DDP training and CPU all-candidate evaluation. `scripts/train_sharpa.sh` gates training on collected-label audits; `scripts/evaluate_sharpa.py` evaluates the five checkpoints through the companion Bench fork.

See the [complete Chinese reproduction guide](https://github.com/MrSingleFXE/BODex/blob/main/docs/SHARPA_PIPELINE.md) and [model provenance, results and development history](https://github.com/MrSingleFXE/BODex/blob/main/docs/SHARPA.md). Follow the original installation instructions below; datasets, generated labels and trained weights are not bundled.

---

# News (September 2026):

Our follow-up work [KPGrasp [CORL 2026]](https://arxiv.org/pdf/2606.09314) provides a much better (i.e., simple yet scalable) learning framework for dexterous grasp generation, although we are still not sure whether we would open-source it in the near future.

# DexLearn

Learning-based grasp synthesis baselines (e.g., diffusion model and normalizing flow) for dexterous hands, used in [BODex (ICRA 2025)](https://pku-epic.github.io/BODex/) and [Dexonomy (RSS 2025)](https://pku-epic.github.io/Dexonomy/)


## Installation
```bash
git submodule update --init --recursive --progress

conda create -n dexlearn python=3.10 
conda activate dexlearn

# pytorch
conda install -c conda-forge mkl=2020.2 -y
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia 

# pytorch3d
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch3d/linux-64/pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2
conda install -y --use-local ./pytorch3d-0.7.8-py310_cu118_pyt210.tar.bz2


# Diffusers 
cd third_party/diffusers
pip install -e .
cd ...

# MinkowskiEngine
cd third_party/MinkowskiEngine
sudo apt install libopenblas-dev
export CUDA_HOME=/usr/local/cuda-11.7
python setup.py install --blas=openblas
cd ...

# nflows
cd third_party/nflows
pip install -e .
cd ...

# dexlearn
pip install -e .
pip install numpy==1.26.4
pip install hydra-core
```
you may need to run the following command to avoid potential errors related to MKL such as `undefined symbol: iJIT_NotifyEvent`
```
conda install -c conda-forge mkl=2020.2 -y

## Getting Started
1. **Prepare assets**. Download Dexonomy dataset from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy). The folder should be organized as below:
```
DexLearn/assets
|- grasp
|   |_ Dexonomy_GRASP_shadow
|        |_ succ_collect
|_ object
    |- DGN_5k
    |   |- valid_split
    |   |- processed_data
    |   |- scene_cfg
    |   |_ vision_data
    |_ objaverse_5k
        |- valid_split
        |- processed_data
    |   |- scene_cfg
        |_ vision_data
```
Similarly, you can download BODex dataset from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex). 

2. **Running**. 
```bash
CUDA_VISIBLE_DEVICES=0 python -m dexlearn.train exp_name=type1 algo=nflow data=dexonomy_shadow
CUDA_VISIBLE_DEVICES=0 python -m dexlearn.sample -e dexonomy_shadow_nflow_type1       
```
If you want to use our pre-trained models, please download the checkpoints from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy) and put them in the `output` folder.
```
DexLearn/output
|- dexonomy_shadow_nflow_cond
|   |- .hydra
|   |- ckpts
```
and then you can run the sampling command below:
```bash
CUDA_VISIBLE_DEVICES=0 python -m dexlearn.sample -e dexonomy_shadow_nflow_cond algo.model.grasp_type_emb.use_predictor=True
```
Now we donot support the training of predictor for grasp type.

3. **Evaluating in simulation**: Please see [DexGraspBench](https://github.com/JYChen18/DexGraspBench?tab=readme-ov-file#running)

## License

This work and the dataset are licensed under [CC BY-NC 4.0][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png

## Citation

If you find this work useful for your research, please consider citing:
```
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}
@article{chen2025dexonomy,
        title={Dexonomy: Synthesizing All Dexterous Grasp Types in a Grasp Taxonomy},
        author={Chen, Jiayi and Ke, Yubin and Peng, Lin and Wang, He},
        journal={Robotics: Science and Systems},
        year={2025}
      }
```

## Acknowledgment

Thanks [@haoranliu](https://lhrrhl0419.github.io/) for his codebase and help in normalizing flow.
