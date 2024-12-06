______________________________________________________________________

<div align="center">

# Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry (ECCV 2024 VCAD Workshop)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2409.08769-B31B1B.svg)](https://arxiv.org/abs/2409.08769)

</div>

## Description

This reporsitory contains codes for paper Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry.


## Installation

Make sure you installed correct PyTorch version for your specific development environment.
Other requirements can be installed via `pip`.

#### Pip

```bash
# clone project
git clone https://github.com/ybkurt/VIFT
cd VIFT

# [OPTIONAL] create conda environment
conda create -n vift python=3.9
conda activate vift

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Downloading KITTI Dataset

```bash
cd data
sh data_prep.sh
```

This script will put the KITTI dataset under `data/kitti_data` folder. 
You need to use `configs/data/kitti_vio.yaml` for dataloading.

You need to use `configs/model/vio.yaml` to train models with KITTI dataset.
You can change the `net` field in config to target your `nn.Module` object. The requirements are as follows for proper functionality:

- See `src/models/components/vio_simple_dense_net.py` for an example `nn.Module` object consisting of multiple linear layers and ReLU activations.

## Downloading Pretrained Image and IMU Encoders

We use pretrained image and IMU encoders of [Visual-Selective-VIO](https://github.com/mingyuyng/Visual-Selective-VIO) model. Download the model weights from repository and put them under the `pretrained_models` directory.


### Caching Latents for KITTI Dataset  

We use pretrained visual and inertial encoders from Visual_Selective_VIO to save the latent vectors for KITTI dataset.


```bash
cd data
python latent_caching.py
python latent_val_caching.py
```

This script will put the latent training KITTI dataset under `data/kitti_latent_data` folder. 
You need to use `configs/data/kitti_vio.yaml` for dataloading.

You need to use `configs/model/vio.yaml` to train models with KITTI dataset.
You can change the `net` field in config to target your `nn.Module` object.

- See `src/models/components/vio_simple_dense_net.py` for an example `nn.Module` object consisting of multiple linear layers and ReLU activations.

## Replicating Experiments in Paper

After saving latents for KITTI dataset, you can run following command to run the experiments in the paper.

```bash
sh scripts/schedule_paper.sh
```

## How to run

Train model with default configuration

```bash
# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml trainer=gpu
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Acknowledgments
This project makes use of code from the following open-source projects:

- [RPMG](https://github.com/JYChen18/RPMG): License: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
- [Visual Selective VIO](https://github.com/mingyuyng/Visual-Selective-VIO)



We are grateful to the authors and contributors of these projects for their valuable work.

## CITATION

If you find our work useful in your research, please consider citing:

```
@article{kurt2024vift,
  title={Causal Transformer for Fusion and Pose Estimation in Deep Visual Inertial Odometry},
  author={Kurt, Yunus Bilge and Akman, Ahmet and Alatan, Ayd{\i}n},
  journal={arXiv preprint arXiv:2409.08769},
  year={2024}
}
```
