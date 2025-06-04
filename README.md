<div align="center">
    <h2>
Semantic-CC: Boosting Remote Sensing Image Change Captioning via  Foundational Knowledge and Semantic Guidance
    </h2>
</div>

## Introduction

The repository is the code implementation of the paper [Semantic-CC: Boosting Remote Sensing Image Change Captioning via  Foundational Knowledge and Semantic Guidance]([https://arxiv.org/abs/2312.16202](https://arxiv.org/abs/2407.14032)), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Open-CD](https://github.com/likyoo/open-cd) projects.

The current branch has been tested under PyTorch 2.x and CUDA 12.1, supports Python 3.10, and is compatible with most CUDA versions.
## Installation

### Dependencies

- Linux or Windows
- Python 3.7+, recommended 3.10
- PyTorch 2.0 or higher, recommended 2.1
- CUDA 11.7 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1
### Environment Installation
We recommend using Anaconda for installation. The following command will create a virtual environment named `seg` and install PyTorch and MMCV.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `seg` and activate it.

```shell
conda create -n ttp python=3.10 -y
conda activate seg
```

**Step 2**: Install [PyTorch2.1.x](https://pytorch.org/get-started/locally/).

Linux/Windows:
```shell
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```
Or

```shell
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV2.1.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.1.0
```

**Step 4**: Install other dependencies.

```shell
pip install -U wandb einops importlib peft==0.8.2 scipy ftfy prettytable torchmetrics==1.3.1 transformers==4.38.1
```

</details>

## Dataset Preparation


### Levir-CD Change Detection Dataset

#### Dataset Download

- Image and label download address: [Levir-CD](https://chenhao.in/LEVIR/).


### Levir-CC Change Caption Dataset

#### Dataset Download

- Image and label download address: [Levir-CC](https://github.com/Chen-Yang-Liu/LEVIR-CC-Dataset).

## Model Training
```shell
python tools/train.py
```
## Model Testing
We suggest saving the generated results and using the built-in testing code in Levir-CC for performance testing.
```shell
python tools/test.py
```

```
