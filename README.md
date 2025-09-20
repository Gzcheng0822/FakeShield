# How to Run FakeShield locally

Python==3.10
CUDA==11.8
Pytorch==2.1.2


## Install Conda

### Download Conda Install sh

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

### Accept License

```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## Create virtual Environment

```
conda create --name FakeShield python=3.10 -y
conda activate FakeShield
pip install --upgrade pip
```

## Clone FakeShield

```
git clone https://github.com/zhipeixu/FakeShield.git
cd FakeShield
```

## Install Cuda Toolkit/cuDNN/Pytorch

```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install -c nvidia cuda-nvcc=11.8 -y
```

## Install GCC/Cmake

```
conda install -c conda-forge cmake lit -y
conda install -c conda-forge gxx_linux-64=11 -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
```

## Install requirements

### Modify `requirements.txt`

```
transformers==4.37.2
accelerate==0.21.0
deepspeed==0.12.6
einops==0.6.1
tokenizers==0.15.1
```

### Install it

```
pip install -r requirements.txt
```

## Install mmcv

```
pip install -U openmim
mim install mmcv-full
```

### Modify `MFLM/mmdet/__init__.py`

```
mmcv_maximum_version = '1.7.2'
```

## Install LLAVA

### Modify `DTE-FDM/pyproject.toml`

```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "llava"
version = "1.2.2.post1"
description = "Towards GPT-4 like large language and visual assistant."
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
]
dependencies = [
    "torch==2.1.2", "torchvision==0.16.2",
    "transformers==4.37.2", "tokenizers==0.15.1", "sentencepiece==0.1.99", "shortuuid",
    "accelerate==0.21.0", "peft", "bitsandbytes",
    "pydantic", "markdown2[all]", "numpy", "scikit-learn==1.2.2",
    "gradio==4.16.0", "gradio_client==0.8.1",
    "requests", "httpx==0.24.0", "uvicorn", "fastapi",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
]

[project.optional-dependencies]
train = ["deepspeed==0.12.6", "ninja", "wandb"]
build = ["build", "twine"]

[project.urls]
"Homepage" = "https://llava-vl.github.io"
"Bug Tracker" = "https://github.com/haotian-liu/LLaVA/issues"

[tool.setuptools.packages.find]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]

[tool.wheel]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]
```

### Install it

```
cd DTE-FDM/
pip install -e .
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Download weight

```
pip install huggingface_hub
huggingface-cli download --resume-download zhipeixu/fakeshield-v1-22b --local-dir weight/fakeshield-v1-22b
wget https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth -P weight/
```

## Run Demo

### Modify `scripts/cli_demo.sh` :

```
WEIGHT_PATH=weight/fakeshield-v1-22b
IMAGE_PATH=playground/images/Sp_D_CND_A_pla0005_pla0023_0281.jpg
DTE_FDM_OUTPUT=playground/DTE-FDM_output.jsonl
MFLM_OUTPUT=playground/MFLM_output

pip install -q transformers==4.37.2 > /dev/null 2>&1

CUDA_VISIBLE_DEVICES=0 \
python -m llava.serve.cli \
    --model-path  ${WEIGHT_PATH}/DTE-FDM \
    --DTG-path ${WEIGHT_PATH}/DTG.pth \
    --image-path ${IMAGE_PATH} \
    --output-path ${DTE_FDM_OUTPUT}
```

### Run sh

```
bash scripts/cli_demo.sh
```

## Finetune

Modify `scripts/MFLM/finetune_lora.sh`

```
#!/bin/sh

pip install transformers==4.28.0  > /dev/null 2>&1

export PATH=/usr/local/cuda/bin:$PATH
export MASTER_PORT=$(shuf -i 2000-65000 -n 1)

export WEIGHT_PATH=./weight/fakeshield-v1-22b/MFLM
export OUTPUT_DIR=./playground/MFLM_train_result
export DATA_PATH=./dataset
export TRAIN_DATA_CHOICE=ps|CASIA2
export VAL_DATA_CHOICE=ps|CASIA1+

 
deepspeed --include localhost:0,1,2,3 --master_port $MASTER_PORT ./MFLM/train_ft.py \
  --version $WEIGHT_PATH \
  --DATA_PATH $DATA_PATH \
  --vision_pretrained ./weight/sam_vit_h_4b8939.pth \
  --vision-tower openai/clip-vit-large-patch14-336 \
  --exp_name $OUTPUT_DIR \
  --lora_r 8 \
  --lora_alpha 16 \
  --lr 3e-4 \
  --batch_size 6 \
  --pretrained \
  --use_segm_data \
  --tamper_segm_data "ps|CASIA2" \
  --val_tamper_dataset "ps|CASIA1+" \
  --epochs 200 \
  --mask_validation \
  --wandb   \
  --auto_resume \
```

## Clean Environment

```
rm -rf FakeShield
conda activate base
conda env remove -n FakeShield --all
```

