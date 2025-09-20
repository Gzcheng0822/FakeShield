# How to Run FakeShield locally

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
git clone https://github.com/Gzcheng0822/FakeShield.git
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

```
pip install -r requirements.txt
```

## Install mmcv

```
pip install -U openmim
mim install mmcv-full==1.7.2
```

## Install LLAVA

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

