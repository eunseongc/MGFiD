# MGFiD (NAACL 2024 Findings)
Hola!🙌 This is an official repository for our paper "Multi-Granularity Guided Fusion-in-Decoder" accepted in NAACL 2024 Findings. <br>

Before running our code, we share our environments to train and infer with MGFiD. <br>
```
CUDA version: 11.7
Torch version: 2.0.1
Numpy version: 1.23.5
```

## Environment setup
```
export env_name={your_env_name}
export home_dir={your_home_dir_path}
docker run --gpus all --shm-size=8G -it -v ${home_dir}:/workspace --name ${env_name} pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
apt-get update
apt-get install -y git
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.29.2 sentence-transformers wandb numpy==1.23.5 kornia
```

## Data
Data link: https://www.icloud.com/iclouddrive/051-FWmTlOqiBPhhsy7oU8TrQ#open%5Fdomain%5Fdata <br>
Please unzip it to the folder. The directory structure is as follows: <br>
```
MGFiD
├── checkpoints/
├── src/
└── open_domain_data/
```

## Training 
Run this script to train the MGFiD from scratch.<br>
```
source run.sh
```
