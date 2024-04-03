## Environment setup
## CUDA version: 11.6
## torch version: 2.0.1
#### numpy version: 1.23.5

# While we attach all the library list in requirements.txt, you only need to install below.

# Data link: https://file.io/7JTY1haSn7HT
# Unzip the file and put two folders, i.e., nq and dev, into the folder named open_domain_data before exectue the run.sh

```
export env_name={your_env_name}
export home_dir={your_home_dir_path}
docker run --gpus all --shm-size=8G -it -v ${home_dir}:/workspace --name ${env_name} pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
apt-get update
apt-get install -y git
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install transformers==4.29.2 sentence-transformers wandb numpy==1.23.5 kornia
```
# MGFiD
