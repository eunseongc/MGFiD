# MGFiD (NAACL 2024 Findings)
Hola!🙌 This is an official repository for our paper "Multi-Granularity Guided Fusion-in-Decoder" accepted in NAACL 2024 Findings. <br>

Before running the code, we share our environments to train and infer with MGFiD. <br>
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
Please unzip it to the folder. The directory structure should be as follows: <br>
```
├── MGFiD
│   ├── checkpoints/
│   ├── src/
│   ├── open_domain_data/
│   │   ├── nq/
│   │   └── tqa/
```

## Checkpoints & Inference
NQ: https://www.icloud.com/iclouddrive/04fVoGxOaOkwaibJXDMUJJiTA#nq%5Fmgfid <br>
TQA: https://www.icloud.com/iclouddrive/0a6ga0K6TxRXZmXBblXxbOnsw#tqa%5Fmgfid <br>

Evaluation commands
```
## For Natural Questions test (NQ test)
source run_eval.sh checkpoints/nq_mgfid/checkpoint/best_dev/

## For TriviaQA (TQA test)
source run_eval.sh checkpoints/tqa_mgfid/checkpoint/best_dev/
```

The expected results should be as follows:
|         | NQ (test) | TQA (test) |
|---------|-----------|------------|
| MGFiD   | 49.9169   | 68.4611    |


## Training 
Run this script to train the MGFiD from scratch.<br>
```
source run.sh
```




## Citation
Please cite our paper:
```
@inproceedings{ChoiLL24,
    author = {Eunseong Choi and
              Hyeri Lee and
              Jongwuk Lee},
    title = {Multi-Granularity Guided Fusion-in-Decoder},
    booktitle = {Findings of the Association for Computational Linguistics: NAACL 2024},
    pages = {2201--2212},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2024.findings-naacl.142},
    year = {2024},
}
```

