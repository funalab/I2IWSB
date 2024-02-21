# PairedImageToImageTranslation

## Overview

対あり画像変換タスク

- repository
  - https://gitlab.com/funalab/pairedimagetoimagetranslation

### Datasets
- JUMP-Cell Painting Consortium
  - https://jump-cellpainting.broadinstitute.org/

### Models
- GAN
  - [cWGAN-GP](https://www.nature.com/articles/s41598-022-12914-x)
- Diffusion / Schrödinger Bridge
  - [Palette](https://arxiv.org/abs/2111.05826)
  - [guided-I2I](https://arxiv.org/abs/2303.08863)
  - [I2SB](https://arxiv.org/abs/2302.05872)
- Custom
  - I2WSB: Image-to-Image Wasserstein Schrödinger Bridge
    - 2024/02/19 Created by Morikura

## Usage
### (1) Create virtual environments

#### venv
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip3 install -r requirements.txt
```

### (2) Train and Test for demo

#### Train 
原則，cfgファイルでハイパーパラメータや設定を管理
学習済みモデルはcfgファイルのsave_dir に記述したディレクトリに best_model.pthとして保存される。 
```shell
python src/tools/gan/train.py --conf_file confs/debug/train.cfg
```
#### Test
テストしたい学習済みモデルのパスをcfgファイルのmodel_dir に指定。
```shell
python src/tools/gan/test.py --conf_file confs/debug/test.cfg
```

ただし、model_dirに下記のようなワイルドカードを設定することで、設定条件に適したfolderからbestな条件を自動探索することも可能
```shell
# test.cfg
model_dir = results/trial/*
```

### (3) Run experiments
holdout
```shell
bash scripts/gan/holdout.sh confs/debug/gan/cwgangp fold1
```
cross validation
```shell
bash scripts/gan/cross_validation.sh confs/debug/gan/cwgangp 
```