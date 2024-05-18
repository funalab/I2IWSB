# JUMP Cell Painting Datasets
### (A) Description
#### JUMP-Cell Painting Consortium
- HP
  - https://jump-cellpainting.broadinstitute.org/

- Github
  - https://github.com/jump-cellpainting/datasets

#### Cell Painting Gallery
- Arxiv, 3 Feb 2024
  - https://arxiv.org/abs/2402.02203

- Nat. Methods, 9 Apr 2024 (CPJUMP1: cpg0000-jump-pilot)
  - https://www.nature.com/articles/s41592-024-02241-6

- AWS
  - https://registry.opendata.aws/cellpainting-gallery/

- Github
  - https://github.com/broadinstitute/cellpainting-gallery

### (B) Download documentation
- official cellpainting-galley documentation
  - https://github.com/broadinstitute/cellpainting-gallery/blob/main/download_instructions.md

- official cpg0000-jump-pilot documentation
  - https://github.com/jump-cellpainting/2023_Chandrasekaran_submitted
  - https://github.com/jump-cellpainting/JUMP-Target

### (C) How to download
#### 1. Create virtual environments using ```venv``` module
```shell
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip3 install awscli==1.32.45
```

#### 2. Run custom shell script to download
```shell
bash download_datasets_JUMP.sh <save_directory_path>
```
Please define a storage path in ```<save_directory_path>```. 
For example,
```shell
bash download_datasets_JUMP.sh ./cellpainting-gallery
```
Using this script, you can download ```illum```, ```image```, ```load_data_csv``` and ```metadata``` folders 
from the specified project and plate folders.

- ```illum```: there are separate illumination correction functions, one for each of the 8 channels imaged in that plate (e.g. BR00117035_IllumAGP.npy is the correction function for the AGP channel.)
- ```image```: there are separate raw image data
- ```load_data_csv```: there are csvs that contains the correspondence between the file path and the fluorescent label
- ```metadata```: there are metadata. 

If you want to download other folders, please refer to the official documentation listed above 
and change the shell scripts manually.

(NOTE)  
The plate_names list to be specified in ```download_datasets_JUMP.sh``` can be selected from the list obtained by the following command
```shell
aws s3 ls s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/ --no-sign-request
```

#### 3. Check data
please check data using ```../notebooks/JUMP_visualization.ipynb```


#### 4. Build datasets
```shell
python build_datasets_JUMP.py --root_path <path> --save_path <save_path> --test_ratio <test_ratio> \
                              --test_split_mode <test_split_mode> --fold_num <fold_num>
```
- ```root_path```: data root path
- ```save_path```: save path
- ```test_ratio```: test dataset ratio
- ```test_split_mode```: split mode (choose wellplate, well or random)
- ```fold_num```: number of cross-validation fold


#### (Option) Extract tests
Commands to use when you want to reduce the number of tests
```shell
python extract_test.py --test_path <path> --num_site_per_well <num_site_per_well>
```
- ```test_path```: test path to extract
- ```num_site_per_well```: number of sites to get from each well