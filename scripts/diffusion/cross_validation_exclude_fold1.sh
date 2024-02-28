#!/bin/zsh

# set path
path=$1

# run train
#python src/tools/diffusion/train.py --conf_file $path/train_fold1.cfg
python src/tools/diffusion/train.py --conf_file $path/train_fold2.cfg
python src/tools/diffusion/train.py --conf_file $path/train_fold3.cfg
python src/tools/diffusion/train.py --conf_file $path/train_fold4.cfg
python src/tools/diffusion/train.py --conf_file $path/train_fold5.cfg

# run test
test_result=$(python src/tools/diffusion/test.py --conf_file $path/test.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"