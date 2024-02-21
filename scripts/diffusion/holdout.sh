#!/bin/zsh

# set path
path=$1
name=$2

# run train
python src/tools/diffusion/train.py --conf_file $path/train/$name.cfg

# run test
test_result=$(python src/tools/diffusion/test.py --conf_file $path/test.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"