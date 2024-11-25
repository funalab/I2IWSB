#!/bin/zsh

# set path
path=$1
name=$2

# run train
python src/tools/custom/i2iwsb/train.py  --conf_file $path/train_$name.cfg

# run test
test_result=$(python src/tools/custom/i2iwsb/test.py --conf_file $path/test.cfg | tee /dev/tty)
source ~/.zshrc
done-notify "${test_result}"