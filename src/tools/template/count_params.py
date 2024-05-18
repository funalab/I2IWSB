import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, create_test_dir, get_device
from src.lib.models.models import get_test_model_DF
from src.lib.datasets.dataset import get_test_dataset
from src.lib.datasets.dataloader import get_test_iterator
from src.lib.runner.get_runner_args import get_test_args_DF
from src.lib.runner.tester import guidedI2ITester, I2SBTester, PaletteTester
from src.lib.utils.utils import calculate_inputsize
from torchinfo import summary

def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    _ = set_seed(args=args)

    # Prepare device
    device = get_device(args=args)

    ''' Model '''
    # Get the pretrained model
    model, best_model_dir = get_test_model_DF(args=args, device=device)

    # Create directory
    save_dir = create_test_dir(args=args, best_model_dir=best_model_dir)

    # input size
    input_size = calculate_inputsize(args=args)

    summary_model = summary(model=model, input_size=input_size)

    with open(f'{save_dir}/params_count_G.txt', 'w') as f:
        print(summary_model, file=f)


if __name__ == '__main__':
    main()
