import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed_for_cWGAN_GP, create_test_dir, get_device
from src.lib.models.models import get_test_model_WSB
from src.lib.datasets.dataset import get_test_dataset
from src.lib.datasets.dataloader import get_test_iterator
from src.lib.runner.get_runner_args import get_test_args_WSB
from src.lib.runner.trainer import cWSBGPTester
from src.lib.utils.utils import calculate_inputsize
from torchinfo import summary

def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Prepare device
    device = get_device(args=args)

    ''' Model '''
    # Get the pretrained model
    model_G, model_D, best_model_dir = get_test_model_WSB(args=args, device=device)

    # Create directory
    save_dir = create_test_dir(args=args, best_model_dir=best_model_dir)

    # input size
    input_size = calculate_inputsize(args=args)

    summary_G = summary(model=model_G, input_size=input_size)

    with open(f'{save_dir}/params_count_G.txt', 'w') as f:
        print(summary_G, file=f)

    summary_D = summary(model=model_D, input_size=input_size)

    with open(f'{save_dir}/params_count_D.txt', 'w') as f:
        print(summary_D, file=f)

if __name__ == '__main__':
    main()
