import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed_for_cWGAN_GP, create_test_dir, get_device
from src.lib.models.models import get_test_model_GAN
from src.lib.datasets.dataset import get_test_dataset
from src.lib.datasets.dataloader import get_test_iterator
from src.lib.runner.get_runner_args import get_test_args
from src.lib.runner.trainer import cWGANGPTester


def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    generator_for_cWGAN_GP = set_seed_for_cWGAN_GP(args=args)

    # Prepare device
    device = get_device(args=args)

    ''' Dataset '''
    # Load datasets
    test_dataset = get_test_dataset(args=args)

    # Create dataloader
    test_iterator = get_test_iterator(args=args, test_dataset=test_dataset)

    ''' Model '''
    # Get the pretrained model
    model_G, model_D, best_model_dir = get_test_model_GAN(args=args, device=device)

    # Create directory
    save_dir = create_test_dir(args=args, best_model_dir=best_model_dir)

    ''' Test '''
    tester_args = get_test_args(args, save_dir, device, test_dataset)
    tester = cWGANGPTester(**tester_args)

    _, _, _ = tester.test(
        model_G=model_G,
        model_D=model_D,
        data_iter=test_iterator,
        phase='test',
    )


if __name__ == '__main__':
    main()
