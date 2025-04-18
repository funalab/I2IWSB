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


def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    _ = set_seed(args=args)

    # Prepare device
    device = get_device(args=args)

    ''' Dataset '''
    # Load datasets
    test_dataset = get_test_dataset(args=args)

    # Create dataloader
    test_iterator = get_test_iterator(args=args, test_dataset=test_dataset)

    ''' Model '''
    # Get the pretrained model
    model, best_model_dir = get_test_model_DF(args=args, device=device)

    # Create directory
    save_dir = create_test_dir(args=args, best_model_dir=best_model_dir)

    ''' Test '''
    tester_args = get_test_args_DF(args, save_dir, device, test_dataset)

    model_name = str(args.model)
    if model_name == 'guided-I2I':
        tester = guidedI2ITester(**tester_args)
    elif model_name == 'I2SB':
        tester = I2SBTester(**tester_args)
    elif model_name == 'Palette':
        tester = PaletteTester(**tester_args)
    else:
        raise NotImplementedError(f"{model_name}")

    _, _ = tester.test(model=model, data_iter=test_iterator, phase='test')


if __name__ == '__main__':
    main()
