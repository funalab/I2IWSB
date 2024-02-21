import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, create_test_dir, get_device
from src.lib.models.models import get_test_model
from src.lib.datasets.dataset import get_test_dataset
from src.lib.datasets.dataloader import get_test_iterator
from src.lib.runner.get_runner_args import get_test_args
from src.lib.runner.tester import Tester


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
    model, best_model_dir = get_test_model(args=args, device=device)

    # Create directory
    save_dir = create_test_dir(args=args, best_model_dir=best_model_dir)

    ''' Test '''
    tester_args = get_test_args(args, save_dir, device, test_dataset)
    tester = Tester(**tester_args)

    _, _, result = tester.test(model=model, data_iter=test_iterator, phase='test')

    with open(os.path.join(save_dir, 'log.json'), 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == '__main__':
    main()
