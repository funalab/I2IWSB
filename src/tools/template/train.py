import os
import sys
sys.path.append(os.getcwd())

from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, create_train_dir, get_device
from src.lib.models.models import get_train_model
from src.lib.datasets.dataset import get_dataset
from src.lib.datasets.dataloader import get_train_iterator
from src.lib.optimizers.optimizers import get_optimizer
from src.lib.optimizers.schedulers import get_scheduler
from src.lib.runner.get_runner_args import get_train_args
from src.lib.runner.trainer import Trainer


def main():
    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    generator = set_seed(args=args)

    # Create directory
    save_dir = create_train_dir(args=args)

    # Prepare device
    device = get_device(args=args)

    ''' Dataset '''
    # Load datasets
    train_dataset, validation_dataset = get_dataset(args=args)

    # Create data iterator (data loader)
    train_iterator, validation_iterator = get_train_iterator(args=args,
                                                             train_dataset=train_dataset,
                                                             validation_dataset=validation_dataset,
                                                             generator=generator)

    ''' Model '''
    # Initialize a model to train
    models = get_train_model(args=args, device=device)

    ''' Optimizer and Scheduler'''
    # Initialize an optimizer
    optimizers = get_optimizer(args=args, model=models)

    # Initialize an scheduler
    schedulers = get_scheduler(args=args, optimizer=optimizers)

    ''' Train '''
    # Initialize a runner
    trainer_args = get_train_args(args=args,
                                  optimizer=optimizers,
                                  scheduler=schedulers,
                                  save_dir=save_dir,
                                  device=device)

    # start train
    trainer = Trainer(**trainer_args)
    trainer.train(
        models=models,
        train_iterator=train_iterator,
        validation_iterator=validation_iterator,
    )


if __name__ == '__main__':
    main()
