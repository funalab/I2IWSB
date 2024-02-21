import os
import sys
sys.path.append(os.getcwd())

from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, create_train_dir, get_device
from src.lib.models.models import get_train_model_DF
from src.lib.datasets.dataset import get_dataset
from src.lib.datasets.dataloader import get_train_iterator
from src.lib.optimizers.optimizers import get_optimizer
from src.lib.optimizers.schedulers import get_scheduler
from src.lib.runner.get_runner_args import get_train_args_DF
from src.lib.runner.trainer import guidedI2ITrainer, I2SBTrainer, PaletteTrainer


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
    model = get_train_model_DF(args=args, device=device)


    ''' Optimizer and Scheduler'''
    # Initialize an optimizer
    optimizer = get_optimizer(args=args, model=model)

    # Initialize an scheduler
    scheduler = get_scheduler(args=args, optimizer=optimizer)

    ''' Train '''
    # Initialize a runner
    trainer_args = get_train_args_DF(args=args,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  save_dir=save_dir,
                                  device=device)

    # start train
    model_name = str(args.model)
    if model_name == 'guided-I2I':
        trainer = guidedI2ITrainer(**trainer_args)
    elif model_name == 'I2SB':
        trainer = I2SBTrainer(**trainer_args)
    elif model_name == 'Palette':
        trainer = PaletteTrainer(**trainer_args)
    else:
        raise NotImplementedError(f"{model_name}")
    trainer.train(
        model=model,
        train_iterator=train_iterator,
        validation_iterator=validation_iterator,
    )


if __name__ == '__main__':
    main()
