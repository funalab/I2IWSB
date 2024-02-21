import os
import sys
sys.path.append(os.getcwd())
import torch
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, create_train_dir, get_device
from src.lib.models.models import get_train_model_WSB
from src.lib.datasets.dataset import get_dataset
from src.lib.datasets.dataloader import get_train_iterator
from src.lib.optimizers.optimizers import get_optimizer_GAN
from src.lib.optimizers.schedulers import get_scheduler_GAN
from src.lib.runner.get_runner_args import get_train_args_WSB
from src.lib.runner.trainer import cWSBGPTrainer

'''
conditional Wasserstein Schrödinger Bridge with gradient penaltyの実装
'''
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
    model_G, model_D = get_train_model_WSB(args=args, device=device)

    ''' Optimizer and Scheduler'''
    # Initialize an optimizer
    optimizer_G, optimizer_D = get_optimizer_GAN(args=args, model_G=model_G, model_D=model_D)

    # Initialize an scheduler
    scheduler_G, scheduler_D = get_scheduler_GAN(args=args, optimizer_G=optimizer_G, optimizer_D=optimizer_D)

    ''' Train '''
    # Initialize a runner
    trainer_args = get_train_args_WSB(args=args,
                                      optimizer_G=optimizer_G,
                                      optimizer_D=optimizer_D,
                                      scheduler_G=scheduler_G,
                                      scheduler_D=scheduler_D,
                                      save_dir=save_dir,
                                      device=device)

    # start train
    if str(args.model) == "cWSB-GP":
        trainer = cWSBGPTrainer(**trainer_args)
    else:
        raise NotImplementedError

    trainer.train(
        model_G=model_G,
        model_D=model_D,
        train_iterator=train_iterator,
        validation_iterator=validation_iterator,
    )


if __name__ == '__main__':
    main()
