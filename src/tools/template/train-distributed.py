import os
import copy
import sys
sys.path.append(os.getcwd())

import random
import torch
import numpy as np
from torch.multiprocessing import Process
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import create_train_dir, get_device
from src.lib.models.models import get_train_model_DF
from src.lib.datasets.dataset import get_dataset
from src.lib.datasets.dataloader import get_train_iterator
from src.lib.runner.get_runner_args import get_train_args_DF
from src.lib.runner.trainer import I2SBDistributedTrainer
from src.lib.utils.distributed_util import init_processes

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    ''' Settings '''
    # Prepare device
    torch.cuda.set_device(int(args.local_rank))
    device = torch.device(f'cuda:{args.local_rank}')

    # set seed: make sure each gpu has differnet seed!
    if int(args.seed) is not None:
        set_seed(int(args.seed) + int(args.global_rank))

    # Set generator
    generator = torch.Generator()
    generator.manual_seed(int(args.seed))

    # Create directory
    save_dir = create_train_dir(args=args)

    ''' Dataset '''
    # Load datasets
    train_dataset, validation_dataset = get_dataset(args=args)

    ''' Model '''
    # Initialize a model to train
    model = get_train_model_DF(args=args, device=device)


    ''' Optimizer and Scheduler'''
    ''' Train '''
    # Initialize a runner
    trainer_args = get_train_args_DF(args=args,
                                     optimizer=None,
                                     scheduler=None,
                                     save_dir=save_dir,
                                     device=device)

    # start train
    model_name = str(args.model)
    if model_name == 'I2SB':
        trainer = I2SBDistributedTrainer(**trainer_args)
    else:
        raise NotImplementedError(f"{model_name}")
    trainer.train(
        args=args,
        model=model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        generator=generator,
    )


if __name__ == '__main__':
    # Parse config parameters
    args = config_paraser()

    set_seed(int(args.seed))

    size = int(args.n_gpu_per_node)
    args.distributed = size > 1
    args.use_fp16 = 'False' # disable fp16 for training

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        args.ngc_job_id = os.environ["NGC_JOB_ID"]

    assert int(args.batchsize) % int(args.microbatch) == 0, \
        f"{int(args.batchsize)} is not dividable by {int(args.microbatch)}!"

    # settings to distribute
    processes = []
    for rank in range(size):
        opt = copy.deepcopy(args)
        opt.local_rank = rank
        global_rank = rank + int(opt.node_rank) * int(opt.n_gpu_per_node)
        global_size = int(opt.num_proc_node) * int(opt.n_gpu_per_node)
        opt.global_rank = global_rank
        opt.global_size = global_size
        print('Node rank %d, local proc %d, global proc %d, global_size %d' % (int(opt.node_rank), rank,
                                                                               global_rank, global_size))
        p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
