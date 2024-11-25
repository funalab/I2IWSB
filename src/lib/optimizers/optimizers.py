import os
import torch
import torch.optim as optim
from src.lib.utils.utils import CustomException


def modify_state(args, optimizer):
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]
        device = torch.device('cuda')
    else:
        device = torch.device(str(args.device))

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    return optimizer


def get_optimizer_WSB(args, model_G, model_D):
    if args.optimizer_g == 'SGD':
        optimizer_G = optim.SGD(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            momentum=float(args.momentum_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'Adadelta':
        optimizer_G = optim.Adadelta(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            rho=float(args.rho_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'Adagrad':
        optimizer_G = optim.Adagrad(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'Adam':
        optimizer_G = optim.Adam(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            betas=eval(args.betas_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'AdamW':
        optimizer_G = optim.AdamW(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            betas=eval(args.betas_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'SparseAdam':
        optimizer_G = optim.SparseAdam(
            params=model_G.parameters(),
            lr=float(args.lr_g)
            )
    elif args.optimizer_g == 'Adamax':
        optimizer_G = optim.Adamax(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'ASGD':
        optimizer_G = optim.ASGD(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            weight_decay=float(args.weight_decay_g)
            )
    elif args.optimizer_g == 'RMSprop':
        optimizer_G = optim.RMSprop(
            params=model_G.parameters(),
            lr=float(args.lr_g),
            momentum=float(args.momentum_g),
            weight_decay=float(args.weight_decay_g)
            )
    else:
        raise ValueError('Unknown optimizer_G name: {}'.format(args.optimizer_g))

    if args.optimizer_d == 'SGD':
        optimizer_D = optim.SGD(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            momentum=float(args.momentum_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'Adadelta':
        optimizer_D = optim.Adadelta(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            rho=float(args.rho_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'Adagrad':
        optimizer_D = optim.Adagrad(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'Adam':
        optimizer_D = optim.Adam(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            betas=eval(args.betas_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'AdamW':
        optimizer_D = optim.AdamW(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            betas=eval(args.betas_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'SparseAdam':
        optimizer_D = optim.SparseAdam(
            params=model_D.parameters(),
            lr=float(args.lr_d)
            )
    elif args.optimizer_d == 'Adamax':
        optimizer_D = optim.Adamax(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'ASGD':
        optimizer_D = optim.ASGD(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            weight_decay=float(args.weight_decay_d)
            )
    elif args.optimizer_d == 'RMSprop':
        optimizer_D = optim.RMSprop(
            params=model_D.parameters(),
            lr=float(args.lr_d),
            momentum=float(args.momentum_d),
            weight_decay=float(args.weight_decay_d)
            )
    else:
        raise ValueError('Unknown optimizer_D name: {}'.format(args.optimizer_d))

    if hasattr(args, 'init_model') and str(args.init_model) != 'None':
        stdict_G = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['optimizer_G']
        stdict_D = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['optimizer_D']

        optimizer_G.load_state_dict(stdict_G)
        optimizer_D.load_state_dict(stdict_D)

        optimizer_G = modify_state(args, optimizer_G)
        optimizer_D = modify_state(args, optimizer_D)

    if hasattr(args,'reuse') and eval(args.reuse):
        if not hasattr(args, 'init_model') or str(args.init_model) == 'None':
            raise CustomException('reuse flag is True, but init_model was not set')

    return optimizer_G, optimizer_D

def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(
            params=model.parameters(),
            lr=float(args.lr),
            rho=float(args.momentum),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'Adagrad':
        optimizer = optim.Adagrad(
            params=model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'SparseAdam':
        optimizer = optim.SparseAdam(
            params=model.parameters(),
            lr=float(args.lr)
            )
    elif args.optimizer == 'Adamax':
        optimizer = optim.Adamax(
            params=model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'ASGD':
        optimizer = optim.ASGD(
            params=model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay)
            )
    elif args.optimizer == 'RMSprop':
        optimizer = optim.RMSprop(
            params=model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            weight_decay=float(args.weight_decay)
            )
    else:
        raise ValueError('Unknown optimizer name: {}'.format(args.optimizer))

    if hasattr(args, 'init_model') and str(args.init_model) != 'None':
        stdict = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['optimizer']

        optimizer.load_state_dict(stdict)

        optimizer = modify_state(args, optimizer)

    if hasattr(args,'reuse') and eval(args.reuse):
        if not hasattr(args, 'init_model') or str(args.init_model) == 'None':
            raise CustomException('reuse flag is True, but init_model was not set')

    return optimizer