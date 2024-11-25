import os
import json
import importlib
import torch
import torch.nn as nn
from glob import glob
from src.lib.losses.losses import *
from src.lib.models.cwgan_gp import UNet, Discriminator
from src.lib.models.guided_I2I import Palette as PaletteForGuidedI2I
#from src.lib.models.guided_diffusion_modules import *
from src.lib.models.diffusion_for_I2SB import I2SB
from src.lib.models.palette import Palette
from src.lib.utils.utils import CustomException
'''
wrapper function to get model
'''

'''
diffusion model
'''
def get_train_model_DF(args, device):

    if eval(args.crop_size) is not None:
        image_size = eval(args.crop_size)[0]
    elif eval(args.resize) is not None:
        image_size = eval(args.resize)[0]
    else:
        image_size = eval(args.image_size)[0]

    in_channels = int(args.in_channels)+int(args.out_channels)
    out_channels = int(args.out_channels)
    if hasattr(args, 'dim_match') and eval(args.dim_match):
        if int(args.in_channels) > int(args.out_channels):
            in_channels = int(args.in_channels) * 2
            out_channels = int(args.in_channels)
        elif int(args.in_channels) < int(args.out_channels):
            in_channels = int(args.out_channels) * 2
            out_channels = int(args.out_channels)

    if args.model == 'guided-I2I':

        unet_params = {
            "in_channel": in_channels,
            "out_channel": out_channels,
            "inner_channel": int(args.inner_channel),
            "channel_mults": eval(args.channel_mults),
            "attn_res": eval(args.attn_res),
            "num_head_channels": int(args.num_head_channels),
            "res_blocks": int(args.res_blocks),
            "dropout": float(args.dropout),
            "image_size": image_size,
            "num_classes": int(args.num_classes) if eval(args.num_classes) is not None else None
        }

        beta_schedule_params = {
            "train": {
                "schedule": str(args.schedule),
                "n_timestep": int(args.n_timestep),
                "linear_start": float(args.linear_start),
                "linear_end": float(args.linear_end)
            },
            "test": {
                "schedule": str(args.schedule),
                "n_timestep": int(args.n_timestep),
                "linear_start": float(args.linear_start),
                "linear_end": float(args.linear_end)
            }
        }

        model = PaletteForGuidedI2I(unet=unet_params,
                                    beta_schedule=beta_schedule_params,
                                    verbose=eval(args.verbose),
                                    module_name=str(args.module_name),  # 'guided_diffusion'
                                    )
        model.set_loss(eval(args.lossfun))
        model.init_weights()
        model.set_new_noise_schedule(device=device, phase='train')

    elif args.model == 'Palette':

        unet_params = {
            'image_size': image_size,
            'in_channel': in_channels,
            'inner_channel': int(args.inner_channel),
            'out_channel': out_channels,
            'res_blocks': int(args.res_blocks),
            'attn_res': eval(args.attn_res),
            'dropout': float(args.dropout),
            'channel_mults': eval(args.channel_mults),
            'conv_resample': eval(args.conv_resample),
            'use_checkpoint': eval(args.use_checkpoint),
            'use_fp16': eval(args.use_fp16),
            'num_heads': int(args.num_heads),
            'num_head_channels': int(args.num_head_channels),
            'num_heads_upsample': int(args.num_heads_upsample),
            'use_scale_shift_norm': eval(args.use_scale_shift_norm),
            'resblock_updown': eval(args.resblock_updown),
            'use_new_attention_order': eval(args.use_new_attention_order),
        }

        beta_schedule_params = {
            "train": {
                "schedule": str(args.schedule),
                "n_timestep": int(args.n_timestep),
                "linear_start": float(args.linear_start),
                "linear_end": float(args.linear_end)
            },
            "test": {
                "schedule": str(args.schedule),
                "n_timestep": int(args.n_timestep),
                "linear_start": float(args.linear_start),
                "linear_end": float(args.linear_end)
            }
        }

        model = Palette(unet=unet_params,
                        beta_schedule=beta_schedule_params,
                        verbose=eval(args.verbose),
                        module_name=str(args.module_name),  # 'guided_diffusion'
                        )
        model.set_loss(eval(args.lossfun))
        model.set_new_noise_schedule(device=device, phase='train')

    elif args.model == 'I2SB':

        attention_ds = []
        for res in eval(args.attn_res):
            attention_ds.append(image_size // int(res))

        unet_params = {
            "image_size": image_size,
            "in_channels": in_channels,
            'num_res_blocks': int(args.num_res_blocks),
            "num_channels": int(args.num_channels),
            "out_channels": out_channels,
            'attention_ds': tuple(attention_ds),
            "dropout": float(args.dropout),
            "channel_mult": eval(args.channel_mults),
            'class_cond': eval(args.class_cond),
            'learn_sigma': eval(args.learn_sigma),
            'use_fp16': eval(args.use_fp16),
            'num_head': int(args.num_head),
            "num_head_channels": int(args.num_head_channels),
            'num_heads_upsample': int(args.num_heads_upsample),
            'use_scale_shift_norm': eval(args.use_scale_shift_norm),
            'resblock_updown': eval(args.resblock_updown),
            'use_new_attention_order': eval(args.use_new_attention_order),
        }
        noise_levels = torch.linspace(float(args.t0),
                                      float(args.t),
                                      int(args.interval), device=device) * int(args.interval)
        model = I2SB(unet_params=unet_params,
                     noise_levels=noise_levels,
                     cond=eval(args.cond_x1),
                     )#ckpt_dir=str(args.init_ckpt_dir)

    else:
        raise NotImplementedError

    if hasattr(args, 'init_model') and str(args.init_model) != 'None':
        stdict = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['model']

        model.load_state_dict(stdict)

    if hasattr(args,'reuse') and eval(args.reuse):
        if not hasattr(args, 'init_model') or str(args.init_model) == 'None':
            raise CustomException('reuse flag is True, but init_model was not set')

    model = model.to(device)

    return model


def get_test_model_DF(args, device):
    if args.model_dir is not None:
        if args.eval_metric != 'None':
            eval_metric = str(args.eval_metric)
            if "*" in args.model_dir:
                if hasattr(args, "exp_name"):
                    folder_name = str(args.exp_name)
                else:
                    folder_name = os.path.basename(os.path.dirname(args.conf_file))

                set_dir = os.path.dirname(args.model_dir)
                model_dirs = glob(os.path.join(set_dir, folder_name, "*", "train", eval_metric))

                best_model_dir = None
                best_metrics = 0.0 if eval(args.eval_maximize) else 10.0**13

                for model_dir in model_dirs:
                    try:
                        # open best_result.json
                        with open(f"{model_dir}/best_{eval_metric}_result.json", "r") as f:
                            load_result = json.load(f)
                    except:
                        load_result = None

                    if load_result is not None:
                        # search best result
                        current_val = load_result[f"best {args.eval_metric}"]

                        if eval(args.eval_maximize):
                            if current_val >= best_metrics:
                                best_metrics = current_val
                                best_model_dir = model_dir
                        else:
                            if current_val <= best_metrics:
                                best_metrics = current_val
                                best_model_dir = model_dir
                if best_model_dir is None:
                    raise ValueError('Cannot search best result. Specified trained model')
                else:
                    best_model_path = os.path.join(best_model_dir, f"best_{eval_metric}_model.pth")
                print(f"[Validation score] {str(args.eval_metric)}:{best_metrics}")
            else:
                best_model_dir = args.model_dir
                best_model_path = os.path.join(best_model_dir, f"best_{eval_metric}_model.pth")
        else:
            eval_metric = 'loss'
            if "*" in args.model_dir:
                if hasattr(args, "exp_name"):
                    folder_name = str(args.exp_name)
                else:
                    folder_name = os.path.basename(os.path.dirname(args.conf_file))

                set_dir = os.path.dirname(args.model_dir)
                model_dirs = glob(os.path.join(set_dir, folder_name, "*", "train", eval_metric))

                best_model_dir = None
                best_metrics = 10.0**13

                for model_dir in model_dirs:
                    try:
                        # open best_result.json
                        with open(f"{model_dir}/best_{eval_metric}_result.json", "r") as f:
                            load_result = json.load(f)
                    except:
                        load_result = None

                    if load_result is not None:
                        # search best result
                        current_val = load_result[f"best {eval_metric}"]
                        if current_val <= best_metrics:
                            best_metrics = current_val
                            best_model_dir = model_dir
                if best_model_dir is None:
                    raise ValueError('Cannot search best result. Specified trained model')
                else:
                    best_model_path = os.path.join(best_model_dir, f"best_{eval_metric}_model.pth")
                print(f"[Validation score] {eval_metric}:{best_metrics}")
            else:
                best_model_dir = args.model_dir
                best_model_path = os.path.join(best_model_dir, f"best_{eval_metric}_model.pth")


        print('Load model from {}'.format(best_model_path))
        model = torch.load(best_model_path)
        model.phase = args.phase
        model = model.to(device)
    else:
        raise ValueError('Specified trained model')

    return model, best_model_dir

'''
WSB
'''
def get_train_model_WSB(args, device):

    if eval(args.crop_size) is not None:
        image_size = eval(args.crop_size)[0]
    elif eval(args.resize) is not None:
        image_size = eval(args.resize)[0]
    else:
        image_size = eval(args.image_size)[0]

    in_channels = int(args.in_channels)+int(args.out_channels)
    out_channels = int(args.out_channels)
    if hasattr(args, 'dim_match') and eval(args.dim_match):
        if int(args.in_channels) > int(args.out_channels):
            in_channels = int(args.in_channels) * 2
            out_channels = int(args.in_channels)
        elif int(args.in_channels) < int(args.out_channels):
            in_channels = int(args.out_channels) * 2
            out_channels = int(args.out_channels)

    if args.model == 'i2iwsb':

        attention_ds = []
        for res in eval(args.attn_res):
            attention_ds.append(image_size // int(res))

        unet_params = {
            "image_size": image_size,
            "in_channels": in_channels,
            'num_res_blocks': int(args.num_res_blocks),
            "num_channels": int(args.num_channels),
            "out_channels": out_channels,
            'attention_ds': tuple(attention_ds),
            "dropout": float(args.dropout),
            "channel_mult": eval(args.channel_mults),
            'class_cond': eval(args.class_cond),
            'learn_sigma': eval(args.learn_sigma),
            'use_fp16': eval(args.use_fp16),
            'num_head': int(args.num_head),
            "num_head_channels": int(args.num_head_channels),
            'num_heads_upsample': int(args.num_heads_upsample),
            'use_scale_shift_norm': eval(args.use_scale_shift_norm),
            'resblock_updown': eval(args.resblock_updown),
            'use_new_attention_order': eval(args.use_new_attention_order),
        }
        noise_levels = torch.linspace(float(args.t0),
                                      float(args.t),
                                      int(args.interval), device=device) * int(args.interval)
        model_G = I2SB(unet_params=unet_params,
                     noise_levels=noise_levels,
                     cond=eval(args.cond_x1),
                     )#ckpt_dir=str(args.init_ckpt_dir)

        model_D = Discriminator(input_nc=in_channels,
                                ndf=int(args.ndf),
                                n_layers=int(args.n_layers))
    else:
        raise NotImplementedError

    if hasattr(args, 'init_model') and str(args.init_model) != 'None':
        stdict_G = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['model_G']
        stdict_D = torch.load(f"{str(args.init_model)}/train/last_epoch_object.cpt")['model_D']

        model_G.load_state_dict(stdict_G)
        model_D.load_state_dict(stdict_D)

    if hasattr(args,'reuse') and eval(args.reuse):
        if not hasattr(args, 'init_model') or str(args.init_model) == 'None':
            raise CustomException('reuse flag is True, but init_model was not set')

    model_G = model_G.to(device)
    model_D = model_D.to(device)

    return model_G, model_D


def get_test_model_I2IWSB(args, device):
    if args.model_dir is not None:
        if args.eval_metric != 'None':
            eval_metric = str(args.eval_metric)
            if "*" in args.model_dir:
                if hasattr(args, "exp_name"):
                    folder_name = str(args.exp_name)
                else:
                    folder_name = os.path.basename(os.path.dirname(args.conf_file))

                set_dir = os.path.dirname(args.model_dir)
                model_dirs = glob(os.path.join(set_dir, folder_name, "*", "train", eval_metric))

                best_model_dir = None
                best_metrics = 0.0 if eval(args.eval_maximize) else 10.0**13

                for model_dir in model_dirs:
                    try:
                        # open best_result.json
                        with open(f"{model_dir}/best_{eval_metric}_result.json", "r") as f:
                            load_result = json.load(f)
                    except:
                        load_result = None

                    if load_result is not None:
                        # search best result
                        current_val = load_result[f"best {args.eval_metric}"]

                        if eval(args.eval_maximize):
                            if current_val >= best_metrics:
                                best_metrics = current_val
                                best_model_dir = model_dir
                        else:
                            if current_val <= best_metrics:
                                best_metrics = current_val
                                best_model_dir = model_dir
                if best_model_dir is None:
                    raise ValueError('Cannot search best result. Specified trained model')
                else:
                    print(f"[Validation score] {eval_metric}:{best_metrics}")
                    best_model_path_G = os.path.join(best_model_dir, f"best_{eval_metric}_model_G.pth")
                    best_model_path_D = os.path.join(best_model_dir, f"best_{eval_metric}_model_D.pth")
            else:
                best_model_dir = args.model_dir
                best_model_path_G = os.path.join(best_model_dir, f"best_{eval_metric}_model_G.pth")
                best_model_path_D = os.path.join(best_model_dir, f"best_{eval_metric}_model_D.pth")
        else:
            eval_metric = 'loss'
            if "*" in args.model_dir:
                if hasattr(args, "exp_name"):
                    folder_name = str(args.exp_name)
                else:
                    folder_name = os.path.basename(os.path.dirname(args.conf_file))

                set_dir = os.path.dirname(args.model_dir)
                model_dirs = glob(os.path.join(set_dir, folder_name, "*", "train", eval_metric))

                best_model_dir = None
                best_metrics = 10.0**13

                for model_dir in model_dirs:
                    try:
                        # open best_result.json
                        with open(f"{model_dir}/best_{eval_metric}_result.json", "r") as f:
                            load_result = json.load(f)
                    except:
                        load_result = None

                    if load_result is not None:
                        # search best result
                        current_val = load_result[f"best {eval_metric}"]
                        if current_val <= best_metrics:
                            best_metrics = current_val
                            best_model_dir = model_dir
                if best_model_dir is None:
                    raise ValueError('Cannot search best result. Specified trained model')
                else:
                    print(f"[Validation score] {eval_metric}:{best_metrics}")
                    best_model_path_G = os.path.join(best_model_dir, f"best_{eval_metric}_model_G.pth")
                    best_model_path_D = os.path.join(best_model_dir, f"best_{eval_metric}_model_D.pth")
            else:
                best_model_dir = args.model_dir
                best_model_path_G = os.path.join(best_model_dir, f"best_{eval_metric}_model_G.pth")
                best_model_path_D = os.path.join(best_model_dir, f"best_{eval_metric}_model_D.pth")

        print('Load generator from {}'.format(best_model_path_G))
        print('Load discriminator from {}'.format(best_model_path_D))
        if args.device == 'cpu':
            model_G = torch.load(best_model_path_G, map_location=torch.device('cpu'))
            model_D = torch.load(best_model_path_D, map_location=torch.device('cpu'))
        else:
            model_G = torch.load(best_model_path_G)
            model_D = torch.load(best_model_path_D)
        model_G.phase = args.phase
        model_D.phase = args.phase
        model_G = model_G.to(device)
        model_D = model_D.to(device)
    else:
        raise ValueError('Specified trained model')

    return model_G, model_D, best_model_dir