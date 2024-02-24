import torch.nn as nn
def get_train_args_GAN(args, optimizer_G, optimizer_D, scheduler_G, scheduler_D, save_dir, device):
    trainer_args = {
        'model': str(args.model),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'scheduler_G': scheduler_G,
        'scheduler_D': scheduler_D,
        'epoch': int(args.epoch),
        'save_dir': save_dir,
        'eval_metrics': eval(args.eval_metrics),
        'eval_maximize': eval(args.eval_maximize),
        'device': device,
        'seed': int(args.seed),
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
    }

    if 'save_model_freq' in args:
        trainer_args["save_model_freq"] = int(args.save_model_freq)
    if 'gen_freq' in args:
        trainer_args["gen_freq"] = int(args.gen_freq)

    if str(args.model) == 'cWGAN-GP':
        trainer_args['lamb'] = str(args.lamb)  # gradient penaltyの強さ
        trainer_args['batch_size'] = int(args.batchsize)

    if hasattr(args, 'reuse') and eval(args.reuse):  # 追加で学習を行うかどうか
        trainer_args['reuse'] = eval(args.reuse)

    return trainer_args


def get_train_args_DF(args, optimizer, scheduler, save_dir, device):
    trainer_args = {
        'model': str(args.model),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'batch_size': int(args.batchsize),
        'epoch': int(args.epoch),
        'save_dir': save_dir,
        'eval_metrics': eval(args.eval_metrics),
        'eval_maximize': eval(args.eval_maximize),
        'device': device,
        'seed': int(args.seed),
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
    }

    if str(args.model) == 'guided-I2I' or str(args.model) == 'Palette':
        trainer_args['ema_start'] = int(args.ema_start)
        trainer_args['ema_iter'] = int(args.ema_iter)
        trainer_args['ema_decay'] = float(args.ema_decay)
        trainer_args['task'] = str(args.task)
        trainer_args['sample_num'] = int(args.sample_num)
    elif str(args.model) == 'I2SB':
        trainer_args['interval'] = int(args.interval)
        trainer_args['t0'] = float(args.t0)
        trainer_args['T'] = float(args.t)
        trainer_args['cond_x1'] = eval(args.cond_x1)
        trainer_args['add_x1_noise'] = eval(args.add_x1_noise)
        trainer_args['use_fp16'] = eval(args.use_fp16)
        trainer_args['ema'] = float(args.ema)
        trainer_args['global_size'] = int(args.global_size)
        trainer_args['microbatch'] = int(args.microbatch)
        trainer_args['ot_ode'] = eval(args.ot_ode)
        trainer_args['beta_max'] = float(args.beta_max)
        if hasattr(args, 'val_per_epoch'):
            trainer_args['val_per_epoch'] = int(args.val_per_epoch)
        if hasattr(args, 'print_train_loss_per_epoch'):
            trainer_args['print_train_loss_per_epoch'] = int(args.print_train_loss_per_epoch)
    else:
        raise NotImplementedError(f"{str(args.model)}")

    if 'save_model_freq' in args:
        trainer_args["save_model_freq"] = int(args.save_model_freq)

    return trainer_args

def get_test_args_DF(args, save_dir, device, test_dataset):
    tester_args = {
        'model': str(args.model),
        'save_dir': save_dir,
        'file_list': test_dataset.filepath_list,
        'in_channels': int(args.in_channels),
        'out_channels': int(args.out_channels),
        'input_channel_list': test_dataset.input_channel_list,
        'output_channel_list': test_dataset.output_channel_list,
        'batch_size': int(args.val_batchsize),
        'device': device,
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
        'image_save': eval(args.image_save),
        }
    if hasattr(args, "crop_size"):
        tester_args['crop_size'] = eval(args.crop_size)

    if str(args.model) == 'guided-I2I' or str(args.model) == 'Palette':
        tester_args['task'] = str(args.task)
        tester_args['sample_num'] = int(args.sample_num)
        if hasattr(args, "lossfun"):
            tester_args['lossfun'] = eval(args.lossfun)

    elif str(args.model) == 'I2SB':
        tester_args['cond_x1'] = eval(args.cond_x1)
        tester_args['add_x1_noise'] = eval(args.add_x1_noise)
        tester_args['ot_ode'] = eval(args.ot_ode)
        tester_args['interval'] = int(args.interval)
        tester_args['beta_max'] = float(args.beta_max)
        tester_args['ema'] = float(args.ema)
    else:
        raise NotImplementedError(f"{str(args.model)}")

    if hasattr(args, 'table_artifact'):
        tester_args['table_artifact'] = eval(args.table_artifact)
    if hasattr(args, 'table_label'):
        tester_args['table_label'] = eval(args.table_label)

    if hasattr(args, 'dim_match'):
        tester_args['dim_match'] = eval(args.dim_match)
        if hasattr(args, 'input_dim_label'):
            tester_args['input_dim_label'] = eval(args.input_dim_label)
        if hasattr(args, 'output_dim_label'):
            tester_args['output_dim_label'] = eval(args.output_dim_label)

    return tester_args

def get_train_args(args, optimizer, scheduler, save_dir, device):
    trainer_args = {
        'model': str(args.model),
        'optimizer': optimizer,
        'scheduler': scheduler,
        'batch_size': int(args.batchsize),
        'epoch': int(args.epoch),
        'save_dir': save_dir,
        'eval_metrics': eval(args.eval_metrics),
        'eval_maximize': eval(args.eval_maximize),
        'device': device,
        'seed': int(args.seed),
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
    }

    if 'save_model_freq' in args:
        trainer_args["save_model_freq"] = int(args.save_model_freq)

    return trainer_args


def get_test_args(args, save_dir, device, test_dataset):
    tester_args = {
        'model': str(args.model),
        'save_dir': save_dir,
        'file_list': test_dataset.filepath_list,
        'input_channel_list': test_dataset.input_channel_list,
        'output_channel_list': test_dataset.output_channel_list,
        'batch_size': int(args.val_batchsize),
        'device': device,
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
        'image_save': eval(args.image_save),
        }
    if hasattr(args, "crop_size"):
        tester_args['crop_size'] = eval(args.crop_size)

    if hasattr(args, 'table_artifact'):
        tester_args['table_artifact'] = eval(args.table_artifact)
    if hasattr(args, 'table_label'):
        tester_args['table_label'] = eval(args.table_label)

    if str(args.model) == 'cWGAN-GP':
        tester_args['lamb'] = str(args.lamb)  # gradient penaltyの強さ


    return tester_args

def get_train_args_WSB(args, optimizer_G, optimizer_D, scheduler_G, scheduler_D, save_dir, device):
    trainer_args = {
        'model': str(args.model),
        'optimizer_G': optimizer_G,
        'optimizer_D': optimizer_D,
        'scheduler_G': scheduler_G,
        'scheduler_D': scheduler_D,
        'epoch': int(args.epoch),
        'save_dir': save_dir,
        'eval_metrics': eval(args.eval_metrics),
        'eval_maximize': eval(args.eval_maximize),
        'device': device,
        'seed': int(args.seed),
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
    }

    if str(args.model) == 'cWSB-GP':
        # cWGAN-GP
        trainer_args['lamb'] = str(args.lamb)  # gradient penaltyの強さ
        trainer_args['batch_size'] = int(args.batchsize)
        if 'save_model_freq' in args:
            trainer_args["save_model_freq"] = int(args.save_model_freq)
        if 'gen_freq' in args:
            trainer_args["gen_freq"] = int(args.gen_freq)

        # I2SB
        trainer_args['interval'] = int(args.interval)
        trainer_args['t0'] = float(args.t0)
        trainer_args['T'] = float(args.t)
        trainer_args['cond_x1'] = eval(args.cond_x1)
        trainer_args['add_x1_noise'] = eval(args.add_x1_noise)
        trainer_args['use_fp16'] = eval(args.use_fp16)
        trainer_args['ema'] = float(args.ema)
        trainer_args['global_size'] = int(args.global_size)
        trainer_args['microbatch'] = int(args.microbatch)
        trainer_args['ot_ode'] = eval(args.ot_ode)
        trainer_args['beta_max'] = float(args.beta_max)
        if hasattr(args, 'val_per_epoch'):
            trainer_args['val_per_epoch'] = int(args.val_per_epoch)
        if hasattr(args, 'print_train_loss_per_epoch'):
            trainer_args['print_train_loss_per_epoch'] = int(args.print_train_loss_per_epoch)
    else:
        raise NotImplementedError(f"{str(args.model)}")

    if hasattr(args, 'reuse') and eval(args.reuse):  # 追加で学習を行うかどうか
        trainer_args['reuse'] = eval(args.reuse)

    return trainer_args

def get_test_args_WSB(args, save_dir, device, test_dataset):
    tester_args = {
        'model': str(args.model),
        'save_dir': save_dir,
        'file_list': test_dataset.filepath_list,
        'in_channels': int(args.in_channels),
        'out_channels': int(args.out_channels),
        'input_channel_list': test_dataset.input_channel_list,
        'output_channel_list': test_dataset.output_channel_list,
        'batch_size': int(args.val_batchsize),
        'device': device,
        'image_dtype': str(args.image_dtype),
        'data_range': int(args.data_range),
        'normalization': str(args.normalization),
        'image_save': eval(args.image_save),
        }
    if hasattr(args, "crop_size"):
        tester_args['crop_size'] = eval(args.crop_size)

    if hasattr(args, 'table_artifact'):
        tester_args['table_artifact'] = eval(args.table_artifact)
    if hasattr(args, 'table_label'):
        tester_args['table_label'] = eval(args.table_label)

    if hasattr(args, 'dim_match'):
        tester_args['dim_match'] = eval(args.dim_match)
        if hasattr(args, 'input_dim_label'):
            tester_args['input_dim_label'] = eval(args.input_dim_label)
        if hasattr(args, 'output_dim_label'):
            tester_args['output_dim_label'] = eval(args.output_dim_label)

    if str(args.model) == 'cWSB-GP':
        # cWGAN-GP
        tester_args['lamb'] = str(args.lamb)  # gradient penaltyの強さ

        # I2SB
        tester_args['cond_x1'] = eval(args.cond_x1)
        tester_args['add_x1_noise'] = eval(args.add_x1_noise)
        tester_args['ot_ode'] = eval(args.ot_ode)
        tester_args['interval'] = int(args.interval)
        tester_args['beta_max'] = float(args.beta_max)
        tester_args['ema'] = float(args.ema)
    else:
        raise NotImplementedError(f"{str(args.model)}")

    return tester_args