import os
import imageio
import json
import pytz
import torch
import shutil
import random
import numpy as np
from datetime import datetime
from skimage import io


class CustomException(Exception):
    pass

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def create_train_dir(args):
    current_datetime = datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y%m%d-%H%M%S')
    if hasattr(args, 'reuse') and eval(args.reuse):
        model_dir_path = str(args.init_model)
        model_dir = os.path.dirname(model_dir_path)
        model_dir_name = os.path.basename(model_dir_path)
        save_dir_root = check_dir(f"{model_dir}/{model_dir_name}_reuse_{str(current_datetime)}")
        shutil.copytree(model_dir_path, save_dir_root, dirs_exist_ok=True)
        save_dir = check_dir(f"{save_dir_root}/train")
    else:
        if hasattr(args, "exp_name"):
            folder_name = str(args.exp_name)
        else:
            folder_name = os.path.basename(os.path.dirname(args.conf_file))
        filename = os.path.splitext(os.path.basename(args.conf_file))[0]
        save_dir = check_dir(os.path.join(args.save_dir, folder_name, f"{filename}_{str(current_datetime)}", 'train'))
        shutil.copy(args.conf_file, os.path.join(save_dir, os.path.basename(args.conf_file)))
    return save_dir


def create_test_dir(args, best_model_dir):
    if eval(args.save_dir) is not None:
        save_dir = check_dir(args.save_dir)
    else:
        relpaces_path = best_model_dir.replace("/train/","/test/")
        save_dir = check_dir(relpaces_path)
    shutil.copy(args.conf_file, os.path.join(save_dir, os.path.basename(args.conf_file)))
    return save_dir

def set_seed(args):
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if hasattr(args, 'reuse') and eval(args.reuse):
        model_dir_path = str(args.init_model)
        last_data = torch.load(f'{model_dir_path}/train/last_epoch_object.cpt')
        # final modelでsaveしたrandom_stateに関して，dataloaderのgeneratorのみここで指定
        generator = torch.Generator()
        generator.set_state(last_data['torch_generator_random_state'])
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return generator

def set_seed_for_cWGAN_GP(args):
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if hasattr(args, 'reuse') and eval(args.reuse):
        model_dir_path = str(args.init_model)
        last_data = torch.load(f'{model_dir_path}/train/last_epoch_object.cpt')

        random.setstate(last_data["random_state"])
        np.random.set_state(last_data['np_random_state'])
        torch.random.set_rng_state(last_data["torch_random_state"])
        if 'torch_cuda_random_state' in last_data:
            torch.cuda.set_rng_state(last_data['torch_cuda_random_state'])

        generator = torch.Generator()
        generator.set_state(last_data['torch_generator_random_state'])
    else:
        generator = torch.Generator()
        generator.manual_seed(seed)
        generator_for_cWGAN_GP = torch.Generator()
        generator_for_cWGAN_GP.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return generator

def get_device(args):
    if "cuda" in args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device[-1]
        device = torch.device('cuda')
    else:
        device = torch.device(str(args.device))
    return device


def print_args(dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        dataset_args    : Argument Namespace object for loading dataset
        model_args      : Argument Namespace object for Generator and Discriminator
        updater_args    : Argument Namespace object for Updater
        runtime_args    : Argument Namespace object for runtime parameters
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    print('============================')
    print('[Dataset]')
    for k, v in dataset_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Model]')
    for k, v in model_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Updater]')
    for k, v in updater_dict.items():
        print('%s = %s' % (k, v))
    print('\n[Runtime]')
    for k, v in runtime_dict.items():
        print('%s = %s' % (k, v))
    print('============================\n')


def export_to_config(save_dir, dataset_args, model_args, updater_args, runtime_args):
    """ Export config file
    Args:
        save_dir (str)      : /path/to/save_dir
        dataset_args (dict) : Dataset arguments
        model_args (dict)   : Model arguments
        updater_args (dict) : Updater arguments
        runtime_args (dict) : Runtime arguments
    """
    dataset_dict = {k: v for k, v in vars(dataset_args).items() if v is not None}
    model_dict = {k: v for k, v in vars(model_args).items() if v is not None}
    updater_dict = {k: v for k, v in vars(updater_args).items() if v is not None}
    runtime_dict = {k: v for k, v in vars(runtime_args).items() if v is not None}
    with open(os.path.join(save_dir, 'parameters.cfg'), 'w') as txt_file:
        txt_file.write('[Dataset]\n')
        for k, v in dataset_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Model]\n')
        for k, v in model_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Updater]\n')
        for k, v in updater_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[Runtime]\n')
        for k, v in runtime_dict.items():
            txt_file.write('%s = %s\n' % (k, v))
        txt_file.write('\n[MN]\n')

'''
labels
'''


def convert_rgb_to_label(
        rgb,
        table_label,
        table_artifact,
        flag_artifact=True
):
    """
    Args:
        rgb (np.ndarray)            : Input 2d label image array (np.uint8) [y, x, RGB]
        table_label (list)          : Input 2d label list [class, RGB]
        table_label (list)          : Input 2d artifact label list [class, RGB]
        flag_artifact (bool)        : Boolian flag for including archifact regions
    """

    label = np.zeros(np.shape(rgb)[:2]).astype(np.int64)
    for tl in table_label:
        label += \
            ((rgb[:, :, 0] == tl[0]) * (rgb[:, :, 1] == tl[1]) * \
             (rgb[:, :, 2] == tl[2])) * (table_label.index(tl) + 1)
    if flag_artifact:
        for ta in table_artifact:
            label += \
                ((rgb[:, :, 0] == ta[0]) * (rgb[:, :, 1] == ta[1]) * \
                 (rgb[:, :, 2] == ta[2])) * (len(table_label) + 1)
    else:
        for ta in table_artifact:
            label += \
                ((rgb[:, :, 0] == ta[0]) * (rgb[:, :, 1] == ta[1]) * \
                 (rgb[:, :, 2] == ta[2])) * (len(table_label) + table_artifact.index(ta) + 1)
    return label.astype(np.uint8)


def convert_label_to_rgb(
        label,
        table_label,
        table_artifact,
        flag_artifact=True
):
    """
    Args:
        label (np.ndarray)          : Input 2d label image array (np.uint8) [y, x, RGB]
        table_label (list)          : Input 2d label list [class, RGB]
        table_label (list)          : Input 2d artifact label list [class, RGB]
        flag_artifact (bool)        : Boolian flag for including archifact regions
    """

    rgb = np.zeros((np.shape(label)[0], np.shape(label)[1], 3)).astype(np.int64)
    for tl in table_label:
        rgb[label == (table_label.index(tl) + 1)] += tl
    if flag_artifact:
        rgb[label == (len(table_label) + 1)] += table_artifact[0]
    else:
        for ta in table_artifact:
            rgb[label == (len(table_label) + table_label.index(ta) + 1)] += ta
    return rgb.astype(np.uint8)


def parse_labels(label: np.ndarray, output_channel_names: list, data_range: int, image_dtype: str):
    output_channel_num = len(output_channel_names)
    data = {ind: np.zeros_like(label) for ind in range(output_channel_num)}
    for ind in range(output_channel_num):
        label_num = ind + 1
        parse_img = np.where(label == label_num, data_range, 0)
        parse_img = parse_img.astype(image_dtype)
        data[ind] = parse_img
    return data


def convert_channels_to_rgbs(images: np.ndarray, table_label: list, table_artifact: list, flag_artifact=True,
                             data_range=255, image_dtype='uint8'):
    def _cal_intensity(img: np.array, data_range: int, val: np.array):
        h, w = img.shape
        img_normalized = img / data_range
        intensity_img = np.zeros((h, w, 3))

        for i in range(h):
            for j in range(w):
                intensity_img[i, j, :] = img_normalized[i, j] * val
        return intensity_img

    def _set_rgb_val(ind: int, table_label: list, table_artifact: list, flag_artifact: bool):
        if table_label is not None and ind >= len(table_label):  # artifact
            if flag_artifact:
                val = table_artifact[0]
            else:
                val = table_artifact[ind - len(table_label)]
        else:
            val = table_label[ind]
        val = np.array(val)
        return val

    def _set_image_to_box(box: dict, img: np.ndarray):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                val = img[i, j, :]
                box[i][j].append(val)
        return box

    channels_num = images.shape[2]
    box = dict()
    for i in range(images.shape[0]):
        box[i] = dict()
        for j in range(images.shape[1]):
            box[i][j] = []

    # convert_channels
    for ind in range(channels_num):
        img = images[:, :, ind]  # grayscale
        val = _set_rgb_val(ind=ind, table_label=table_label, table_artifact=table_artifact, flag_artifact=flag_artifact)
        intensity_img = _cal_intensity(img=img, data_range=data_range, val=val)
        box = _set_image_to_box(box=box, img=intensity_img)

    # calculate mean
    out = np.zeros((images.shape[0], images.shape[1], 3))
    for i in range(images.shape[0]):
        for j in range(images.shape[1]):
            val = np.array(box[i][j])
            out[i, j, :] = np.mean(val, axis=0)

    # rescale
    out_rescale = out * (data_range / np.max(out))
    out_rescale = out_rescale.astype(image_dtype)
    return out_rescale

def save_image_function(save_dir, filename, img):
    if img.dtype == 'uint16':
        imageio.imwrite(f"{save_dir}/{filename}.png", img)
    else:
        io.imsave(f"{save_dir}/{filename}.png",
                  img,
                  check_contrast=False)