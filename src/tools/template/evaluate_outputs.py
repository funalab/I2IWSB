import os
import sys
sys.path.append(os.getcwd())
import json
import numpy as np
import pandas as pd
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, CustomException
from tqdm import tqdm
from skimage import io
from glob import glob
from multiprocessing import Pool


def save_dict_to_json(savefilepath, data_dict):

    with open(savefilepath, "w") as f:
        json.dump(data_dict, f)


def get_img_dir_root(args):
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
                best_metrics = 0.0 if eval(args.eval_maximize) else 10.0 ** 13

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
                best_model_dir = args.model_dir
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
                best_metrics = 10.0 ** 13

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
                best_model_dir = args.model_dir
    else:
        raise ValueError('Specified trained model')

    save_dir = best_model_dir.replace("/train/", "/test/")
    img_dir_root = f"{save_dir}/images"
    print(f'[img dir root] {img_dir_root}')

    return img_dir_root


def get_img_path(args, img_dir_root):  # f"{img_dir}/{p}_channel_{channel_id}.tiff"
    model_name = str(args.model)
    if model_name == 'cWGAN-GP':
        img_dir_each = glob(f"{img_dir_root}/*")
    elif model_name == 'guided-I2I' or model_name == 'Palette':
        img_dir_each = glob(f"{img_dir_root}/*/Images/*/Out")
    elif model_name == 'I2SB' or model_name == 'cWSB-GP':
        img_dir_each = glob(f"{img_dir_root}/*/Predict")
    else:
        raise NotImplementedError

    with open(f"{args.dataset_path}/{args.output_channel_path}", 'r') as f:
        channel_id_list = [line.rstrip() for line in f]

    img_path_list = []
    for p in img_dir_each:
        for channel_id in channel_id_list:
            img_path = f"{p}_channel_{channel_id}.tiff"
            img_path_list.append(img_path)

    return img_path_list


def get_img_path_gt(args):  # f"{args.root_path}/images/{p}-{channel_id}sk1fk1fl1.tiff"
    with open(f"{args.dataset_path}/{args.split_list_test}", 'r') as f:
        img_dir_root_gt_path_list = [line.rstrip() for line in f]
    with open(f"{args.dataset_path}/{args.output_channel_path}", 'r') as f:
        channel_id_list = [line.rstrip() for line in f]

    img_dir_root_gt_path_list = [f"{args.root_path}/images/{p}" for p in img_dir_root_gt_path_list]
    img_path_gt_list = []
    for p in img_dir_root_gt_path_list:
        for channel_id in channel_id_list:
            img_path_gt = f"{p}-{channel_id}sk1fk1fl1.tiff"
            img_path_gt_list.append(img_path_gt)
    return img_path_gt_list


def summarize_by_image_id(paths):
    out = {}
    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        pos = filename[:12]
        ch = filename[-3:]
        id = f"{pos}-{ch}"
        if not id in out.keys():
            out[id] = path
        else:
            raise CustomException(f"Duplicated id detected: {id}")
    return out


def evaluate(input):
    img_id, path = input
    # load img
    img = io.imread(path)

    val_list = []

    return img_id, [float(v) for v in val_list]


def evaluate_main(summarized_path_dict, save_dir, file_name, process_num=16):
    # run
    out = {}
    out_df = []
    inputs = [[k, v] for k, v in summarized_path_dict.items()]
    with Pool(process_num) as p:  # 並列+tqdm
        with tqdm(total=len(inputs)) as pbar:
            for res in p.imap_unordered(evaluate, inputs):
                img_id, val_list = res
                pos, ch = img_id.split('-')

                out[img_id] = val_list
                val_dict = {'pos': pos, 'ch': ch}
                val_dict_list = {f'val_{str(i).zfill(2)}': v for i, v in enumerate(val_list)}
                val_dict = dict(val_dict, **val_dict_list)
                out_df.append(val_dict)
                pbar.update(1)

    df = pd.DataFrame.from_dict(val_dict)

    # save
    save_dict_to_json(savefilepath=f'{save_dir}/{file_name}.json',
                      data_dict=out)
    df.to_csv(f'{save_dir}/{file_name}.csv', index=False)

    return df


def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    _ = set_seed(args=args)

    ''' Get image paths '''
    # get img_dir_root
    img_dir_root = get_img_dir_root(args=args)


    # get each img_path
    img_path_list = get_img_path(args=args, img_dir_root=img_dir_root)
    summarized_path_dict = summarize_by_image_id(paths=img_path_list)

    # get ground truth img_path
    img_path_gt_list = get_img_path_gt(args=args)
    summarized_path_gt_dict = summarize_by_image_id(paths=img_path_gt_list)

    ''' Analyze images '''
    # analyze each image
    result_df = evaluate_main(summarized_path_dict=summarized_path_dict,
                              save_dir=os.path.dirname(img_dir_root),
                              file_name='analyzed_result',
                              process_num=16)
    result_df_gt = evaluate_main(summarized_path_dict=summarized_path_gt_dict,
                                 save_dir=os.path.dirname(img_dir_root),
                                 file_name='analyzed_result_gt',
                                 process_num=16)

    # analyze


if __name__ == '__main__':
    main()
