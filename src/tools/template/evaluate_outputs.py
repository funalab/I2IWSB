import gc
import os
import sys
sys.path.append(os.getcwd())
import json
import random
import pickle
import numpy as np
import pandas as pd
from src.lib.utils.cmd_args import config_paraser
from src.lib.utils.utils import set_seed, CustomException, save_dict_to_json
from tqdm import tqdm
from skimage import io
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops_table, label
from glob import glob
from multiprocessing import Pool
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    return path


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
            if model_name == 'cWGAN-GP':
                folder_name = os.path.basename(p)
                img_path = f"{p}/{folder_name}_channel_{channel_id}.tif"
            else:
                img_path = f"{p}_channel_{channel_id}.tif"
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


def summarize_by_image_id(paths, gt_mode=False):
    out = {}
    for path in paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        pos = filename[:12]
        if gt_mode:
            ch = filename.split('-')[1][:3]
        else:
            ch = filename[-3:]
        id = f"{pos}-{ch}"
        if not id in out.keys():
            out[id] = path
        else:
            raise CustomException(f"Duplicated id detected: {id}, {filename}")
    return out


def plot_dimension_reduction(res, save_dir, name, filename, xlabel, ylabel, pca=None):
    #plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    linewidth = 2

    # add legend
    fig = plt.figure(figsize=(10,10))
    pos_info = np.array(res)
    plt.scatter(pos_info[:, 0], pos_info[:, 1], color='#378CE7', alpha=0.5)

    plt.xlabel(xlabel, fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.grid(linestyle='--', color="lightgray", linewidth=linewidth / 4, alpha=0.5)
    plt.tick_params(labelsize=26)

    ax = plt.gca()
    ax.spines["right"].set_linewidth(linewidth)
    ax.spines["top"].set_linewidth(linewidth)
    ax.spines["left"].set_linewidth(linewidth)
    ax.spines["bottom"].set_linewidth(linewidth)
    ax.set_axisbelow(True)

    plt.savefig(os.path.join(save_dir, f'{filename}.pdf'), bbox_inches="tight", dpi=600)
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), bbox_inches="tight", dpi=600)
    plt.close()

    if name == 'PCA' and pca is not None:
        fig = plt.figure(figsize=(8,8))
        plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), marker="o",markersize=3,
                linestyle='dashed',linewidth=linewidth*0.5, color='k')
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative contribution rate")
        plt.grid(linestyle='--', color="lightgray", linewidth=linewidth / 4, alpha=0.5)

        ax = plt.gca()
        ax.spines["right"].set_linewidth(linewidth)
        ax.spines["top"].set_linewidth(linewidth)
        ax.spines["left"].set_linewidth(linewidth)
        ax.spines["bottom"].set_linewidth(linewidth)
        ax.set_axisbelow(True)

        plt.savefig(os.path.join(save_dir, f'result_PCA_explained_variance_ratio.pdf'), bbox_inches="tight", dpi=600)
        plt.close()


def dimension_reduction(img_dict, save_root):

    for ch in img_dict.keys():
        print("=" * 100)
        print(f'[ch] {ch}')
        print("=" * 100)
        data = img_dict[ch]
        # PCA
        print('-'*100)
        print('PCA...')
        name = 'PCA'
        save_dir = f'{save_root}/{name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        pca = PCA()
        res = pca.fit_transform(data)

        # save data
        with open(f'{save_dir}/pca.pkl', 'wb') as pickle_file:
            pickle.dump(pca, pickle_file)
        df = pd.DataFrame(res)
        df.to_csv(f'{save_dir}/result_{name}.csv',index=False)

        df_ratio = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(df.columns))])
        df_ratio.to_csv(f'{save_dir}/result_{name}explained_variance_ratio.csv')

        # visualize
        filename = f'result_{name}'
        xlabel = '{} axis 1 ({:.1f}%)'.format(name, df_ratio.iloc[0, 0]*100)
        ylabel = '{} axis 2 ({:.1f}%)'.format(name, df_ratio.iloc[1, 0]*100)
        plot_dimension_reduction(res, save_dir, name, filename, xlabel, ylabel, pca=pca)

        #################################
        # t-SNE
        print('-'*100)
        print('t-SNE...')
        name = 't-SNE'
        SEED = 109
        random.seed(SEED)
        np.random.seed(SEED)
        save_dir = check_dir(f'{save_root}/{name}')

        tsne = TSNE(n_components=2, random_state=SEED, n_iter=1000)
        embedded = tsne.fit_transform(data)

        with open(f'{save_dir}/tsne.pkl', 'wb') as pickle_file:
            pickle.dump(tsne, pickle_file)

        df = pd.DataFrame(embedded)
        df.to_csv(f'{save_dir}/result_{name}_rand-{SEED}.csv', index=False)

        # visualize
        filename = f'result_{name}_rand-{SEED}.pdf'
        xlabel = f'{name} axis 1'
        ylabel = f'{name} axis 2'
        plot_dimension_reduction(res, save_dir, name, filename, xlabel, ylabel)


def watershed_segmentation(binary_img, kernel_size, min_distance):
    distance = ndi.distance_transform_edt(binary_img)
    coords = peak_local_max(distance,
                            footprint=np.ones((kernel_size, kernel_size)),
                            labels=binary_img,
                            min_distance=min_distance)#ball(radius=kernel_size)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_img)
    return labels


def evaluate(input):
    # parse input
    img_id, path, save_dir = input
    pos, ch = img_id.split('-')

    # settings
    ksize = 5
    sigma = 1.0
    truncate = ((ksize-1)/2-0.5)/sigma

    # load image
    img_raw = io.imread(path)

    # preprocess
    img = ndi.median_filter(img_raw, size=ksize)
    img = ndi.gaussian_filter(img, sigma=sigma, truncate=truncate)

    # binarize
    thresh = threshold_otsu(img)
    img_binary = img > thresh

    # segmentation
    img_label = watershed_segmentation(binary_img=img_binary, kernel_size=11, min_distance=11)

    # calculate stats
    props = regionprops_table(img_label, properties=['label', 'area'])
    df = pd.DataFrame(props)

    save_dir = check_dir(f'{save_dir}/{pos}')
    df.to_csv(f'{save_dir}/analyzed_label_{ch}.csv', index=False)

    # stats
    stats_dict = {
        'count': float(df['area'].size),
        'ratio': float(df['area'].sum()/img.size),
        'intensity_mean': float(np.mean(img_raw)),
    }

    return img_id, img_raw, stats_dict


def evaluate_main(summarized_path_dict, save_dir, file_name, process_num=16):
    ch_size_dict = {}
    for k in summarized_path_dict.keys():
        pos, ch = k.split('-')
        if not ch in ch_size_dict.keys():
            ch_size_dict[ch] = 1
        else:
            ch_size_dict[ch] += 1

    # run
    out = {}
    out_df = []
    img_dict = {}
    ch_count = {}
    inputs = [[k, v, save_dir] for k, v in summarized_path_dict.items()]
    with Pool(process_num) as p:  # 並列+tqdm
        with tqdm(total=len(inputs)) as pbar:
            for res in p.imap_unordered(evaluate, inputs):
                img_id, img_raw, stats_dict = res
                pos, ch = img_id.split('-')

                val_dict = {'pos': pos, 'ch': ch}
                val_dict = dict(val_dict, **stats_dict)

                out[img_id] = stats_dict
                out_df.append(val_dict)
                if not ch in img_dict.keys():
                    data = np.zeros((ch_size_dict[ch], img_raw.size))
                    ch_count[ch] = 0
                    data[ch_count[ch], :] = img_raw.flatten()
                    img_dict[ch] = data
                else:
                    ch_count[ch] += 1
                    img_dict[ch][ch_count[ch], :] = img_raw.flatten()

                pbar.update(1)

    df = pd.DataFrame(out_df)
    print(df.head())

    # save
    save_dict_to_json(savefilepath=f'{save_dir}/{file_name}.json',
                      data_dict=out)
    df.to_csv(f'{save_dir}/{file_name}.csv', index=False)

    # dimension reduction
    dimension_reduction(img_dict=img_dict, save_root=check_dir(f"{save_dir}/dimension_reduction"))

    del img_dict
    gc.collect()

    return df


def analyze_dataframe(df, save_dir, file_name):
    df_stats_mean = df.drop(columns='pos').groupby('ch').mean()
    df_stats_mean.columns = [f'{d}_mean' for d in df_stats_mean.columns]
    df_stats_std = df.drop(columns='pos').groupby('ch').std()  # 標本標準偏差 ddof=0
    df_stats_std.columns = [f'{d}_std' for d in df_stats_std.columns]

    df_stats = pd.concat([df_stats_mean, df_stats_std], axis=1)
    #df_stats = df_stats.sort_index(axis='columns')
    df_stats.to_csv(f'{save_dir}/{file_name}_stats.csv', index=False)


def evaluate_compare(x, y, savefilepath=None):
    lr = LinearRegression()
    lr.fit(x, y)
    y_pred = lr.predict(x)
    mse = mean_squared_error(y_pred=y_pred, y_true=y)
    mae = mean_absolute_error(y_pred=y_pred, y_true=y)
    r2 = r2_score(y_pred=y_pred, y_true=y)
    res = {'mse': float(mse), 'mae': float(mae), 'r2': float(r2)}

    print('[Results]')
    print(res)

    if savefilepath is not None:
        save_dict_to_json(savefilepath=f"{savefilepath}.json", data_dict=res)


def show_joint(x, y, savefilepath=None, show_mode=False):
    df = pd.DataFrame(np.squeeze(np.array([x, y]).T), columns=['Ground truth', 'Predict'])

    #plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    figsize = [6, 6]

    fig = plt.figure(figsize=figsize)

    g = sns.JointGrid(data=df, x="Ground truth", y="Predict")
    g.plot_joint(sns.scatterplot, color='dodgerblue')
    g.plot_marginals(sns.histplot, kde=True, bins=50, color='orangered')

    if savefilepath is not None:
        plt.savefig(f"{savefilepath}.png", bbox_inches="tight", dpi=600)
        plt.savefig(f"{savefilepath}.pdf", bbox_inches="tight", dpi=600)
    if show_mode:
        plt.show()
    else:
        plt.close()
    del fig


def compare_dataframe(df, df_gt, save_dir_root):

    columns = df.drop(['pos', 'ch'], axis=1).columns.values.tolist()
    for ch in df['ch'].unique().tolist():
        for col in columns:
            x = df[(df['ch'] == ch)][col].tolist()
            y = df_gt[(df_gt['ch'] == ch)][col].tolist()

            save_dir = check_dir(f"{save_dir_root}/{ch}-{col}")
            evaluate_compare(x, y, savefilepath=f"{save_dir}/evaluate_result")
            show_joint(x, y, savefilepath=f"{save_dir}/visualize_joint", show_mode=False)

def main():

    ''' Settings '''
    # Parse config parameters
    args = config_paraser()

    # Set seed
    _ = set_seed(args=args)

    ''' Get image paths '''
    # get img_dir_root
    img_dir_root = get_img_dir_root(args=args)
    print(f'img_dir_root: {img_dir_root}')

    # get each img_path
    print('getting image paths...')
    img_path_list = get_img_path(args=args, img_dir_root=img_dir_root)
    summarized_path_dict = summarize_by_image_id(paths=img_path_list, gt_mode=False)
    print(f'img_path size: {len(img_path_list)}')

    # get ground truth img_path
    print('getting ground truth image paths...')
    img_path_gt_list = get_img_path_gt(args=args)
    summarized_path_gt_dict = summarize_by_image_id(paths=img_path_gt_list, gt_mode=True)
    print(f'img_path size: {len(img_path_gt_list)}')

    ''' Analyze images '''
    # analyze each image
    print('analyze each images...')
    save_dir = check_dir(f'{os.path.dirname(img_dir_root)}/analyze')
    result_df = evaluate_main(summarized_path_dict=summarized_path_dict,
                              save_dir=check_dir(f'{save_dir}/predict'),
                              file_name='analyzed_result',
                              process_num=16)
    print('analyze ground truth each images...')
    result_df_gt = evaluate_main(summarized_path_dict=summarized_path_gt_dict,
                                 save_dir=check_dir(f'{save_dir}/ground_truth'),
                                 file_name='analyzed_result',
                                 process_num=16)

    # analyze dataframe
    print('analyze dataframes...')
    analyze_dataframe(df=result_df, save_dir=check_dir(f'{save_dir}/predict'), file_name='analyzed_result')
    analyze_dataframe(df=result_df_gt, save_dir=check_dir(f'{save_dir}/ground_truth'), file_name='analyzed_result')

    print('compare dataframes...')
    compare_dataframe(df=result_df, df_gt=result_df_gt, save_dir_root=check_dir(f'{save_dir}/compare'))


if __name__ == '__main__':
    main()
