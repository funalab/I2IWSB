import os
import random
import pickle
import pandas as pd
import numpy as np
from skimage import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from PIL import Image
from glob import glob


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    return path


def convert_paths_to_ch_dict(img_path_list, ch_list):
    ch_dict = {ch:[] for ch in ch_list}
    for img_path in img_path_list:
        ch = img_path[-7:-4]
        ch_dict[ch].append(img_path)
    return ch_dict


def is_list_equal(x):
    return len(set(x)) <= 1


def dimension_reduction(img_path_dict, ch_list, data_species, save_root, SEED, resize_shape=(64, 64)):
    print('load images...')
    data_dict = {ch: {} for ch in ch_list}
    data_num_dict = {ch: [] for ch in ch_list}
    for model_name in tqdm(img_path_dict.keys()):
        for ch in ch_list:
            # if not ch in pca_dict:
            #     pca_dict[ch] = None
            # if not ch in tsne_dict:
            #     tsne_dict[ch] = None
            
            img_path_list = img_path_dict[model_name][ch]

            data = []
            for img_path in img_path_list:
                img = io.imread(img_path)
                if img.size > resize_shape[0]*resize_shape[1]:
                    img = Image.fromarray(img)
                    img = img.resize(resize_shape, Image.NEAREST)
                    img = np.array(img)
                data.append(img.flatten())
            data = np.array(data)
            data_num_dict[ch].append(data.shape[0])
            data_dict[ch][model_name] = data
            del data

    data_num = {ch: 0 for ch in ch_list}
    data_num_check = []
    for ch in ch_list:
        if is_list_equal(data_num_dict[ch]):
            data_num[ch] = data_num_dict[ch][0]
            data_num_check.append(data_num_dict[ch][0])
    print('Count is equal?:', is_list_equal(data_num_check))

    with open(f'{save_root}/data_all.pkl', 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

    print('dimension reduction...')
    pca_res_dict = {}
    pca_dict = {}
    tsne_dict = {}
    df_ratio_dict = {}
    data_species_count = len(data_species)
    for ch in ch_list:
        print(f'{ch}')
        data_model = data_dict[ch]
        data = np.zeros((data_num[ch] * data_species_count, resize_shape[0] * resize_shape[1]))
        cnt = 0
        for model_name in tqdm(data_model.keys()):
            data_each_model = data_dict[ch][model_name]
            data[cnt:cnt+data_num[ch],:] = data_each_model
            cnt += data_num[ch]

        save_dir = check_dir(f'{save_root}/{ch}')
        with open(f'{save_dir}/data.pkl', 'wb') as pickle_file:
            pickle.dump(data_dict, pickle_file)

        #################################
        # PCA
        name = 'PCA'
        save_dir = check_dir(f'{save_root}/{ch}/{name}')

        pca = PCA()
        res = pca.fit_transform(data)

        # save data
        df = pd.DataFrame(res)
        df.to_csv(f'{save_dir}/result_{name}.csv',index=False)

        df_ratio = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(df.columns))])
        df_ratio.to_csv(f'{save_dir}/result_{name}explained_variance_ratio.csv')

        with open(f'{save_dir}/pca_res.pkl', 'wb') as pickle_file:
            pickle.dump(res, pickle_file)

        with open(f'{save_dir}/pca.pkl', 'wb') as pickle_file:
            pickle.dump(pca, pickle_file)

        #################################
        # t-SNE
        name = 't-SNE'
        random.seed(SEED)
        np.random.seed(SEED)
        save_dir = check_dir(f'{save_root}/{ch}/{name}/seed-{SEED}')

        tsne = TSNE(n_components=2, random_state=SEED, n_iter=1000)
        embedded = tsne.fit_transform(data)

        with open(f'{save_dir}/tsne.pkl', 'wb') as pickle_file:
            pickle.dump(tsne, pickle_file)

        df = pd.DataFrame(embedded)
        df.to_csv(f'{save_dir}/result_{name}_rand-{SEED}.csv', index=False)

        pca_res_dict[ch] = res
        pca_dict[ch] = pca
        tsne_dict[ch] = embedded
        df_ratio_dict[ch] = df_ratio

    with open(f'{save_root}/pca_res_dict.pkl', 'wb') as pickle_file:
        pickle.dump(pca_res_dict, pickle_file)

    with open(f'{save_root}/pca_dict.pkl', 'wb') as pickle_file:
        pickle.dump(pca_dict, pickle_file)

    with open(f'{save_root}/tsne_dict.pkl', 'wb') as pickle_file:
        pickle.dump(tsne_dict, pickle_file)

    with open(f'{save_root}/data_num_dict.pkl', 'wb') as pickle_file:
        pickle.dump(data_num, pickle_file)

    with open(f'{save_root}/df_ratio_dict.pkl', 'wb') as pickle_file:
        pickle.dump(df_ratio_dict, pickle_file)

    return pca_res_dict, pca_dict, tsne_dict, data_num, df_ratio_dict


def plot_dimension_reduction(res_dict, colors_dict, markers_dict, data_num_dict,
                             data_species, df_ratio_dict, save_dir, name, filename, pca_dict=None):
    chs = list(res_dict.keys())
    save_dir = check_dir(f'{save_dir}/{name}')
    for ch in chs:
        data_num = data_num_dict[ch]
        df_ratio = df_ratio_dict[ch]
        save_dir_ch = check_dir(f'{save_dir}/{ch}')


        if pca_dict is not None:
            xlabel = '{} axis 1 ({:.2f}%)'.format(name, df_ratio.iloc[0, 0] * 100)
            ylabel = '{} axis 2 ({:.2f}%)'.format(name, df_ratio.iloc[1, 0] * 100)
        else:
            xlabel = '{} axis 1'.format(name)
            ylabel = '{} axis 2'.format(name)

        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.size"] = 20
        linewidth = 2

        # add legend
        fig = plt.figure(figsize=(10,10))
        cnt = 0
        res = res_dict[ch]
        for model_name in data_species:
            pos_info = np.array(res)
            plt.scatter(pos_info[cnt:cnt+data_num, 0], pos_info[cnt:cnt+data_num, 1],
                        color=colors_dict[model_name], alpha=0.5, label=model_name,
                        marker=markers_dict[model_name])
            cnt += data_num

        plt.xlabel(xlabel, fontsize=28)
        plt.ylabel(ylabel, fontsize=28)
        plt.tick_params(labelsize=26)
        plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.1),
                   ncol=3, fontsize=26)#

        if pca_dict is not None:
            # plt.xscale('symlog')
            # plt.yscale('symlog')
            # plt.grid(which='major', color='lightgray', linestyle='--', linewidth=linewidth / 4, alpha=0.5)
            # plt.grid(which='minor', color='lightgray', linestyle='--', linewidth=linewidth / 4, alpha=0.5)
            plt.grid(linestyle='--', color="lightgray", linewidth=linewidth / 4, alpha=0.5)
        else:
            plt.grid(linestyle='--', color="lightgray", linewidth=linewidth / 4, alpha=0.5)

        ax = plt.gca()
        ax.spines["right"].set_linewidth(linewidth)
        ax.spines["top"].set_linewidth(linewidth)
        ax.spines["left"].set_linewidth(linewidth)
        ax.spines["bottom"].set_linewidth(linewidth)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        plt.savefig(os.path.join(save_dir_ch, f'{filename}-{ch}.pdf'), bbox_inches="tight", dpi=600)
        plt.savefig(os.path.join(save_dir_ch, f'{filename}-{ch}.png'), bbox_inches="tight", dpi=600)
        plt.close()

        if pca_dict is not None:
                pca_ch = pca_dict[ch]
                fig = plt.figure(figsize=(8,8))
                plt.plot([0] + list(np.cumsum(pca_ch.explained_variance_ratio_)), marker="o",markersize=3,
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

                plt.savefig(os.path.join(save_dir_ch, f'result_PCA_explained_variance_ratio-{ch}.pdf'),
                            bbox_inches="tight", dpi=600)
                plt.savefig(os.path.join(save_dir_ch, f'result_PCA_explained_variance_ratio-{ch}.png'),
                            bbox_inches="tight", dpi=600)
                plt.close()


def main():
    root = os.getcwd()
    exp_name = 'trial_wellplate_epoch100_batch28'
    wsb_paths = glob(f'{root}/results/i2iwsb/{exp_name}/train_fold1*/'
                     f'test/loss/images/*/Predict/*.tif')
    gt_paths = glob(f'{root}/results/wsb/{exp_name}/train_fold1*/'
                    f'test/loss/images/*/GT/*.tif')
    i2sb_paths = glob(f'{root}/results/i2sb/{exp_name}/train_fold5*/'
                      f'test/loss/images/*/Predict/*.tif')
    guided_paths = glob(f'{root}/results/guidedI2I/{exp_name}/train_fold2*/'
                        f'test/loss/images/BR00117011__2020-11-08T19_57_47-Measurement1/Images/*/Out/*.tif')
    palette_paths = glob(f'{root}/results/palette/{exp_name}/train_fold3*/'
                         f'test/loss/images/BR00117011__2020-11-08T19_57_47-Measurement1/Images/*/Out/*.tif')

    print(len(wsb_paths), len(gt_paths), len(i2sb_paths), len(guided_paths), len(palette_paths))
    ch_list = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']

    wsb_dict = convert_paths_to_ch_dict(wsb_paths, ch_list)
    gt_dict = convert_paths_to_ch_dict(gt_paths, ch_list)
    i2sb_dict = convert_paths_to_ch_dict(i2sb_paths, ch_list)
    guided_dict = convert_paths_to_ch_dict(guided_paths, ch_list)
    palette_dict = convert_paths_to_ch_dict(palette_paths, ch_list)

    img_path_dict = {
        'True': gt_dict,
        'Palette': palette_dict,
        'guided-I2I': guided_dict,
        'I$^{2}$SB': i2sb_dict,
        'Ours': wsb_dict,
    }
    data_species = list(img_path_dict.keys())

    SEED = 109
    resize_shape = io.imread(gt_paths[0]).shape #(64, 64)
    print(f'resize shape: {resize_shape}')
    save_root = check_dir(f'{root}/results/{exp_name}/visualize_dimension_reduction')
    save_root = check_dir(f'{save_root}/resize_shape_{resize_shape[0]}-{resize_shape[1]}')

    pca_res_dict, pca_dict, tsne_dict, data_num_dict, df_ratio_dict  = None, None, None, None, None
    flag = 0
    if os.path.exists(f'{save_root}/pca_res_dict.pkl'):
        with open(f'{save_root}/pca_res_dict.pkl', 'rb') as pickle_file:
            pca_res_dict = pickle.load(pickle_file)
            flag += 1

    if os.path.exists(f'{save_root}/pca_dict.pkl'):
        with open(f'{save_root}/pca_dict.pkl', 'rb') as pickle_file:
            pca_dict = pickle.load(pickle_file)
            flag += 1

    if os.path.exists(f'{save_root}/tsne_dict.pkl'):
        with open(f'{save_root}/tsne_dict.pkl', 'rb') as pickle_file:
            tsne_dict = pickle.load(pickle_file)
            flag += 1

    if os.path.exists(f'{save_root}/data_num_dict.pkl'):
        with open(f'{save_root}/data_num_dict.pkl', 'rb') as pickle_file:
            data_num_dict = pickle.load(pickle_file)
            flag += 1

    if os.path.exists(f'{save_root}/df_ratio_dict.pkl'):
        with open(f'{save_root}/df_ratio_dict.pkl', 'rb') as pickle_file:
            df_ratio_dict = pickle.load(pickle_file)
            flag += 1

    # run
    if flag < 5:
        pca_res_dict, pca_dict, tsne_dict, data_num_dict, df_ratio_dict = dimension_reduction(img_path_dict,
                                                                               ch_list,
                                                                               data_species,
                                                                               save_root,
                                                                               SEED,
                                                                               resize_shape)

    # visualize
    print('Visualization...')
    save_dir = check_dir(f'{save_root}/figures')
    colors_dict = {
        'True': 'red',#'#000000', # black
        'Palette': '#009E73', # green
        'guided-I2I': '#0072B2', # blue
        'I$^{2}$SB': '#D55E00', # orange
        'Ours': 'purple', # pink
    }
    markers_dict = {
        'True': '^',
        'Palette': '<',
        'guided-I2I': '>',
        'I$^{2}$SB': 'v',
        'Ours': 'o',
    }

    # PCA
    name = 'PCA'
    filename = f'result_{name}'
    plot_dimension_reduction(pca_res_dict, colors_dict, markers_dict, data_num_dict,
                             data_species, df_ratio_dict, save_dir, name, filename, pca_dict)

    # t-SNE
    name = 't-SNE'
    filename = f'result_{name}_rand-{SEED}'
    plot_dimension_reduction(tsne_dict, colors_dict, markers_dict, data_num_dict,
                             data_species, df_ratio_dict, save_dir, name, filename)
    print('finish')


if __name__ == '__main__':
    main()