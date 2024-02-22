import os
import json
import torch
from glob import glob
from PIL import Image
from skimage import io
from multiprocessing import Pool
from skimage.color import rgb2gray
from torch.utils.data import Dataset
from src.lib.datasets.augmentations import *
from src.lib.utils.utils import CustomException



class DatasetWrapper(Dataset):
    def __init__(self,  dataset_name, root_path, dataset_path, split_list, input_channel_path, output_channel_path,
                 channel_table_path, resize, convert_gray, crop_size, crop_range, crop_augmentation, rotation_augmentation,
                 normalization, image_dtype, data_range, in_channels, out_channels, image_size, model_name,
                 dim_match, input_dim_label, output_dim_label, concat_channels_process_num):
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.dataset_path = dataset_path
        with open(f"{self.dataset_path}/{split_list}", 'r') as f:
            self.filepath_list = [line.rstrip() for line in f]
        with open(f"{self.dataset_path}/{input_channel_path}", 'r') as f:
            self.input_channel_list = [line.rstrip() for line in f]
        with open(f"{self.dataset_path}/{output_channel_path}", 'r') as f:
            self.output_channel_list = [line.rstrip() for line in f]
        with open(f"{self.dataset_path}/{channel_table_path}", 'r') as f:
            self.channel_table = json.load(f)
        self.resize = resize
        self.convert_gray = convert_gray
        self.crop_size = crop_size
        self.crop_range = crop_range
        self.crop_augmentation = crop_augmentation
        self.rotation_augmentation = rotation_augmentation
        self.normalization = normalization
        self.image_dtype = image_dtype
        self.data_range = data_range
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.model_name = model_name
        self.dim_match = dim_match
        self.input_dim_label = input_dim_label
        self.output_dim_label = output_dim_label
        self.concat_channels_process_num = concat_channels_process_num

    def __len__(self):
        return len(self.filepath_list)

    def _load_img(self, i, mode):
        def _correct_image(root_path, filepath, channel_id, channel_table):
            img_path = f"{root_path}/images/{filepath}-{channel_id}sk1fk1fl1.tiff"
            id = filepath[i][:10]
            channel_name = channel_table[channel_id]
            correction_path = f"{root_path}/illum/{id}/{id}_Illum{channel_name}.npy"

            # Illumination correction
            img = io.imread(img_path)
            correction_function = np.load(correction_path, allow_pickle=True)
            img_corrected = img / correction_function
            return img_corrected

        if self.dataset_name == 'JUMP':
            # set channel list
            if mode == 'input':
                channel_list = self.input_channel_list
            elif mode == 'output':
                channel_list = self.output_channel_list
            else:
                raise CustomException(f"Invalid image mode : {mode}")

            # concat images
            image_list = [_correct_image(root_path=self.root_path,
                                         filepath=self.filepath_list[i],
                                         channel_id=channel_id,
                                         channel_table=self.channel_table)
                          for channel_id in channel_list]

            image = np.stack(image_list, axis=2)
        else:
            raise NotImplementedError

        return image

    def _check_shape(self, image):
        if len(image.shape) == 2:  # gray
            image = np.expand_dims(image, axis=2)
            gray_flag = True
        elif len(image.shape) == 3:  # color or multicolor
            gray_flag = False
        else:
            raise CustomException(f"Invalid image shape : {image.shape}")
        return image, gray_flag

    def _rgb_to_gray(self, image):
        image = rgb2gray(image) * self.data_range
        image = image.astype(self.image_dtype)
        image = np.expand_dims(image, axis=2)
        return image

    def _resize(self, image, gray_flag):
        if self.convert_gray or gray_flag:
            image = np.squeeze(image)

        channel = image.shape[2]
        out = np.zeros((self.resize[0], self.resize[1], channel))
        for c in range(channel):
            image_c = Image.fromarray(image[:,:,c])
            if image_c.size[0] > self.resize[0] or image_c.size[1] > self.resize[1]:  # 縮小
                image_c = image_c.resize((self.resize[1], self.resize[0]))
            elif image_c.size[0] < self.resize[0] or image_c.size[1] < self.resize[1]:  # 拡大
                image_c = image_c.resize((self.resize[1], self.resize[0]), resample=Image.BICUBIC)
            out[:,:,c] = np.array(image_c)

        if self.convert_gray or gray_flag:
            out = np.expand_dims(out, axis=2)
        return out

    def _dim_matching(self, image_source, image_target):
        dim_source = image_source.shape[0]
        dim_target = image_target.shape[0]

        if dim_source > dim_target and self.output_dim_label is not None:
            image_target_add_dims = [image_target]
            add_dim_num = dim_source - dim_target
            for i in range(add_dim_num):
                label = self.output_dim_label[dim_target + i]
                image_target_add_label = torch.zeros(image_target.shape[1], image_target.shape[2])
                for k in range(len(label)):
                    image_target_add_label += image_target[k, :, :] * (label[k] / self.data_range)
                image_target_add_label = torch.unsqueeze(image_target_add_label, dim=0)
                image_target_add_dims.append(image_target_add_label)

            image_target_expand = torch.cat(image_target_add_dims, axis=0)
            return image_source, image_target_expand

        elif dim_source < dim_target and self.input_dim_label is not None:
            image_source_add_dims = [image_source]
            add_dim_num = dim_target - dim_source
            for i in range(add_dim_num):
                label = self.input_dim_label[dim_source + i]
                image_source_add_label = torch.zeros(image_source.shape[1], image_source.shape[2])
                for k in range(len(label)):
                    image_source_add_label += image_source[k, :, :] * (label[k] / self.data_range)
                image_source_add_label = torch.unsqueeze(image_source_add_label, dim=0)
                image_source_add_dims.append(image_source_add_label)

            image_source_expand = torch.cat(image_source_add_dims, axis=0)
            return image_source_expand, image_target
        else:
            return image_source, image_target

    def get_image(self, i):
        # load image
        image_source = self._load_img(i, mode='input')
        image_target = self._load_img(i, mode='output')

        # check shape
        image_source, gray_flag_source = self._check_shape(image_source)
        image_target, gray_flag_target = self._check_shape(image_target)

        # convert gray if True
        if self.convert_gray:
            image_source = self._rgb_to_gray(image_source)
            image_target = self._rgb_to_gray(image_target)

        # resize
        if self.resize is not None:
            image_source = self._resize(image_source, gray_flag_source)
            image_target = self._resize(image_target, gray_flag_target)

        # crop augmentation
        if self.crop_size is not None:
            image_source, top, left = image_crop(image=image_source,
                                                 crop_size=self.crop_size,
                                                 crop_range=self.crop_range,
                                                 augmentation=self.crop_augmentation,
                                                 top=None,
                                                 left=None)
            image_target, _, _ = image_crop(image=image_target,
                                            crop_size=self.crop_size,
                                            crop_range=self.crop_range,
                                            augmentation=self.crop_augmentation,
                                            top=top,
                                            left=left)

        # rotation augmentation
        if self.rotation_augmentation:
            image_source, rot_flag, flip_flag = image_roration(image=image_source, rot_flag=None, flip_flag=None)
            image_target, _, _ = image_roration(image=image_target, rot_flag=rot_flag, flip_flag=flip_flag)
        # normalization
        if self.normalization == 'minmax':
            image_source = min_max_normalize_one_image(image_source)
            image_target = min_max_normalize_one_image(image_target)
        elif self.normalization == 'zscore':
            raise NotImplementedError
            # image_source = zscore_normalize_one_image(image_source)
            # image_target = zscore_normalize_one_image(image_target)
        elif self.normalization == 'std':
            image_source = standardize_one_image(image_source)
            image_target = standardize_one_image(image_target)
        else:
            raise NotImplementedError

        # check channels
        assert image_source.shape[2] == self.in_channels, \
            f"Invalid input image channels: settings {self.in_channels}, current {image_source.shape[2]}"
        assert image_target.shape[2] == self.out_channels, \
            f"Invalid output image channels: settings {self.out_channels}, current {image_target.shape[2]}"

        # transpose for Tensor
        assert image_source.shape[0] == image_source.shape[1], f"Invalid source image shape: {image_source.shape}"
        assert image_target.shape[0] == image_target.shape[1], f"Invalid target image shape: {image_source.shape}"
        assert image_source.shape[0] == image_target.shape[0], \
            f"Cannot match source and image shape: {image_source.shape},  {image_target.shape}"
        image_source = image_source.transpose(2, 0, 1)
        image_target = image_target.transpose(2, 0, 1)

        # Tensor
        image_source = torch.tensor(image_source).float()
        image_target = torch.tensor(image_target).float()

        # dimension matching
        if self.dim_match:
            image_source, image_target = self._dim_matching(image_source=image_source, image_target=image_target)

        return image_source, image_target

    def __getitem__(self, i):
        image_source, image_target = self.get_image(i)
        if self.model_name == 'guided-I2I':
            weak_label = 0
            return image_source, image_target, weak_label
        elif self.model_name in ['cWGAN-GP', 'I2SB', 'Palette', 'cWSB-GP']:
            return image_source, image_target
        else:
            raise NotImplementedError(f"Invalid model name {self.model_name}")



def get_dataset(args):
    train_dataset = DatasetWrapper(
        dataset_name=str(args.dataset_name),
        root_path=str(args.root_path),
        dataset_path=str(args.dataset_path),
        input_channel_path=str(args.input_channel_path),
        output_channel_path=str(args.output_channel_path),
        channel_table_path=str(args.channel_table_path),
        split_list=str(args.split_list_train),
        convert_gray=eval(args.convert_gray),
        resize=eval(args.resize) if hasattr(args, "resize") else None,
        crop_size=eval(args.crop_size) if hasattr(args, "crop_size") else None,
        crop_range=eval(args.crop_range) if hasattr(args, "crop_range") else None,
        crop_augmentation=eval(args.crop_augmentation) if hasattr(args, "crop_augmentation") else False,
        rotation_augmentation=eval(args.rotation_augmentation) if hasattr(args, "rotation_augmentation") else False,
        normalization=str(args.normalization) if hasattr(args, "normalization") else "minmax",
        image_dtype=str(args.image_dtype) if hasattr(args, "image_dtype") else "uint8",
        data_range=int(args.data_range) if hasattr(args, "data_range") else 255,
        in_channels=int(args.in_channels),
        out_channels=int(args.out_channels),
        image_size=eval(args.resize) if eval(args.resize) is not None else eval(args.image_size),
        model_name=str(args.model),
        dim_match=eval(args.dim_match) if hasattr(args, 'dim_match') else False,
        input_dim_label=eval(args.input_dim_label) if hasattr(args, 'input_dim_label') else None,
        output_dim_label=eval(args.output_dim_label) if hasattr(args, 'output_dim_label') else None,
        concat_channels_process_num=int(args.concat_channels_process_num) if hasattr(args, 'concat_channels_process_num') else 1,
    )
    validation_dataset = DatasetWrapper(
        dataset_name=str(args.dataset_name),
        root_path=str(args.root_path),
        dataset_path=str(args.dataset_path),
        input_channel_path=str(args.input_channel_path),
        output_channel_path=str(args.output_channel_path),
        channel_table_path=str(args.channel_table_path),
        split_list=str(args.split_list_validation),
        convert_gray=eval(args.convert_gray),
        resize=eval(args.resize) if hasattr(args, "resize") else None,
        crop_size=eval(args.crop_size) if hasattr(args, "crop_size") else None,
        crop_range=eval(args.crop_range) if hasattr(args, "crop_range") else None,
        crop_augmentation=False,
        rotation_augmentation=False,
        normalization=str(args.normalization) if hasattr(args, "normalization") else "minmax",
        image_dtype=str(args.image_dtype) if hasattr(args, "image_dtype") else "uint8",
        data_range=int(args.data_range) if hasattr(args, "data_range") else 255,
        in_channels=int(args.in_channels),
        out_channels=int(args.out_channels),
        image_size=eval(args.resize) if eval(args.resize) is not None else eval(args.image_size),
        model_name=str(args.model),
        dim_match=eval(args.dim_match) if hasattr(args, 'dim_match') else False,
        input_dim_label=eval(args.input_dim_label) if hasattr(args, 'input_dim_label') else None,
        output_dim_label=eval(args.output_dim_label) if hasattr(args, 'output_dim_label') else None,
        concat_channels_process_num=int(args.concat_channels_process_num) if hasattr(args, 'concat_channels_process_num') else 1,
    )
    print('-- train_dataset.size = {}\n-- validation_dataset.size = {}'.format(
        train_dataset.__len__(), validation_dataset.__len__()))
    return train_dataset, validation_dataset


def get_test_dataset(args):
    test_dataset = DatasetWrapper(
        dataset_name=str(args.dataset_name),
        root_path=str(args.root_path),
        dataset_path=str(args.dataset_path),
        input_channel_path=str(args.input_channel_path),
        output_channel_path=str(args.output_channel_path),
        channel_table_path=str(args.channel_table_path),
        split_list=str(args.split_list_test),
        convert_gray=eval(args.convert_gray),
        resize=eval(args.resize) if hasattr(args, "resize") else None,
        crop_size=None,  # crop_sizeが指定されていたとしても，dataloaderの段階では生データを入力させる．cropするのはtester.pyの中．
        crop_range=None,
        crop_augmentation=False,
        rotation_augmentation=False,
        normalization=str(args.normalization) if hasattr(args, "normalization") else "minmax",
        image_dtype=str(args.image_dtype) if hasattr(args, "image_dtype") else "uint8",
        data_range=int(args.data_range) if hasattr(args, "data_range") else 255,
        in_channels=int(args.in_channels),
        out_channels=int(args.out_channels),
        image_size=eval(args.resize) if eval(args.resize) is not None else eval(args.image_size),
        model_name=str(args.model),
        dim_match=eval(args.dim_match) if hasattr(args, 'dim_match') else False,
        input_dim_label=eval(args.input_dim_label) if hasattr(args, 'input_dim_label') else None,
        output_dim_label=eval(args.output_dim_label) if hasattr(args, 'output_dim_label') else None,
        concat_channels_process_num=int(args.concat_channels_process_num) if hasattr(args, 'concat_channels_process_num') else 1,
    )
    print('-- test_dataset.size = {}'.format(test_dataset.__len__()))
    return test_dataset
