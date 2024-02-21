import math
import torch
import random
import itertools
import numpy as np
from scipy.stats import beta

"""
Normalization
"""
def min_max_normalize_one_image(image):

    max_int = image.max()
    min_int = image.min()
    if max_int - min_int > 0:
        out = (image - min_int) / (max_int - min_int)
    else:
        out = np.zeros_like(image)

    return out


def zscore_normalize_one_image(image):

    mean_int = image.mean()
    std_int = image.std()
    if std_int == 0:
        out = (image - mean_int) / std_int
    else:
        out = np.zeros_like(image)

    return out


# data_rangeは
def standardize_one_image(image, mean=0.5, std=0.5):
    # imageを0-1に
    image_norm = min_max_normalize_one_image(image)
    # imageを-1-1に
    out = (image_norm - mean) / std
    return out

"""
Crop/Rotation
"""


def image_crop(image, crop_size=(1024, 1024), crop_range=1024, augmentation=True, top=None, left=None):
    height, width, _ = image.shape
    assert height >= crop_size[0]
    assert width >= crop_size[1]

    if augmentation:
        # get cropping position (image)
        if top is None and left is None:
            top = random.randint(-crop_range, crop_range) + int((height - crop_size[0])/2)
            left = random.randint(-crop_range, crop_range) + int((width - crop_size[1])/2)
            if top < 0 or top + crop_size[0] > height:
                top = 0
            if left < 0 or left + crop_size[1] > width:
                left = 0

        bottom = top + crop_size[0]
        right = left + crop_size[1]
        cropped_image = image[top:bottom, left:right, :]

    else:
        top = int((height - crop_size[0])/2)
        left = int((width - crop_size[1])/2)
        bottom = top + crop_size[0]
        right = left + crop_size[1]
        cropped_image = image[top:bottom, left:right, :]

    return cropped_image, top, left


def image_roration(image, rot_flag=None, flip_flag=None):
    # augmentation image rotation & flip
    if rot_flag is None and flip_flag is None:
        rot_flag = random.randint(0, 3)
        flip_flag = random.randint(0, 1)
    else:
        pass

    if len(image.shape) == 3:
        new_image = image.copy().transpose(2, 0, 1)

        for c in range(len(new_image)):
            new_image[c] = np.rot90(new_image[c], k=rot_flag)
            if flip_flag:
                new_image[c] = np.flip(new_image[c], axis=0)

        new_image = new_image.transpose(1,2,0)

    elif len(image.shape) == 2:
        new_image = np.rot90(image.copy(), k=rot_flag)
        if flip_flag:
            new_image = np.flip(new_image)
    else:
        raise ValueError("Invalid image shape")
    return new_image, rot_flag, flip_flag

def tensor2numpy(image: torch.tensor, device: str):
    # tensor to numpy
    image = image.to(device).detach().numpy()
    image = np.squeeze(image)
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)  # CHW to HWC
    image = image.astype('float32')
    return image

def numpy2tensor(image: np.ndarray):
    if len(image.shape) == 3:
        image = image.transpose(2, 0, 1)
    image = torch.tensor(image).float()
    return image

def image_sliding_crop(image: torch.tensor, crop_size=(1024, 1024), slide_mode='half'):
    # tensor to numpy
    image = tensor2numpy(image=image, device=torch.device('cpu'))

    height, width, _ = image.shape
    h_crop, w_crop = crop_size
    assert height >= h_crop
    assert width >= w_crop

    # get cropping position (image)
    if slide_mode == 'half':  # 半分ずつずらす
        h_slide, w_slide = int(np.floor(h_crop/2)), int(np.floor(w_crop/2))
    elif slide_mode == 'one':  # 1ピクセルごとにずらす
        h_slide, w_slide = 1, 1
    else:
        raise ValueError(f"Invalid slide mode: {slide_mode}")

    tops = list(np.arange(0, height-h_crop+1, h_slide))
    lefts = list(np.arange(0, width-w_crop+1, w_slide))

    # 余りが出る場合
    if tops[-1] + h_crop != height:
        tops.append(height-h_crop)
    if lefts[-1] + w_crop != width:
        lefts.append(width-w_crop)

    cropped_images = []
    for top, left in itertools.product(tops, lefts):
        bottom = top + h_crop
        right = left + w_crop
        cropped_image = image[top:bottom, left:right, :]
        pos = [top, bottom, left, right]

        # numpy to tensor
        cropped_image = numpy2tensor(image=cropped_image)

        cropped_images.append({'pos':pos,'data':cropped_image})

    return cropped_images

def concatinate_slides(images: list, source: torch.tensor):
    # prepare box to save
    source_numpy = tensor2numpy(image=source, device=torch.device('cpu'))
    data_shape = source_numpy.shape
    out_box = dict()
    for i in range(data_shape[0]):
        out_box[i] = dict()
        for j in range(data_shape[1]):
            out_box[i][j] = dict()
            for k in range(data_shape[2]):
                out_box[i][j][k] = []

    # collect data
    for image in images:
        top, bottom, left, right = image['pos']
        img = image['data']
        # tensor to numpy
        img = tensor2numpy(image=img, device=torch.device('cpu'))

        for i, img_i in zip(range(top, bottom), range(img.shape[0])):
            for j, img_j in zip(range(left, right), range(img.shape[1])):
                for k in range(img.shape[2]):
                    val = img[img_i, img_j, k]
                    val = val.astype('float64')  # 丸め誤差抑制
                    out_box[i][j][k].append(val)

    # calculate average
    out_image = np.zeros(data_shape).astype('float32')
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            for k in range(data_shape[2]):
                val_list = out_box[i][j][k]
                out_image[i, j, k] = np.mean(val_list).astype('float32')

    # numpy to tensor
    out_image = numpy2tensor(image=out_image)
    return out_image