import copy
import os
import cv2
import csv
import time
import json
import torch
import torch.nn.functional as F
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import torch.nn as nn
import torch.autograd as autograd
from PIL import Image
from torch.autograd import Variable
from matplotlib import pyplot as plt
from src.lib.datasets.augmentations import image_sliding_crop, concatinate_slides, numpy2tensor, tensor2numpy
from src.lib.utils.utils import check_dir, convert_channels_to_rgbs, save_image_function
from src.lib.losses.losses import GenLoss
from skimage import io
import imageio
from src.lib.models.diffusion_for_I2SB import Diffusion as DiffusionI2SB

import math
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from PIL import Image
from torch_ema import ExponentialMovingAverage

def PSNR(img_1, img_2, data_range=255):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


class cWGANGPTester(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
        self.crop_size = kwargs['crop_size'] if 'crop_size' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None
        self.lamb = torch.tensor(float(kwargs['lamb'])).float().to(torch.device(self.device))
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
        self.gen_criterion = GenLoss()
        self.image_save = kwargs['image_save'] if 'image_save' in kwargs else False

        self.file_list = kwargs['file_list'] if 'file_list' in kwargs else None
        self.input_channel_list = kwargs['input_channel_list'] if 'input_channel_list' in kwargs else None
        self.output_channel_list = kwargs['output_channel_list'] if 'output_channel_list' in kwargs else None
        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.result_csv_path = os.path.join(self.save_dir, 'result_at_each_image.csv')
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']


    def _test_func(self, data, model_G, model_D, epoch, phase, save_dir_img, cnt):
        input_real, output_real = data
        input_real = input_real.to(torch.device(self.device))
        real_imgs = output_real.to(torch.device(self.device))

        with torch.no_grad():
            if phase == 'test' and self.crop_size is not None:
                # test phaseかつcrop augmentationさせてた場合，slideさせて平均を取る
                cropped_input_reals = image_sliding_crop(image=input_real,
                                                         crop_size=self.crop_size,
                                                         slide_mode='half')

                fake_img_box = []
                for cropped_input_real_data in cropped_input_reals:
                    # parse
                    pos = cropped_input_real_data['pos']
                    cropped_input_real = cropped_input_real_data['data']
                    cropped_input_real = cropped_input_real.to(torch.device(self.device))
                    cropped_input_real = torch.unsqueeze(cropped_input_real, dim=0)  # batch
                    # inference
                    cropped_fake_imgs = model_G(cropped_input_real)
                    cropped_fake_imgs = torch.squeeze(cropped_fake_imgs)

                    fake_img_box.append({'pos': pos, 'data': cropped_fake_imgs})

                fake_imgs = concatinate_slides(images=fake_img_box, source=real_imgs)
                fake_imgs = torch.unsqueeze(fake_imgs, dim=0)  # batch
            else:
                fake_imgs = model_G(input_real)

            fake_imgs = fake_imgs.to(torch.device(self.device))

        # create input sets for discriminator
        real_concat_with_input = torch.cat((real_imgs, input_real), 1)
        fake_concat_with_input = torch.cat((fake_imgs, input_real), 1)
        real_concat_with_input = real_concat_with_input.to(torch.device(self.device))
        fake_concat_with_input = fake_concat_with_input.to(torch.device(self.device))

        # discriminate
        real_d = model_D(real_concat_with_input).mean()
        fake_d = model_D(fake_concat_with_input).mean()

        # calculate gradient penalty
        gradient_penalty = self._calculate_gradient_penalty(model_D, real_concat_with_input, fake_concat_with_input)

        # loss
        loss_D = fake_d - real_d + self.lamb * gradient_penalty
        loss_G = self.gen_criterion(fake_d, fake_imgs, real_imgs, epoch)

        ### 3. evaluate output
        ssims, mses, maes, psnrs = self._evaluate(fake_imgs, real_imgs, phase, cnt)

        if phase == 'test' and self.image_save:
            # number = str(cnt).zfill(4)
            fake_imgs_image = self._convert_tensor_to_image(fake_imgs)
            if len(self.output_channel_list) == 0:  # rgb
                fake_imgs_image_bgr = cv2.cvtColor(fake_imgs_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{save_dir_img}/generated_{self.file_list[cnt]}.tif", fake_imgs_image_bgr)
                if self.image_dtype != 'uint8':
                    fake_imgs_image_uint8 = (fake_imgs_image / self.data_range) * 255
                    fake_imgs_image_uint8 = fake_imgs_image_uint8.astype('uint8')
                    fake_imgs_image_uint8_bgr = cv2.cvtColor(fake_imgs_image_uint8, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{save_dir_img}/generated_{self.file_list[cnt]}.png", fake_imgs_image_uint8_bgr)
            else:
                name = os.path.basename(self.file_list[cnt])
                save_dir_img_each = check_dir(f"{save_dir_img}/{name}")

                for channel_ind in range(len(self.output_channel_list)):
                    filename_ch = f"{name}_channel_{self.output_channel_list[channel_ind]}"
                    fake_imgs_image_channel = fake_imgs_image[:, :, channel_ind]
                    save_image_function(save_dir=save_dir_img_each,
                                        filename=filename_ch,
                                        img=fake_imgs_image_channel)

                save_dir_img_each_composite = check_dir(f"{save_dir_img_each}/Composite")
                composite = convert_channels_to_rgbs(images=fake_imgs_image,
                                                     table_label=self.table_label,
                                                     table_artifact=self.table_artifact,
                                                     flag_artifact=True,
                                                     data_range=self.data_range,
                                                     image_dtype=self.image_dtype)

                save_image_function(save_dir=save_dir_img_each_composite,
                                    filename=name,
                                    img=composite)

        return loss_D, loss_G, ssims, mses, maes, psnrs

    def test(self, model_G, model_D, data_iter, phase="test", epoch=0):
        start = time.time()

        if phase == 'test':
            with open(self.result_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'file_name', 'ssim', 'mse', 'mae', 'psnr'])

        # turn on the testing mode; clean up the history
        model_G.eval()
        model_D.eval()
        model_G.phase = phase
        model_D.phase = phase

        save_dir_img = None
        if phase == 'test':
            save_dir_img = check_dir(os.path.join(self.save_dir, "images"))

        ssim_list = []
        mse_list = []
        mae_list = []
        psnr_list = []

        loss_G_list = []
        loss_D_list = []

        # predict
        if phase == 'test':
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, data in enumerate(tqdm(data_iter, disable=tqdm_disable)):
            output = self._test_func(data, model_G, model_D, epoch, phase, save_dir_img, i)
            loss_G, loss_D, ssims, mses, maes, psnrs = output
            # save results
            loss_D_list.append(loss_D.to(torch.device('cpu')).detach().numpy())
            loss_G_list.append(loss_G.to(torch.device('cpu')).detach().numpy())
            ssim_list += ssims
            mse_list += mses
            mae_list += maes
            psnr_list += psnrs

        evaluates_dict = {
            "ssim": ssim_list,
            "mse": mse_list,
            "mae": mae_list,
            "psnr": psnr_list,
        }

        loss_G_mean = float(abs(np.mean(loss_G_list)))
        loss_D_mean = float(abs(np.mean(loss_D_list)))
        elapsed_time = time.time() - start

        print_text = "[{}] loss G: {}, D: {} | {}, elapsed time: {} s".\
            format(phase, loss_G_mean, loss_D_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time)))
        print(print_text)

        if phase == 'validation':
            model_G.phase = 'train'
            model_D.phase = 'train'

        elif phase == 'test':
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write(print_text)
            self._save_log(loss_G_list, loss_D_list, evaluates_dict)

        return loss_G_list, loss_D_list, evaluates_dict

    def _convert_tensor_to_image(self, image):
        if torch.is_tensor(image):
            image = image.to(torch.device('cpu')).detach().numpy()
        image = np.squeeze(image)
        if self.normalization == 'minmax':
            # 0-1の範囲を0-255(uint8なら)に変換
            image = image * self.data_range
        elif self.normalization == 'zscore':
            # 0-1の範囲を0-255(uint8なら)に変換
            raise NotImplementedError
        elif self.normalization == 'std':
            # -1-1の範囲を0-255(uint8なら)に変換
            image = (image + 1) / 2
            image = image * self.data_range
        else:
            raise NotImplementedError

        image = np.clip(image, 0, self.data_range)
        image = image.astype(self.image_dtype)
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        return image

    def _calculate_gradient_penalty(self, model_D, real_images, fake_images):
        # generate random eta
        eta = torch.rand(self.batch_size, 1, 1, 1, device=torch.device('cpu')) # self.device
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(torch.device(self.device))
        # calculate interpolated
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = model_D(interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(torch.device(self.device)),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def _evaluate(self, fake_imgs, real_imgs, phase, cnt):
        ssims, mses, maes, psnrs = [], [], [], []

        for ind in range(fake_imgs.shape[0]):
            ssim_channel = []
            mse_channel = []
            mae_channel = []
            psnr_channel = []
            for c in range(fake_imgs.shape[1]):
                pred = fake_imgs[ind, c, :, :].cpu().detach().numpy().copy()
                true = real_imgs[ind, c, :, :].cpu().detach().numpy().copy()

                pred = self._convert_tensor_to_image(pred)
                true = self._convert_tensor_to_image(true)

                ssim = structural_similarity(pred, true, data_range=self.data_range)
                #print('koko',np.max(true),np.max(pred),true.shape,pred.shape)
                mse = mean_squared_error(pred, true)
                mae = mean_absolute_error(pred, true)
                psnr = PSNR(pred, true, data_range=self.data_range)

                ssim_channel.append(ssim)
                mse_channel.append(mse)
                mae_channel.append(mae)
                psnr_channel.append(psnr)

            ssims.append(ssim_channel)
            mses.append(mse_channel)
            maes.append(mae_channel)
            psnrs.append(psnr_channel)

            if phase == 'test':
                with open(self.result_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt,
                                    self.file_list[cnt],
                                    np.mean(ssim_channel),
                                    np.mean(mse_channel),
                                    np.mean(mae_channel),
                                    np.mean(psnr_channel),
                                    ])

        return ssims, mses, maes, psnrs

    def _save_log(self, loss_G_list_test, loss_D_list_test, evaluates_dict_test):
        # create dict
        result = {}
        # loss in validation
        result['loss_G_test'] = float(np.mean(loss_G_list_test))
        result['loss_G_test_list'] = [float(l) for l in loss_G_list_test]
        result['loss_D_test'] = float(np.mean(loss_D_list_test))
        result['loss_D_test_list'] = [float(l) for l in loss_D_list_test]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_test.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_test[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result[f'{key}_test'] = float(np.mean(data_list))
            result[f'{key}_test_list'] = data_list
            result[f'{key}_test_list_channels'] = data_list_channels

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(result, f, indent=4)

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key) + ": " + "{}".format(vals_mean)
            out.append(text)
        return ", ".join(out)


'''
guided-I2I
'''

class guidedI2ITester(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
        self.crop_size = kwargs['crop_size'] if 'crop_size' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None

        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None

        self.image_save = kwargs['image_save'] if 'image_save' in kwargs else False

        self.file_list = kwargs['file_list'] if 'file_list' in kwargs else None
        self.input_channel_list = kwargs['input_channel_list'] if 'input_channel_list' in kwargs else None
        self.output_channel_list = kwargs['output_channel_list'] if 'output_channel_list' in kwargs else None
        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.result_csv_path = os.path.join(self.save_dir, 'result_at_each_image.csv')
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.sample_num = kwargs['sample_num']
        CustomResult = collections.namedtuple('CustomResult', 'name result')
        self.results_dict = CustomResult([], [])
        self.task = kwargs['task']
        self.eval_metrics = kwargs['eval_metrics'] if 'eval_metrics' in kwargs else None
        self.lossfun = kwargs['lossfun'] if 'lossfun' in kwargs else None


    def _parse_data(self, data):
        image_source, image_target, weak_label = data
        cond_image = image_source.to(torch.device(self.device))
        gt_image = image_target.to(torch.device(self.device))
        weak_label = weak_label.to(self.device)
        return cond_image, gt_image, weak_label, self.batch_size

    def _check_flag(self, phase):
        if (phase == 'validation' and self.eval_metrics is not None) or (phase == 'test'):
            return True
        else:
            return False

    def _crop_and_concatinate(self, model, input_real, real_imgs, phase, weak_label):
        # test phaseかつcrop augmentationさせてた場合，slideさせて平均を取る
        cropped_input_reals = image_sliding_crop(image=input_real,
                                                 crop_size=self.crop_size,
                                                 slide_mode='half')
        cropped_real_imgs = image_sliding_crop(image=real_imgs,
                                               crop_size=self.crop_size,
                                               slide_mode='half')
        loss_box = []
        if self._check_flag(phase=phase):
            fake_img_box = []
            visuals_box = []

        for cropped_real_imgs_data, cropped_input_real_data in zip(cropped_real_imgs, cropped_input_reals):
            # parse
            pos = cropped_input_real_data['pos']
            cropped_input_real = cropped_input_real_data['data']
            cropped_input_real = cropped_input_real.to(torch.device(self.device))
            cropped_input_real = torch.unsqueeze(cropped_input_real, dim=0)  # batch

            cropped_real_imgs = cropped_real_imgs_data['data']
            cropped_real_imgs = cropped_real_imgs.to(torch.device(self.device))
            cropped_real_imgs = torch.unsqueeze(cropped_real_imgs, dim=0)  # batch

            # inference
            cropped_loss = model(cropped_real_imgs, weak_label, cropped_input_real, mask=None)
            cropped_loss = cropped_loss.to('cpu').clone().detach().numpy()
            loss_box.append(cropped_loss)

            if self._check_flag(phase=phase):
                cropped_fake_imgs, cropped_visuals = model.restoration(
                    cropped_input_real,
                    weak_label,
                    classifier_scale=1, y_t=None,
                    y_0=cropped_real_imgs, mask=None,
                    sample_num=self.sample_num
                )
                cropped_fake_imgs = torch.squeeze(cropped_fake_imgs)
                cropped_fake_imgs = cropped_fake_imgs.to(torch.device(self.device)).clone().detach()

                cropped_visuals = torch.squeeze(cropped_visuals)
                cropped_visuals = cropped_visuals.to(torch.device(self.device)).clone().detach()

                fake_img_box.append({'pos': pos, 'data': cropped_fake_imgs})
                visuals_box.append({'pos': pos, 'data': cropped_visuals})

        loss = float(abs(np.mean(loss_box)))

        if self._check_flag(phase=phase):
            fake_imgs = concatinate_slides(images=fake_img_box, source=real_imgs)
            fake_imgs = torch.unsqueeze(fake_imgs, dim=0)  # batch
            fake_imgs = fake_imgs.to(torch.device(self.device))

            visuals = None
            for ind, channel in enumerate(range(visuals_box[0]['data'].shape[0])):
                visuals_box_modify_each_channel = []
                for idx in range(len(visuals_box)):
                    visuals_process = visuals_box[idx]['data'][channel, :, :, :]
                    visuals_box_modify_each_channel.append({'pos': visuals_box[idx]['pos'], 'data': visuals_process})

                visuals_each_channel = concatinate_slides(images=visuals_box_modify_each_channel,
                                                          source=real_imgs)
                visuals_each_channel = torch.unsqueeze(visuals_each_channel, dim=0)  # batch
                visuals_each_channel = visuals_each_channel.to(torch.device(self.device))
                if ind == 0:
                    visuals = visuals_each_channel
                else:
                    visuals = torch.cat([visuals, visuals_each_channel], dim=0)

            visuals = visuals.to(torch.device(self.device))
            return torch.tensor(loss), fake_imgs, visuals
        else:
            return torch.tensor(loss), None, None


    def _test_func(self, data, model, phase, save_dir_img, cnt):
        # parse data
        input_real, real_imgs, weak_label, batch_size = self._parse_data(data=data)

        with torch.no_grad():
            if phase == 'test' and self.crop_size is not None:
                loss, fake_imgs, visuals = self._crop_and_concatinate(model, input_real, real_imgs, phase, weak_label)
            else:
                # inference
                loss = model(real_imgs, weak_label, input_real, mask=None)

                if self._check_flag(phase=phase):
                    fake_imgs, visuals = model.restoration(input_real, weak_label,
                                                           classifier_scale=1, y_t=None,
                                                           y_0=real_imgs, mask=None,
                                                           sample_num=self.sample_num)

                    fake_imgs = fake_imgs.to(torch.device(self.device)).clone().detach()

            if phase == 'test' and self.image_save:
                results = self._save_current_results(path=self.file_list[cnt],
                                                     gt_image=real_imgs,
                                                     visuals=visuals)
                self._save_images_test(save_dir=save_dir_img, results=results)

        ssims, mses, maes, psnrs = None, None, None, None
        if self._check_flag(phase=phase):
            ### 3. evaluate output
            ssims, mses, maes, psnrs = self._evaluate(fake_imgs, real_imgs, phase, cnt)

        return loss, ssims, mses, maes, psnrs

    def test(self, model, data_iter, phase="test"):
        start = time.time()

        if phase == 'test':
            with open(self.result_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'file_name', 'ssim', 'mse', 'mae', 'psnr'])

        # turn on the testing mode; clean up the history
        model.eval()
        model.phase = phase

        save_dir_img = None
        if phase == 'test':
            save_dir_img = check_dir(os.path.join(self.save_dir, "images"))
            #model.set_loss(self.lossfun)
            model.set_new_noise_schedule(device=self.device, phase='test')

        if self._check_flag(phase=phase):
            ssim_list = []
            mse_list = []
            mae_list = []
            psnr_list = []

        loss_list = []

        # predict
        if phase == 'test':
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, data in enumerate(tqdm(data_iter, disable=tqdm_disable)):
            output = self._test_func(data, model, phase, save_dir_img, i)
            loss, ssims, mses, maes, psnrs = output
            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())
            if self._check_flag(phase=phase):
                ssim_list += ssims
                mse_list += mses
                mae_list += maes
                psnr_list += psnrs

        if self._check_flag(phase=phase):
            evaluates_dict = {
                "ssim": ssim_list,
                "mse": mse_list,
                "mae": mae_list,
                "psnr": psnr_list,
            }

        loss_mean = float(abs(np.mean(loss_list)))
        elapsed_time = time.time() - start

        if self._check_flag(phase=phase):
            print_text = "[{}] loss: {} | {}, elapsed time: {} s".\
                format(phase, loss_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time)))
        else:
            print_text = "[{}] loss: {} , elapsed time: {} s".\
                format(phase, loss_mean, int(np.floor(elapsed_time)))
        print(print_text)

        if phase == 'validation':
            model.phase = 'train'

        elif phase == 'test':
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write(print_text)
            self._save_log(loss_list, evaluates_dict)

        if self._check_flag(phase=phase):
            return loss_list, evaluates_dict
        else:
            return loss_list

    def _convert_tensor_to_image(self, image):
        if torch.is_tensor(image):
            image = image.to(torch.device('cpu')).detach().numpy()
        image = np.squeeze(image)
        if self.normalization == 'minmax':
            # 0-1の範囲を0-255(uint8なら)に変換
            image = image * self.data_range
        elif self.normalization == 'zscore':
            # 0-1の範囲を0-255(uint8なら)に変換
            raise NotImplementedError
        elif self.normalization == 'std':
            # -1-1の範囲を0-255(uint8なら)に変換
            image = (image + 1) / 2
            image = image * self.data_range
        else:
            raise NotImplementedError

        image = np.clip(image, 0, self.data_range)
        image = image.astype(self.image_dtype)
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        return image

    def _evaluate(self, fake_imgs, real_imgs, phase, cnt):
        ssims, mses, maes, psnrs = [], [], [], []

        for ind in range(fake_imgs.shape[0]):
            ssim_channel = []
            mse_channel = []
            mae_channel = []
            psnr_channel = []
            for c in range(fake_imgs.shape[1]):
                pred = fake_imgs[ind, c, :, :].cpu().detach().numpy().copy()
                true = real_imgs[ind, c, :, :].cpu().detach().numpy().copy()

                pred = self._convert_tensor_to_image(pred)
                true = self._convert_tensor_to_image(true)

                ssim = structural_similarity(pred, true, data_range=self.data_range)
                #print('koko',np.max(true),np.max(pred),true.shape,pred.shape)
                mse = mean_squared_error(pred, true)
                mae = mean_absolute_error(pred, true)
                psnr = PSNR(pred, true, data_range=self.data_range)

                ssim_channel.append(ssim)
                mse_channel.append(mse)
                mae_channel.append(mae)
                psnr_channel.append(psnr)

            ssims.append(ssim_channel)
            mses.append(mse_channel)
            maes.append(mae_channel)
            psnrs.append(psnr_channel)

            if phase == 'test':
                with open(self.result_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt,
                                    self.file_list[cnt],
                                    np.mean(ssim_channel),
                                    np.mean(mse_channel),
                                    np.mean(mae_channel),
                                    np.mean(psnr_channel),
                                    ])

        return ssims, mses, maes, psnrs

    def _save_log(self, loss_list_test, evaluates_dict_test):
        # create dict
        result = {}
        # loss in validation
        result['loss_test'] = float(np.mean(loss_list_test))
        result['loss_test_list'] = [float(l) for l in loss_list_test]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_test.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_test[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result[f'{key}_test'] = float(np.mean(data_list))
            result[f'{key}_test_list'] = data_list
            result[f'{key}_test_list_channels'] = data_list_channels

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(result, f, indent=4)

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key) + ": " + "{}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_current_results(self, path, gt_image, visuals):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT-{}'.format(path))
            ret_result.append(gt_image[idx].detach().float().cpu())

            ret_path.append('Process-{}'.format(path))
            ret_result.append(visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Out-{}'.format(path))
            ret_result.append(visuals[idx - self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)

        return self.results_dict._asdict()

    def _save_images_test(self, save_dir, results):

        ''' get names and corresponding images from results[OrderedDict] '''

        names = results['name']
        outputs = results['result']

        for i in range(len(names)):
            name = names[i]
            id = name[name.find("-")+1:]
            print(id)
            mode = name[:name.find("-")]
            save_dir_each = check_dir(f"{save_dir}/{id}/{mode}")

            im_out = outputs[i].detach().numpy()

            if mode == "Process":
                batchsize = im_out.shape[0]
            else:
                batchsize = 1

            for b in range(batchsize):
                if mode == "Process":
                    im_out_b = im_out[b, :, :, :]

                    b_str = str(b).zfill(len(str(b)))
                    save_dir_each_b = check_dir(f"{save_dir_each}/Process-{b_str}")

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{id}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{id}",
                                            img=im_out_b)
                else:
                    im_out_b = im_out.copy()

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{id}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{id}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each}/Composite")
                        composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                             table_label=self.table_label,
                                                             table_artifact=self.table_artifact,
                                                             flag_artifact=True,
                                                             data_range=self.data_range,
                                                             image_dtype=self.image_dtype)
                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{id}",
                                            img=composite)

def make_beta_schedule_for_I2SB(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

class I2SBTester(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
        self.crop_size = kwargs['crop_size'] if 'crop_size' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None

        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None

        self.image_save = kwargs['image_save'] if 'image_save' in kwargs else False

        self.file_list = kwargs['file_list'] if 'file_list' in kwargs else None
        self.in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else None
        self.out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else None
        self.input_channel_list = kwargs['input_channel_list'] if 'input_channel_list' in kwargs else None
        self.output_channel_list = kwargs['output_channel_list'] if 'output_channel_list' in kwargs else None
        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.result_csv_path = os.path.join(self.save_dir, 'result_at_each_image.csv')
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.eval_metrics = kwargs['eval_metrics'] if 'eval_metrics' in kwargs else None
        self.cond_x1 = kwargs['cond_x1']
        self.add_x1_noise = kwargs['add_x1_noise']
        self.ot_ode = kwargs['ot_ode']
        self.interval = kwargs['interval']
        self.beta_max = kwargs['beta_max']
        self.ema = kwargs['ema'] if 'ema' in kwargs else None
        self.emer = kwargs['emer'] if 'emer' in kwargs else None
        self.diffusion = kwargs['diffusion'] if 'diffusion' in kwargs else None

        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.dim_match = kwargs['dim_match'] if 'dim_match' in kwargs else None
        self.input_dim_label = kwargs['input_dim_label'] if 'input_dim_label' in kwargs else None
        self.output_dim_label = kwargs['output_dim_label'] if 'output_dim_label' in kwargs else None
        self.distributed = kwargs['distributed'] if 'distributed' in kwargs else False


    def sample_batch(self, data):
        # clean_img is "target" domain and corrupt_img is "source" domain
        img_source, img_target = data
        mask = None

        x0 = img_target.detach().to(self.device)
        x1 = img_source.detach().to(self.device)

        cond = x1.detach() if self.cond_x1 else None

        if self.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, cond

    def _check_flag(self, phase):
        if (phase == 'validation' and self.eval_metrics is not None) or (phase == 'test'):
            return True
        else:
            return False

    def _crop_and_concatinate(self, model, x0, x1, phase, cond, mask):
        # loss
        step = torch.randint(0, self.interval, (x0.shape[0],))

        # test phaseかつcrop augmentationさせてた場合，slideさせて平均を取る
        cropped_x0s = image_sliding_crop(image=x0, crop_size=self.crop_size, slide_mode='half')
        cropped_x1s = image_sliding_crop(image=x1, crop_size=self.crop_size, slide_mode='half')
        cropped_conds = image_sliding_crop(image=cond, crop_size=self.crop_size, slide_mode='half')
        if mask is not None:
            cropped_masks = image_sliding_crop(image=mask, crop_size=self.crop_size, slide_mode='half')
        else:
            cropped_masks = None

        loss_box = []
        if self._check_flag(phase=phase):
            fake_img_box = []
            #visuals_box = []

        for cropped_x0_data, cropped_x1_data, cropped_cond_data in zip(cropped_x0s, cropped_x1s, cropped_conds):
            # parse
            pos = cropped_x0_data['pos']
            # target
            cropped_x0 = cropped_x0_data['data']
            cropped_x0 = cropped_x0.to(torch.device(self.device))
            cropped_x0 = torch.unsqueeze(cropped_x0, dim=0)  # batch
            # source
            cropped_x1 = cropped_x1_data['data']
            cropped_x1 = cropped_x1.to(torch.device(self.device))
            cropped_x1 = torch.unsqueeze(cropped_x1, dim=0)  # batch

            cropped_cond = cropped_cond_data['data']
            cropped_cond = cropped_cond.to(torch.device(self.device))
            cropped_cond = torch.unsqueeze(cropped_cond, dim=0)  # batch

            # inference
            cropped_xt = self.diffusion.q_sample(step, cropped_x0, cropped_x1, ot_ode=self.ot_ode)
            cropped_label = self.compute_label(step, cropped_x0, cropped_xt)
            cropped_pred = model(cropped_xt, step, cond=cropped_cond)
            assert cropped_xt.shape == cropped_label.shape == cropped_pred.shape

            if cropped_masks is not None:
                cropped_mask = [d['data'] if d['pos'] == pos else None for d in cropped_masks][0]
                cropped_pred = cropped_mask * cropped_pred
                cropped_label = cropped_mask * cropped_label
            else:
                cropped_mask = None

            cropped_loss = F.mse_loss(cropped_pred, cropped_label)

            cropped_loss = cropped_loss.to('cpu').clone().detach().numpy()
            loss_box.append(cropped_loss)

            if self._check_flag(phase=phase):
                cropped_fake_imgs, _ = self.ddpm_sampling(
                    model, cropped_x1, mask=cropped_mask, cond=cropped_cond, clip_denoise=True, verbose=False
                )

                cropped_fake_imgs = torch.squeeze(cropped_fake_imgs)
                cropped_fake_imgs = cropped_fake_imgs.to(torch.device(self.device)).clone().detach()

                #cropped_visuals = torch.squeeze(cropped_visuals)
                #cropped_visuals = cropped_visuals.to(torch.device(self.device)).clone().detach()

                fake_img_box.append({'pos': pos, 'data': cropped_fake_imgs})
                #visuals_box.append({'pos': pos, 'data': cropped_visuals})

        loss = float(abs(np.mean(loss_box)))

        if self._check_flag(phase=phase):
            fake_imgs = None
            for ind, process in enumerate(range(fake_img_box[0]['data'].shape[0])):
                fake_imgs_box_modify_each_process = []
                for idx in range(len(fake_img_box)):
                    fake_imgs_process = fake_img_box[idx]['data'][process, :, :, :]
                    fake_imgs_box_modify_each_process.append({'pos': fake_img_box[idx]['pos'], 'data': fake_imgs_process})

                fake_imgs_each_process = concatinate_slides(images=fake_imgs_box_modify_each_process,
                                                            source=x1)
                fake_imgs_each_process = torch.unsqueeze(fake_imgs_each_process, dim=0)  # batch
                fake_imgs_each_process = fake_imgs_each_process.to(torch.device(self.device))
                if ind == 0:
                    fake_imgs = fake_imgs_each_process
                else:
                    fake_imgs = torch.cat([fake_imgs, fake_imgs_each_process], dim=0)

            fake_imgs = fake_imgs.to(torch.device(self.device))

            visuals = None
            # for ind, process in enumerate(range(visuals_box[0]['data'].shape[0])):
            #     visuals_box_modify_each_process = []
            #     for idx in range(len(visuals_box)):
            #         visuals_process = visuals_box[idx]['data'][process, :, :, :]
            #         visuals_box_modify_each_process.append({'pos': visuals_box[idx]['pos'], 'data': visuals_process})
            #
            #     visuals_each_process = concatinate_slides(images=visuals_box_modify_each_process,
            #                                               source=x1)
            #     visuals_each_process = torch.unsqueeze(visuals_each_process, dim=0)  # batch
            #     visuals_each_process = visuals_each_process.to(torch.device(self.device))
            #     if ind == 0:
            #         visuals = visuals_each_process
            #     else:
            #         visuals = torch.cat([visuals, visuals_each_process], dim=0)
            #
            # visuals = visuals.to(torch.device(self.device))
            return torch.tensor(loss), fake_imgs, visuals
        else:
            return torch.tensor(loss), None, None

    def _test_func(self, data, model, phase, save_dir_img, cnt):
        # parse data
        # x0 target, x1 source
        x0, x1, mask, cond = self.sample_batch(data=data)

        with torch.no_grad():
            if phase == 'test' and self.crop_size is not None:
                loss, fake_imgs, _ = self._crop_and_concatinate(model, x0, x1, phase, cond, mask)
            else:
                # inference
                # loss
                step = torch.randint(0, self.interval, (x0.shape[0],))
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
                label = self.compute_label(step, x0, xt)
                pred = model(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape
                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                loss = F.mse_loss(pred, label)

                if self._check_flag(phase=phase):
                    fake_imgs, _ = self.ddpm_sampling(
                        model, x1, mask=mask, cond=cond, clip_denoise=True, verbose=False
                    )

                    fake_imgs = fake_imgs.to(torch.device(self.device)).clone().detach()

        ssims, mses, maes, psnrs = None, None, None, None
        if self._check_flag(phase=phase):
            ### 3. evaluate output
            fake_imgs_for_evaluate = torch.squeeze(fake_imgs)
            evaluate_idx = 0
            fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate[evaluate_idx, :, :, :], axis=0)
            x0_for_evaluate = x0.clone()

            if self.dim_match and self.output_dim_label is not None:
                if len(self.output_dim_label) > self.out_channels:
                    x0_for_evaluate = x0[0, :self.out_channels, :, :]
                    fake_imgs_for_evaluate = fake_imgs_for_evaluate[0, :self.out_channels, :, :]

                    x0_for_evaluate = torch.unsqueeze(x0_for_evaluate, axis=0)
                    fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate, axis=0)

            if phase == 'test' and self.image_save:
                self._save_images_test(path=self.file_list[cnt],
                                       save_dir=save_dir_img,
                                       real_imgs=x0_for_evaluate.detach().clone().float().cpu(),
                                       fake_imgs=fake_imgs_for_evaluate.detach().clone().float().cpu(),
                                       visuals=None)

            ssims, mses, maes, psnrs = self._evaluate(fake_imgs_for_evaluate, x0_for_evaluate, phase, cnt)

        return loss, ssims, mses, maes, psnrs

    def test(self, model, data_iter, phase="test"):
        start = time.time()
        # turn on the testing mode; clean up the history
        model.eval()
        model.phase = phase

        save_dir_img = None
        if phase == 'test':
            with open(self.result_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'file_name', 'ssim', 'mse', 'mae', 'psnr'])

            save_dir_img = check_dir(os.path.join(self.save_dir, "images"))

            betas = make_beta_schedule_for_I2SB(n_timestep=self.interval,
                                                linear_end=self.beta_max / self.interval)
            betas = np.concatenate([betas[:self.interval // 2], np.flip(betas[:self.interval // 2])])
            self.diffusion = DiffusionI2SB(betas, self.device)
            self.emer = ExponentialMovingAverage(model.parameters(), decay=self.ema)

        if self._check_flag(phase=phase):
            ssim_list = []
            mse_list = []
            mae_list = []
            psnr_list = []

        loss_list = []

        # predict
        if phase == 'test':
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, data in enumerate(tqdm(data_iter, disable=tqdm_disable)):
            output = self._test_func(data, model, phase, save_dir_img, i)
            loss, ssims, mses, maes, psnrs = output

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())
            if self._check_flag(phase=phase):
                ssim_list += ssims
                mse_list += mses
                mae_list += maes
                psnr_list += psnrs

        if self._check_flag(phase=phase):
            evaluates_dict = {
                "ssim": ssim_list,
                "mse": mse_list,
                "mae": mae_list,
                "psnr": psnr_list,
            }

        loss_mean = float(abs(np.mean(loss_list)))
        elapsed_time = time.time() - start

        if self._check_flag(phase=phase):
            print_text = "[{}] loss: {} | {}, elapsed time: {} s".\
                format(phase, loss_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time)))
        else:
            print_text = "[{}] loss: {} , elapsed time: {} s".\
                format(phase, loss_mean, int(np.floor(elapsed_time)))
        print(print_text)

        if phase == 'validation':
            model.phase = 'train'

        elif phase == 'test':
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write(print_text)
            self._save_log(loss_list, evaluates_dict)

        if self._check_flag(phase=phase):
            return loss_list, evaluates_dict
        else:
            return loss_list

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def _space_indices(self, num_steps, count):
        assert count <= num_steps

        if count <= 1:
            frac_stride = 1
        else:
            frac_stride = (num_steps - 1) / (count - 1)

        cur_idx = 0.0
        taken_steps = []
        for _ in range(count):
            taken_steps.append(round(cur_idx))
            cur_idx += frac_stride

        return taken_steps

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    @torch.no_grad()
    def ddpm_sampling(self, model, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or self.interval-1
        assert 0 < nfe < self.interval == len(self.diffusion.betas)
        steps = self._space_indices(self.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in self._space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        # print(f"[DDPM Sampling] steps={self.interval}, {nfe=}, {log_steps=}!")
        # self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(self.device)
        if cond is not None: cond = cond.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.emer.average_parameters():
            model.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
                out = model(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=self.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    def _convert_tensor_to_image(self, image):
        if torch.is_tensor(image):
            image = image.to(torch.device('cpu')).detach().numpy()
        image = np.squeeze(image)
        if self.normalization == 'minmax':
            # 0-1の範囲を0-255(uint8なら)に変換
            image = image * self.data_range
        elif self.normalization == 'zscore':
            # 0-1の範囲を0-255(uint8なら)に変換
            raise NotImplementedError
        elif self.normalization == 'std':
            # -1-1の範囲を0-255(uint8なら)に変換
            image = (image + 1) / 2
            image = image * self.data_range
        else:
            raise NotImplementedError

        image = np.clip(image, 0, self.data_range)
        image = image.astype(self.image_dtype)
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        return image

    def _evaluate(self, fake_imgs, real_imgs, phase, cnt):
        ssims, mses, maes, psnrs = [], [], [], []

        for ind in range(fake_imgs.shape[0]):
            ssim_channel = []
            mse_channel = []
            mae_channel = []
            psnr_channel = []
            for c in range(fake_imgs.shape[1]):
                pred = fake_imgs[ind, c, :, :].cpu().detach().numpy().copy()
                true = real_imgs[ind, c, :, :].cpu().detach().numpy().copy()

                pred = self._convert_tensor_to_image(pred)
                true = self._convert_tensor_to_image(true)

                ssim = structural_similarity(pred, true, data_range=self.data_range)
                #print('koko',np.max(true),np.max(pred),true.shape,pred.shape)
                mse = mean_squared_error(pred, true)
                mae = mean_absolute_error(pred, true)
                psnr = PSNR(pred, true, data_range=self.data_range)

                ssim_channel.append(ssim)
                mse_channel.append(mse)
                mae_channel.append(mae)
                psnr_channel.append(psnr)

            ssims.append(ssim_channel)
            mses.append(mse_channel)
            maes.append(mae_channel)
            psnrs.append(psnr_channel)

            if phase == 'test':
                with open(self.result_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt,
                                    self.file_list[cnt],
                                    np.mean(ssim_channel),
                                    np.mean(mse_channel),
                                    np.mean(mae_channel),
                                    np.mean(psnr_channel),
                                    ])

        return ssims, mses, maes, psnrs

    def _save_log(self, loss_list_test, evaluates_dict_test):
        # create dict
        result = {}
        # loss in validation
        result['loss_test'] = float(np.mean(loss_list_test))
        result['loss_test_list'] = [float(l) for l in loss_list_test]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_test.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_test[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result[f'{key}_test'] = float(np.mean(data_list))
            result[f'{key}_test_list'] = data_list
            result[f'{key}_test_list_channels'] = data_list_channels

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(result, f, indent=4)

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key) + ": " + "{}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_images_test(self, path, save_dir, real_imgs, fake_imgs, visuals):
        def _save_func(name, save_dir_each, data, mode):
            data = torch.squeeze(data.detach().clone())
            save_dir_each = check_dir(f"{save_dir_each}/{mode}")

            if mode in ["GT", "Predict"]:
                batchsize = 1
            else:
                batchsize = data.shape[0]

            for b in range(batchsize):
                if mode in ["GT", "Predict"]:
                    im_out_b = data.detach().clone().cpu()

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{name}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{name}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each}/Composite")

                        if self.dim_match and self.output_dim_label is not None:
                            if len(self.output_dim_label) > self.out_channels:
                                im_out_b = im_out_b[:self.out_channels, :, :]
                                output_dim_label = self.output_dim_label[:self.out_channels]
                            else:
                                output_dim_label = self.output_dim_label.copy()

                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=output_dim_label,
                                                                 table_artifact=[],
                                                                 flag_artifact=False,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        else:
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=self.table_label,
                                                                 table_artifact=self.table_artifact,
                                                                 flag_artifact=True,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{name}",
                                            img=composite)
                else:
                    im_out_b = data[b, :, :, :].detach().clone().cpu()

                    b_str = str(b).zfill(len(str(b)))
                    save_dir_each_b = check_dir(f"{save_dir_each}/{mode}-{b_str}")

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{name}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{name}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each_b}/Composite")
                        if self.dim_match and self.output_dim_label is not None:
                            if len(self.output_dim_label) > self.out_channels:
                                im_out_b = im_out_b[:self.out_channels, :, :]
                                output_dim_label = self.output_dim_label[:self.out_channels]
                            else:
                                output_dim_label = self.output_dim_label.copy()

                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=output_dim_label,
                                                                 table_artifact=[],
                                                                 flag_artifact=False,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        else:
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=self.table_label,
                                                                 table_artifact=self.table_artifact,
                                                                 flag_artifact=True,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{name}",
                                            img=composite)
        name = os.path.splitext(os.path.basename(path))[0]
        save_dir_each = check_dir(f"{save_dir}/{name}")

        _save_func(name=name, save_dir_each=save_dir_each, data=real_imgs, mode='GT')
        _save_func(name=name, save_dir_each=save_dir_each, data=fake_imgs, mode='Predict')
        #_save_func(name=name, save_dir_each=save_dir_each, data=visuals, mode='Predict_Visuals')


'''
Palette
'''

class PaletteTester(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
        self.crop_size = kwargs['crop_size'] if 'crop_size' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None

        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None

        self.image_save = kwargs['image_save'] if 'image_save' in kwargs else False

        self.file_list = kwargs['file_list'] if 'file_list' in kwargs else None
        self.in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else None
        self.out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else None
        self.input_channel_list = kwargs['input_channel_list'] if 'input_channel_list' in kwargs else None
        self.output_channel_list = kwargs['output_channel_list'] if 'output_channel_list' in kwargs else None
        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.result_csv_path = os.path.join(self.save_dir, 'result_at_each_image.csv')
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.sample_num = kwargs['sample_num']
        CustomResult = collections.namedtuple('CustomResult', 'name result')
        self.results_dict = CustomResult([], [])
        self.task = kwargs['task']
        self.eval_metrics = kwargs['eval_metrics'] if 'eval_metrics' in kwargs else None
        self.lossfun = kwargs['lossfun'] if 'lossfun' in kwargs else None

        self.dim_match = kwargs['dim_match'] if 'dim_match' in kwargs else None
        self.input_dim_label = kwargs['input_dim_label'] if 'input_dim_label' in kwargs else None
        self.output_dim_label = kwargs['output_dim_label'] if 'output_dim_label' in kwargs else None

    def _parse_data(self, data):
        image_source, image_target = data
        image_source = image_source.to(torch.device(self.device))
        image_target = image_target.to(torch.device(self.device))
        return image_source, image_target, self.batch_size
    def _check_flag(self, phase):
        if (phase == 'validation' and self.eval_metrics is not None) or (phase == 'test'):
            return True
        else:
            return False

    def _crop_and_concatinate(self, model, input_real, real_imgs, phase):
        # test phaseかつcrop augmentationさせてた場合，slideさせて平均を取る
        cropped_input_reals = image_sliding_crop(image=input_real,
                                                 crop_size=self.crop_size,
                                                 slide_mode='half')
        cropped_real_imgs = image_sliding_crop(image=real_imgs,
                                               crop_size=self.crop_size,
                                               slide_mode='half')
        loss_box = []
        if self._check_flag(phase=phase):
            fake_img_box = []
            visuals_box = []

        for cropped_real_imgs_data, cropped_input_real_data in zip(cropped_real_imgs, cropped_input_reals):
            # parse
            pos = cropped_input_real_data['pos']
            cropped_input_real = cropped_input_real_data['data']
            cropped_input_real = cropped_input_real.to(torch.device(self.device))
            cropped_input_real = torch.unsqueeze(cropped_input_real, dim=0)  # batch

            cropped_real_imgs = cropped_real_imgs_data['data']
            cropped_real_imgs = cropped_real_imgs.to(torch.device(self.device))
            cropped_real_imgs = torch.unsqueeze(cropped_real_imgs, dim=0)  # batch

            # inference
            cropped_loss = model(cropped_real_imgs, cropped_input_real, mask=None)
            cropped_loss = cropped_loss.to('cpu').clone().detach().numpy()
            loss_box.append(cropped_loss)

            if self._check_flag(phase=phase):
                cropped_fake_imgs, cropped_visuals = model.restoration(cropped_input_real, sample_num=self.sample_num)
                cropped_fake_imgs = torch.squeeze(cropped_fake_imgs)
                cropped_fake_imgs = cropped_fake_imgs.to(torch.device(self.device)).clone().detach()

                cropped_visuals = torch.squeeze(cropped_visuals)
                cropped_visuals = cropped_visuals.to(torch.device(self.device)).clone().detach()

                fake_img_box.append({'pos': pos, 'data': cropped_fake_imgs})
                visuals_box.append({'pos': pos, 'data': cropped_visuals})

        loss = float(abs(np.mean(loss_box)))

        if self._check_flag(phase=phase):
            fake_imgs = concatinate_slides(images=fake_img_box, source=real_imgs)
            fake_imgs = torch.unsqueeze(fake_imgs, dim=0)  # batch
            fake_imgs = fake_imgs.to(torch.device(self.device))

            visuals = None
            for ind, channel in enumerate(range(visuals_box[0]['data'].shape[0])):
                visuals_box_modify_each_channel = []
                for idx in range(len(visuals_box)):
                    visuals_process = visuals_box[idx]['data'][channel, :, :, :]
                    visuals_box_modify_each_channel.append({'pos': visuals_box[idx]['pos'], 'data': visuals_process})

                visuals_each_channel = concatinate_slides(images=visuals_box_modify_each_channel,
                                                          source=real_imgs)
                visuals_each_channel = torch.unsqueeze(visuals_each_channel, dim=0)  # batch
                visuals_each_channel = visuals_each_channel.to(torch.device(self.device))
                if ind == 0:
                    visuals = visuals_each_channel
                else:
                    visuals = torch.cat([visuals, visuals_each_channel], dim=0)

            visuals = visuals.to(torch.device(self.device))
            return torch.tensor(loss), fake_imgs, visuals
        else:
            return torch.tensor(loss), None, None

    def _test_func(self, data, model, phase, save_dir_img, cnt):
        # parse data
        input_real, real_imgs, batch_size = self._parse_data(data=data)

        with torch.no_grad():
            if phase == 'test' and self.crop_size is not None:
                loss, fake_imgs, visuals = self._crop_and_concatinate(model, input_real, real_imgs, phase)
            else:
                # inference
                loss = model(real_imgs, input_real, mask=None)

                if self._check_flag(phase=phase):
                    fake_imgs, visuals = model.restoration(input_real, sample_num=self.sample_num)

                    fake_imgs = fake_imgs.to(torch.device(self.device)).clone().detach()

            if phase == 'test' and self.image_save:
                results = self._save_current_results(path=self.file_list[cnt],
                                                     gt_image=real_imgs,
                                                     visuals=visuals)
                self._save_images_test(save_dir=save_dir_img, results=results)

        ssims, mses, maes, psnrs = None, None, None, None
        if self._check_flag(phase=phase):
            ### 3. evaluate output
            fake_imgs_for_evaluate = fake_imgs.clone()
            real_imgs_for_evaluate = real_imgs.clone()

            if self.dim_match and self.output_dim_label is not None:
                if len(self.output_dim_label) > self.out_channels:
                    real_imgs_for_evaluate = real_imgs[0, :self.out_channels, :, :]
                    fake_imgs_for_evaluate = fake_imgs_for_evaluate[0, :self.out_channels, :, :]

                    real_imgs_for_evaluate = torch.unsqueeze(real_imgs_for_evaluate, axis=0)
                    fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate, axis=0)

            ssims, mses, maes, psnrs = self._evaluate(fake_imgs_for_evaluate, real_imgs_for_evaluate, phase, cnt)

        return loss, ssims, mses, maes, psnrs

    def test(self, model, data_iter, phase="test"):
        start = time.time()

        if phase == 'test':
            with open(self.result_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'file_name', 'ssim', 'mse', 'mae', 'psnr'])

        # turn on the testing mode; clean up the history
        model.eval()
        model.phase = phase

        save_dir_img = None
        if phase == 'test':
            save_dir_img = check_dir(os.path.join(self.save_dir, "images"))
            #model.set_loss(self.lossfun)
            model.set_new_noise_schedule(device=self.device, phase='test')

        if self._check_flag(phase=phase):
            ssim_list = []
            mse_list = []
            mae_list = []
            psnr_list = []

        loss_list = []

        # predict
        if phase == 'test':
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, data in enumerate(tqdm(data_iter, disable=tqdm_disable)):
            output = self._test_func(data, model, phase, save_dir_img, i)
            loss, ssims, mses, maes, psnrs = output

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())
            if self._check_flag(phase=phase):
                ssim_list += ssims
                mse_list += mses
                mae_list += maes
                psnr_list += psnrs

        if self._check_flag(phase=phase):
            evaluates_dict = {
                "ssim": ssim_list,
                "mse": mse_list,
                "mae": mae_list,
                "psnr": psnr_list,
            }

        loss_mean = float(abs(np.mean(loss_list)))
        elapsed_time = time.time() - start

        if self._check_flag(phase=phase):
            print_text = "[{}] loss: {} | {}, elapsed time: {} s".\
                format(phase, loss_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time)))
        else:
            print_text = "[{}] loss: {} , elapsed time: {} s".\
                format(phase, loss_mean, int(np.floor(elapsed_time)))
        print(print_text)

        if phase == 'validation':
            model.phase = 'train'

        elif phase == 'test':
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write(print_text)
            self._save_log(loss_list, evaluates_dict)

        if self._check_flag(phase=phase):
            return loss_list, evaluates_dict
        else:
            return loss_list

    def _convert_tensor_to_image(self, image):
        if torch.is_tensor(image):
            image = image.to(torch.device('cpu')).detach().numpy()
        image = np.squeeze(image)
        if self.normalization == 'minmax':
            # 0-1の範囲を0-255(uint8なら)に変換
            image = image * self.data_range
        elif self.normalization == 'zscore':
            # 0-1の範囲を0-255(uint8なら)に変換
            raise NotImplementedError
        elif self.normalization == 'std':
            # -1-1の範囲を0-255(uint8なら)に変換
            image = (image + 1) / 2
            image = image * self.data_range
        else:
            raise NotImplementedError

        image = np.clip(image, 0, self.data_range)
        image = image.astype(self.image_dtype)
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        return image

    def _evaluate(self, fake_imgs, real_imgs, phase, cnt):
        ssims, mses, maes, psnrs = [], [], [], []

        for ind in range(fake_imgs.shape[0]):
            ssim_channel = []
            mse_channel = []
            mae_channel = []
            psnr_channel = []
            for c in range(fake_imgs.shape[1]):
                pred = fake_imgs[ind, c, :, :].cpu().detach().numpy().copy()
                true = real_imgs[ind, c, :, :].cpu().detach().numpy().copy()

                pred = self._convert_tensor_to_image(pred)
                true = self._convert_tensor_to_image(true)

                ssim = structural_similarity(pred, true, data_range=self.data_range)
                #print('koko',np.max(true),np.max(pred),true.shape,pred.shape)
                mse = mean_squared_error(pred, true)
                mae = mean_absolute_error(pred, true)
                psnr = PSNR(pred, true, data_range=self.data_range)

                ssim_channel.append(ssim)
                mse_channel.append(mse)
                mae_channel.append(mae)
                psnr_channel.append(psnr)

            ssims.append(ssim_channel)
            mses.append(mse_channel)
            maes.append(mae_channel)
            psnrs.append(psnr_channel)

            if phase == 'test':
                with open(self.result_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt,
                                    self.file_list[cnt],
                                    np.mean(ssim_channel),
                                    np.mean(mse_channel),
                                    np.mean(mae_channel),
                                    np.mean(psnr_channel),
                                    ])

        return ssims, mses, maes, psnrs

    def _save_log(self, loss_list_test, evaluates_dict_test):
        # create dict
        result = {}
        # loss in validation
        result['loss_test'] = float(np.mean(loss_list_test))
        result['loss_test_list'] = [float(l) for l in loss_list_test]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_test.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_test[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result[f'{key}_test'] = float(np.mean(data_list))
            result[f'{key}_test_list'] = data_list
            result[f'{key}_test_list_channels'] = data_list_channels

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(result, f, indent=4)

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key) + ": " + "{}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_current_results(self, path, gt_image, visuals):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT-{}'.format(path))
            ret_result.append(gt_image[idx].detach().float().cpu())

            ret_path.append('Process-{}'.format(path))
            ret_result.append(visuals[idx::self.batch_size].detach().float().cpu())

            ret_path.append('Out-{}'.format(path))
            ret_result.append(visuals[idx - self.batch_size].detach().float().cpu())

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)

        return self.results_dict._asdict()

    def _save_images_test(self, save_dir, results):

        ''' get names and corresponding images from results[OrderedDict] '''

        names = results['name']
        outputs = results['result']

        for i in range(len(names)):
            name = names[i]
            id = name[name.find("-")+1:]
            mode = name[:name.find("-")]
            save_dir_each = check_dir(f"{save_dir}/{id}/{mode}")

            im_out = outputs[i].detach().numpy()

            if mode == "Process":
                batchsize = im_out.shape[0]
            else:
                batchsize = 1

            for b in range(batchsize):
                if mode == "Process":
                    im_out_b = im_out[b, :, :, :]

                    b_str = str(b).zfill(len(str(b)))
                    save_dir_each_b = check_dir(f"{save_dir_each}/Process-{b_str}")

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{id}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{id}",
                                            img=im_out_b)
                else:
                    im_out_b = im_out.copy()

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{id}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{id}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each}/Composite")

                        if self.dim_match and self.output_dim_label is not None:
                            if len(self.output_dim_label) > self.out_channels:
                                im_out_b = im_out_b[:self.out_channels, :, :]
                                output_dim_label = self.output_dim_label[:self.out_channels]
                            else:
                                output_dim_label = self.output_dim_label.copy()

                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=output_dim_label,
                                                                 table_artifact=[],
                                                                 flag_artifact=False,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        else:
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=self.table_label,
                                                                 table_artifact=self.table_artifact,
                                                                 flag_artifact=True,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{id}",
                                            img=composite)

'''
cWSB-GP
'''

class cWSBGPTester(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else None
        self.crop_size = kwargs['crop_size'] if 'crop_size' in kwargs else None
        self.device = kwargs['device'] if 'device' in kwargs else None
        self.lamb = torch.tensor(float(kwargs['lamb'])).float().to(torch.device(self.device))
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else None
        self.image_save = kwargs['image_save'] if 'image_save' in kwargs else False
        self.mae_loss = nn.L1Loss(reduction='none')

        self.file_list = kwargs['file_list'] if 'file_list' in kwargs else None
        self.in_channels = kwargs['in_channels'] if 'in_channels' in kwargs else None
        self.out_channels = kwargs['out_channels'] if 'out_channels' in kwargs else None
        self.input_channel_list = kwargs['input_channel_list'] if 'input_channel_list' in kwargs else None
        self.output_channel_list = kwargs['output_channel_list'] if 'output_channel_list' in kwargs else None
        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.result_csv_path = os.path.join(self.save_dir, 'result_at_each_image.csv')
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.eval_metrics = kwargs['eval_metrics'] if 'eval_metrics' in kwargs else None
        self.cond_x1 = kwargs['cond_x1']
        self.add_x1_noise = kwargs['add_x1_noise']
        self.ot_ode = kwargs['ot_ode']
        self.interval = kwargs['interval']
        self.beta_max = kwargs['beta_max']
        self.ema = kwargs['ema'] if 'ema' in kwargs else None
        self.emer = kwargs['emer'] if 'emer' in kwargs else None
        self.diffusion = kwargs['diffusion'] if 'diffusion' in kwargs else None

        self.table_label = kwargs['table_label'] if 'table_label' in kwargs else None
        self.table_artifact = kwargs['table_artifact'] if 'table_artifact' in kwargs else None

        self.dim_match = kwargs['dim_match'] if 'dim_match' in kwargs else None
        self.input_dim_label = kwargs['input_dim_label'] if 'input_dim_label' in kwargs else None
        self.output_dim_label = kwargs['output_dim_label'] if 'output_dim_label' in kwargs else None
        self.distributed = kwargs['distributed'] if 'distributed' in kwargs else False

    def sample_batch(self, data):
        # clean_img is "target" domain and corrupt_img is "source" domain
        img_source, img_target = data
        mask = None

        x0 = img_target.detach().to(self.device)
        x1 = img_source.detach().to(self.device)

        cond = x1.detach() if self.cond_x1 else None

        if self.add_x1_noise: # only for decolor
            x1 = x1 + torch.randn_like(x1)

        assert x0.shape == x1.shape

        return x0, x1, mask, cond

    def _check_flag(self, phase):
        if (phase == 'validation' and self.eval_metrics is not None) or (phase == 'test'):
            return True
        else:
            return False

    def _crop_and_concatinate(self, model, x0, x1, phase, cond, mask):
        # loss
        step = torch.randint(0, self.interval, (x0.shape[0],))

        # test phaseかつcrop augmentationさせてた場合，slideさせて平均を取る
        cropped_x0s = image_sliding_crop(image=x0, crop_size=self.crop_size, slide_mode='half')
        cropped_x1s = image_sliding_crop(image=x1, crop_size=self.crop_size, slide_mode='half')
        cropped_conds = image_sliding_crop(image=cond, crop_size=self.crop_size, slide_mode='half')
        if mask is not None:
            cropped_masks = image_sliding_crop(image=mask, crop_size=self.crop_size, slide_mode='half')
        else:
            cropped_masks = None

        loss_box = []
        if self._check_flag(phase=phase):
            fake_img_box = []
            #visuals_box = []

        for cropped_x0_data, cropped_x1_data, cropped_cond_data in zip(cropped_x0s, cropped_x1s, cropped_conds):
            # parse
            pos = cropped_x0_data['pos']
            # target
            cropped_x0 = cropped_x0_data['data']
            cropped_x0 = cropped_x0.to(torch.device(self.device))
            cropped_x0 = torch.unsqueeze(cropped_x0, dim=0)  # batch
            # source
            cropped_x1 = cropped_x1_data['data']
            cropped_x1 = cropped_x1.to(torch.device(self.device))
            cropped_x1 = torch.unsqueeze(cropped_x1, dim=0)  # batch

            cropped_cond = cropped_cond_data['data']
            cropped_cond = cropped_cond.to(torch.device(self.device))
            cropped_cond = torch.unsqueeze(cropped_cond, dim=0)  # batch

            # inference
            cropped_xt = self.diffusion.q_sample(step, cropped_x0, cropped_x1, ot_ode=self.ot_ode)
            cropped_label = self.compute_label(step, cropped_x0, cropped_xt)
            cropped_pred = model(cropped_xt, step, cond=cropped_cond)
            assert cropped_xt.shape == cropped_label.shape == cropped_pred.shape

            if cropped_masks is not None:
                cropped_mask = [d['data'] if d['pos'] == pos else None for d in cropped_masks][0]
                cropped_pred = cropped_mask * cropped_pred
                cropped_label = cropped_mask * cropped_label
            else:
                cropped_mask = None

            cropped_loss = F.mse_loss(cropped_pred, cropped_label)

            cropped_loss = cropped_loss.to('cpu').clone().detach().numpy()
            loss_box.append(cropped_loss)

            if self._check_flag(phase=phase):
                cropped_fake_imgs, _ = self.ddpm_sampling(
                    model, cropped_x1, mask=cropped_mask, cond=cropped_cond, clip_denoise=True, verbose=False
                )

                cropped_fake_imgs = torch.squeeze(cropped_fake_imgs)
                cropped_fake_imgs = cropped_fake_imgs.to(torch.device(self.device)).clone().detach()

                #cropped_visuals = torch.squeeze(cropped_visuals)
                #cropped_visuals = cropped_visuals.to(torch.device(self.device)).clone().detach()

                fake_img_box.append({'pos': pos, 'data': cropped_fake_imgs})
                #visuals_box.append({'pos': pos, 'data': cropped_visuals})

        loss = float(abs(np.mean(loss_box)))

        if self._check_flag(phase=phase):
            fake_imgs = None
            for ind, process in enumerate(range(fake_img_box[0]['data'].shape[0])):
                fake_imgs_box_modify_each_process = []
                for idx in range(len(fake_img_box)):
                    fake_imgs_process = fake_img_box[idx]['data'][process, :, :, :]
                    fake_imgs_box_modify_each_process.append({'pos': fake_img_box[idx]['pos'], 'data': fake_imgs_process})

                fake_imgs_each_process = concatinate_slides(images=fake_imgs_box_modify_each_process,
                                                            source=x1)
                fake_imgs_each_process = torch.unsqueeze(fake_imgs_each_process, dim=0)  # batch
                fake_imgs_each_process = fake_imgs_each_process.to(torch.device(self.device))
                if ind == 0:
                    fake_imgs = fake_imgs_each_process
                else:
                    fake_imgs = torch.cat([fake_imgs, fake_imgs_each_process], dim=0)

            fake_imgs = fake_imgs.to(torch.device(self.device))

            visuals = None
            # for ind, process in enumerate(range(visuals_box[0]['data'].shape[0])):
            #     visuals_box_modify_each_process = []
            #     for idx in range(len(visuals_box)):
            #         visuals_process = visuals_box[idx]['data'][process, :, :, :]
            #         visuals_box_modify_each_process.append({'pos': visuals_box[idx]['pos'], 'data': visuals_process})
            #
            #     visuals_each_process = concatinate_slides(images=visuals_box_modify_each_process,
            #                                               source=x1)
            #     visuals_each_process = torch.unsqueeze(visuals_each_process, dim=0)  # batch
            #     visuals_each_process = visuals_each_process.to(torch.device(self.device))
            #     if ind == 0:
            #         visuals = visuals_each_process
            #     else:
            #         visuals = torch.cat([visuals, visuals_each_process], dim=0)
            #
            # visuals = visuals.to(torch.device(self.device))
            return torch.tensor(loss), fake_imgs, visuals
        else:
            return torch.tensor(loss), None, None


    def _test_func(self, data, model_G, model_D, epoch, phase, save_dir_img, cnt):
        # parse data
        # x0 target, x1 source
        x0, x1, mask, cond = self.sample_batch(data=data)

        x1 = x1.to(torch.device(self.device))
        x0 = x0.to(torch.device(self.device))

        with torch.no_grad():
            if phase == 'test' and self.crop_size is not None:
                loss, fake_imgs, _ = self._crop_and_concatinate(model_G, x0, x1, phase, cond, mask)
            else:
                # inference
                # loss
                step = torch.randint(0, self.interval, (x0.shape[0],))
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
                label = self.compute_label(step, x0, xt)
                pred = model_G(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape
                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                if self._check_flag(phase=phase):
                    fake_imgs, _ = self.ddpm_sampling(
                        model_G, x1, mask=mask, cond=cond, clip_denoise=True, verbose=False
                    )

                    fake_imgs = fake_imgs.to(torch.device(self.device)).clone().detach()

        # create input sets for discriminator
        if phase == 'test' and self.crop_size is not None:
            fake_imgs_for_evaluate = torch.squeeze(fake_imgs)
            evaluate_idx = 0
            fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate[evaluate_idx, :, :, :], axis=0)
            reals = torch.cat((x1, x0), 1)
            fakes = torch.cat((fake_imgs_for_evaluate, x0), 1)
            loss_mse = F.mse_loss(fake_imgs_for_evaluate, x1)
        else:
            reals = torch.cat((label, x0), 1)
            fakes = torch.cat((pred, x0), 1)
            loss_mse = F.mse_loss(pred, label)

        reals = reals.to(torch.device(self.device))
        fakes = fakes.to(torch.device(self.device))

        # discriminate
        real_d = model_D(reals).mean()
        fake_d = model_D(fakes).mean()

        # calculate gradient penalty
        gradient_penalty = self._calculate_gradient_penalty(model_D, reals, fakes)

        # loss
        loss_D = fake_d - real_d + self.lamb * gradient_penalty
        adversarial_loss = -torch.mean(fake_d)
        if phase == 'test' and self.crop_size is not None:
            image_loss = self.mae_loss(fake_imgs_for_evaluate, x1)
        else:
            image_loss = self.mae_loss(pred, label)
        loss_G = image_loss.mean() + 0.01 * adversarial_loss / (epoch + 1)

        ssims, mses, maes, psnrs = None, None, None, None
        if self._check_flag(phase=phase):
            ### 3. evaluate output
            fake_imgs_for_evaluate = torch.squeeze(fake_imgs)
            evaluate_idx = 0
            fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate[evaluate_idx, :, :, :], axis=0)
            x0_for_evaluate = x0.clone()

            if self.dim_match and self.output_dim_label is not None:
                if len(self.output_dim_label) > self.out_channels:
                    x0_for_evaluate = x0[0, :self.out_channels, :, :]
                    fake_imgs_for_evaluate = fake_imgs_for_evaluate[0, :self.out_channels, :, :]

                    x0_for_evaluate = torch.unsqueeze(x0_for_evaluate, axis=0)
                    fake_imgs_for_evaluate = torch.unsqueeze(fake_imgs_for_evaluate, axis=0)

            if phase == 'test' and self.image_save:
                self._save_images_test(path=self.file_list[cnt],
                                       save_dir=save_dir_img,
                                       real_imgs=x0_for_evaluate.detach().clone().float().cpu(),
                                       fake_imgs=fake_imgs_for_evaluate.detach().clone().float().cpu(),
                                       visuals=None)

            ssims, mses, maes, psnrs = self._evaluate(fake_imgs_for_evaluate, x0_for_evaluate, phase, cnt)
        return loss_mse, loss_G, loss_D, ssims, mses, maes, psnrs

    def test(self, model_G, model_D, data_iter, phase="test", epoch=0):
        start = time.time()
        # turn on the testing mode; clean up the history
        model_G.eval()
        model_D.eval()
        model_G.phase = phase
        model_D.phase = phase

        save_dir_img = None
        if phase == 'test':
            with open(self.result_csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id', 'file_name', 'ssim', 'mse', 'mae', 'psnr'])

            save_dir_img = check_dir(os.path.join(self.save_dir, "images"))

            betas = make_beta_schedule_for_I2SB(n_timestep=self.interval,
                                                linear_end=self.beta_max / self.interval)
            betas = np.concatenate([betas[:self.interval // 2], np.flip(betas[:self.interval // 2])])
            self.diffusion = DiffusionI2SB(betas, self.device)
            self.emer = ExponentialMovingAverage(model_G.parameters(), decay=self.ema)

        if self._check_flag(phase=phase):
            ssim_list = []
            mse_list = []
            mae_list = []
            psnr_list = []

        loss_mse_list = []
        loss_G_list = []
        loss_D_list = []

        # predict
        if phase == 'test':
            tqdm_disable = False
        else:
            tqdm_disable = True

        for i, data in enumerate(tqdm(data_iter, disable=tqdm_disable)):
            output = self._test_func(data, model_G, model_D, epoch, phase, save_dir_img, i)
            loss_mse, loss_G, loss_D, ssims, mses, maes, psnrs = output
            # save results
            loss_mse_list.append(loss_mse.to(torch.device('cpu')).detach().numpy())
            loss_D_list.append(loss_D.to(torch.device('cpu')).detach().numpy())
            loss_G_list.append(loss_G.to(torch.device('cpu')).detach().numpy())
            if self._check_flag(phase=phase):
                ssim_list += ssims
                mse_list += mses
                mae_list += maes
                psnr_list += psnrs

        evaluates_dict = None
        if self._check_flag(phase=phase):
            evaluates_dict = {
                "ssim": ssim_list,
                "mse": mse_list,
                "mae": mae_list,
                "psnr": psnr_list,
            }

        loss_mse_mean = float(abs(np.mean(loss_mse_list)))
        loss_G_mean = float(abs(np.mean(loss_G_list)))
        loss_D_mean = float(abs(np.mean(loss_D_list)))
        elapsed_time = time.time() - start

        if self._check_flag(phase=phase):
            print_text = "[{}] loss mse : {}, loss G: {}, D: {} | {}, elapsed time: {} s". \
                format(phase, loss_mse_mean, loss_G_mean, loss_D_mean, self._print_eval_statics(evaluates_dict),
                       int(np.floor(elapsed_time)))
        else:
            print_text = "[{}] loss mse : {}, loss G: {}, D: {}, elapsed time: {} s". \
                format(phase, loss_mse_mean, loss_G_mean, loss_D_mean, int(np.floor(elapsed_time)))
        print(print_text)

        if phase == 'validation':
            model_G.phase = 'train'
            model_D.phase = 'train'

        elif phase == 'test':
            with open(os.path.join(self.save_dir, "result.txt"), "w") as f:
                f.write(print_text)
            self._save_log(loss_mse_list, loss_G_list, loss_D_list, evaluates_dict)

        return loss_mse_list, loss_G_list, loss_D_list, evaluates_dict

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def _space_indices(self, num_steps, count):
        assert count <= num_steps

        if count <= 1:
            frac_stride = 1
        else:
            frac_stride = (num_steps - 1) / (count - 1)

        cur_idx = 0.0
        taken_steps = []
        for _ in range(count):
            taken_steps.append(round(cur_idx))
            cur_idx += frac_stride

        return taken_steps

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    @torch.no_grad()
    def ddpm_sampling(self, model, x1, mask=None, cond=None, clip_denoise=False, nfe=None, log_count=10, verbose=True):

        # create discrete time steps that split [0, INTERVAL] into NFE sub-intervals.
        # e.g., if NFE=2 & INTERVAL=1000, then STEPS=[0, 500, 999] and 2 network
        # evaluations will be invoked, first from 999 to 500, then from 500 to 0.
        nfe = nfe or self.interval-1
        assert 0 < nfe < self.interval == len(self.diffusion.betas)
        steps = self._space_indices(self.interval, nfe+1)

        # create log steps
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in self._space_indices(len(steps)-1, log_count)]
        assert log_steps[0] == 0
        # print(f"[DDPM Sampling] steps={self.interval}, {nfe=}, {log_steps=}!")
        # self.log.info(f"[DDPM Sampling] steps={opt.interval}, {nfe=}, {log_steps=}!")

        x1 = x1.to(self.device)
        if cond is not None: cond = cond.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)

        with self.emer.average_parameters():
            model.eval()

            def pred_x0_fn(xt, step):
                step = torch.full((xt.shape[0],), step, device=self.device, dtype=torch.long)
                out = model(xt, step, cond=cond)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=self.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        b, *xdim = x1.shape
        assert xs.shape == pred_x0.shape == (b, log_count, *xdim)

        return xs, pred_x0

    def _convert_tensor_to_image(self, image):
        if torch.is_tensor(image):
            image = image.to(torch.device('cpu')).detach().numpy()
        image = np.squeeze(image)
        if self.normalization == 'minmax':
            # 0-1の範囲を0-255(uint8なら)に変換
            image = image * self.data_range
        elif self.normalization == 'zscore':
            # 0-1の範囲を0-255(uint8なら)に変換
            raise NotImplementedError
        elif self.normalization == 'std':
            # -1-1の範囲を0-255(uint8なら)に変換
            image = (image + 1) / 2
            image = image * self.data_range
        else:
            raise NotImplementedError

        image = np.clip(image, 0, self.data_range)
        image = image.astype(self.image_dtype)
        if len(image.shape) == 3:
            image = image.transpose(1, 2, 0)  # CHW to HWC
        return image

    def _calculate_gradient_penalty(self, model_D, real_images, fake_images):
        # generate random eta
        eta = torch.rand(self.batch_size, 1, 1, 1, device=torch.device('cpu')) # self.device
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(torch.device(self.device))
        # calculate interpolated
        interpolated = eta * real_images + ((1 - eta) * fake_images)
        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)
        # calculate probability of interpolated examples
        prob_interpolated = model_D(interpolated)
        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                  grad_outputs=torch.ones(prob_interpolated.size()).to(torch.device(self.device)),
                                  create_graph=True, retain_graph=True)[0]
        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return grad_penalty

    def _evaluate(self, fake_imgs, real_imgs, phase, cnt):
        ssims, mses, maes, psnrs = [], [], [], []

        for ind in range(fake_imgs.shape[0]):
            ssim_channel = []
            mse_channel = []
            mae_channel = []
            psnr_channel = []
            for c in range(fake_imgs.shape[1]):
                pred = fake_imgs[ind, c, :, :].cpu().detach().numpy().copy()
                true = real_imgs[ind, c, :, :].cpu().detach().numpy().copy()

                pred = self._convert_tensor_to_image(pred)
                true = self._convert_tensor_to_image(true)

                ssim = structural_similarity(pred, true, data_range=self.data_range)
                #print('koko',np.max(true),np.max(pred),true.shape,pred.shape)
                mse = mean_squared_error(pred, true)
                mae = mean_absolute_error(pred, true)
                psnr = PSNR(pred, true, data_range=self.data_range)

                ssim_channel.append(ssim)
                mse_channel.append(mse)
                mae_channel.append(mae)
                psnr_channel.append(psnr)

            ssims.append(ssim_channel)
            mses.append(mse_channel)
            maes.append(mae_channel)
            psnrs.append(psnr_channel)

            if phase == 'test':
                with open(self.result_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cnt,
                                    self.file_list[cnt],
                                    np.mean(ssim_channel),
                                    np.mean(mse_channel),
                                    np.mean(mae_channel),
                                    np.mean(psnr_channel),
                                    ])

        return ssims, mses, maes, psnrs

    def _save_log(self, loss_mse_list_test, loss_G_list_test, loss_D_list_test, evaluates_dict_test):
        # create dict
        result = {}
        # loss in validation
        result['loss_mse_test'] = float(np.mean(loss_mse_list_test))
        result['loss_mse_test_list'] = [float(l) for l in loss_mse_list_test]
        result['loss_G_test'] = float(np.mean(loss_G_list_test))
        result['loss_G_test_list'] = [float(l) for l in loss_G_list_test]
        result['loss_D_test'] = float(np.mean(loss_D_list_test))
        result['loss_D_test_list'] = [float(l) for l in loss_D_list_test]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_test.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_test[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result[f'{key}_test'] = float(np.mean(data_list))
            result[f'{key}_test_list'] = data_list
            result[f'{key}_test_list_channels'] = data_list_channels

        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(result, f, indent=4)

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key) + ": " + "{}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_images_test(self, path, save_dir, real_imgs, fake_imgs, visuals):
        def _save_func(name, save_dir_each, data, mode):
            data = torch.squeeze(data.detach().clone())
            save_dir_each = check_dir(f"{save_dir_each}/{mode}")

            if mode in ["GT", "Predict"]:
                batchsize = 1
            else:
                batchsize = data.shape[0]

            for b in range(batchsize):
                if mode in ["GT", "Predict"]:
                    im_out_b = data.detach().clone().cpu()

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{name}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each,
                                            filename=f"{name}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each}/Composite")

                        if self.dim_match and self.output_dim_label is not None:
                            if len(self.output_dim_label) > self.out_channels:
                                im_out_b = im_out_b[:self.out_channels, :, :]
                                output_dim_label = self.output_dim_label[:self.out_channels]
                            else:
                                output_dim_label = self.output_dim_label.copy()
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=output_dim_label,
                                                                 table_artifact=[],
                                                                 flag_artifact=False,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        else:
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=self.table_label,
                                                                 table_artifact=self.table_artifact,
                                                                 flag_artifact=True,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)
                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{name}",
                                            img=composite)
                else:
                    im_out_b = data[b, :, :, :].detach().clone().cpu()

                    b_str = str(b).zfill(len(str(b)))
                    save_dir_each_b = check_dir(f"{save_dir_each}/{mode}-{b_str}")

                    channels = im_out_b.shape[0]
                    for channel in range(channels):
                        im = im_out_b[channel, :, :]
                        im = self._convert_tensor_to_image(im)
                        if len(self.output_channel_list) > 0:
                            channel_name = self.output_channel_list[channel]
                        else:
                            channel_name = channel
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{name}_channel_{channel_name}",
                                            img=im)
                    if im_out_b.shape[0] == 1 or im_out_b.shape[0] == 3:
                        im_out_b = self._convert_tensor_to_image(im_out_b)
                        save_image_function(save_dir=save_dir_each_b,
                                            filename=f"{name}",
                                            img=im_out_b)
                    else:
                        save_dir_img_each_composite = check_dir(f"{save_dir_each_b}/Composite")
                        if self.dim_match and self.output_dim_label is not None:
                            if len(self.output_dim_label) > self.out_channels:
                                im_out_b = im_out_b[:self.out_channels, :, :]
                                output_dim_label = self.output_dim_label[:self.out_channels]
                            else:
                                output_dim_label = self.output_dim_label.copy()

                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=output_dim_label,
                                                                 table_artifact=[],
                                                                 flag_artifact=False,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        else:
                            composite = convert_channels_to_rgbs(images=self._convert_tensor_to_image(im_out_b),
                                                                 table_label=self.table_label,
                                                                 table_artifact=self.table_artifact,
                                                                 flag_artifact=True,
                                                                 data_range=self.data_range,
                                                                 image_dtype=self.image_dtype)

                        save_image_function(save_dir=save_dir_img_each_composite,
                                            filename=f"{name}",
                                            img=composite)
        name = os.path.splitext(os.path.basename(path))[0]
        save_dir_each = check_dir(f"{save_dir}/{name}")

        _save_func(name=name, save_dir_each=save_dir_each, data=real_imgs, mode='GT')
        _save_func(name=name, save_dir_each=save_dir_each, data=fake_imgs, mode='Predict')
        #_save_func(name=name, save_dir_each=save_dir_each, data=visuals, mode='Predict_Visuals')
