import os
import gc
import copy
import json
import time
import datetime
import itertools
import random
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from sklearn import metrics
import torch.autograd as autograd
from torch.autograd import Variable
from src.lib.utils.visualizer import show_loss_GAN, show_loss, show_metrics, \
    show_metrics_validation, show_loss_distributed, show_loss_WSB
from src.lib.runner.tester import cWGANGPTester, guidedI2ITester, I2SBTester, PaletteTester, cWSBGPTester
from src.lib.utils.utils import CustomException, check_dir
from src.lib.losses.losses import GenLoss
from src.lib.models.diffusion_for_I2SB import Diffusion as DiffusionI2SB
from src.lib.optimizers.optimizers import get_optimizer
from src.lib.optimizers.schedulers import get_scheduler
from src.lib.datasets.dataloader import setup_loader

import math
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel as DDP

def PSNR(img_1, img_2, data_range=255):
    mse = np.mean((img_1.astype(float) - img_2.astype(float)) ** 2)
    return 10 * np.log10((data_range ** 2) / mse)


class cWGANGPTrainer(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.optimizer_G = kwargs['optimizer_G']
        self.optimizer_D = kwargs['optimizer_D']
        self.scheduler_G = kwargs['scheduler_G']
        self.scheduler_D = kwargs['scheduler_D']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']
        self.gen_criterion = GenLoss()
        self.gen_freq = kwargs['gen_freq'] if 'gen_freq' in kwargs else 1

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']
        self.save_model_freq = kwargs['save_model_freq'] if 'save_model_freq' in kwargs else None

        self.device = kwargs['device']
        self.seed = kwargs['seed']
        self.reuse = kwargs['reuse'] if 'reuse' in kwargs else False
        if self.reuse:
            with open(os.path.join(self.save_dir, 'last_epoch.json'), 'r') as f:
                last_data = json.load(f)
            self.last_epoch = last_data['last_epoch']

            with open(os.path.join(self.save_dir, 'log.json'), 'r') as f:
                self.results = json.load(f)

            self._best_vals = {}
            self._best_epochs = {}
            for ind, metric in enumerate(self.eval_metrics):
                save_dir_each_metric = os.path.join(self.save_dir, metric)
                with open(os.path.join(save_dir_each_metric, f'best_{metric}_result.json'), 'r') as f:
                    data = json.load(f)
                self._best_epochs[metric] = data['best epoch']
                self._best_vals[metric] = data['best {}'.format(metric)]
        else:
            self.last_epoch = 0
            self._best_vals = {}
            self._best_epochs = {}
            for ind, metric in enumerate(self.eval_metrics):
                self._best_epochs[metric] = 0
                if self.eval_maximizes[ind]:
                    self._best_vals[metric] = 0.0
                else:
                    self._best_vals[metric] = 10.0**13

            self.results = {}

        self.lamb = torch.tensor(float(kwargs['lamb'])).float().to(torch.device(self.device))
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']


    def train(self, model_G, model_D, train_iterator, validation_iterator):

        # create validator
        validator_args = {
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'lamb': self.lamb,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            }

        if self.reuse:
            # final modelでsaveしたrandom_stateは，_calculate_gradient_penalty()のために再度ここで指定する必要がある
            last_data = torch.load(f'{self.save_dir}/last_epoch_object.cpt')

            random.setstate(last_data["random_state"])
            np.random.set_state(last_data['np_random_state'])
            torch.random.set_rng_state(last_data["torch_random_state"])
            if 'torch_cuda_random_state' in last_data:
                torch.cuda.set_rng_state(last_data['torch_cuda_random_state'])

        validator = cWGANGPTester(**validator_args)
        memory_list = []

        for epoch in range(self.last_epoch, self.epoch_num):
            print("Epoch: {}/{}".format(epoch+1, self.epoch_num))

            '''Train'''
            # turn on network training mode
            model_G.train()
            model_D.train()

            # train
            out_train = self._train_step(model_G, model_D, train_iterator, epoch)

            '''Validation'''
            # validation
            out_val = validator.test(model_G, model_D, validation_iterator, phase="validation", epoch=epoch)

            # save logs
            self._save_log(epoch, out_train, out_val)

            # check best model
            self._check_model(model_G, model_D, out_val, epoch)

            # memory manage
            mem = float(torch.cuda.memory_allocated()/1024**2)
            print("memory (MiB): ", mem)
            memory_list.append(mem)

            # save final model
            self._save_final_model(model_G, model_D, train_iterator, epoch)

        memory_list = memory_list[1:]  # ind1は多くなる傾向があるため
        print("Memory increased  (MiB):", np.max(memory_list)-np.min(memory_list))
        show_loss_GAN(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        for metric in self.eval_metrics:
            save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
            show_metrics(eval_metrics=metric,
                         log_path=os.path.join(self.save_dir, "log.json"),
                         save_dir=save_dir_each_metric,
                         show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        print('=' * 100)
        for metric in self.eval_metrics:
            text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
            print(text)
        print('='*100)

    def _train_step(self, model_G, model_D, data_iterator, epoch):
        start = time.time()

        loss_G_list = []
        loss_D_list = []

        ssim_list = []
        mse_list = []
        mae_list = []
        psnr_list = []

        for ind, (input_real, output_real) in enumerate(data_iterator):

            input_real = input_real.to(torch.device(self.device))
            output_real = output_real.to(torch.device(self.device))

            ### 1. train Discriminator first ###
            # set gradient zero
            self.optimizer_D.zero_grad()

            # generate fake image
            with torch.no_grad():
                fake_imgs = model_G(input_real)

            # set real image
            real_imgs = output_real

            # create input sets for discriminator
            real_concat_with_input = torch.cat((real_imgs, input_real), 1)
            fake_concat_with_input = torch.cat((fake_imgs, input_real), 1)
            real_concat_with_input = real_concat_with_input.to(torch.device(self.device))
            fake_concat_with_input = fake_concat_with_input.to(torch.device(self.device))

            # discriminate
            real_d = model_D(real_concat_with_input).mean()
            fake_d = model_D(fake_concat_with_input).mean()

            # calculate gradient penalty
            gradient_penalty = self._calculate_gradient_penalty(model_D,
                                                                real_concat_with_input,
                                                                fake_concat_with_input)
            # loss
            loss_D = fake_d - real_d + self.lamb * gradient_penalty
            loss_D.create_graph = True
            #loss_D.backward(retain_graph=True)
            loss_D.backward(retain_graph=False)

            # update params in discriminator
            self.optimizer_D.step()

            ### 2. train Generator second ###
            # set gradient zero
            self.optimizer_G.zero_grad()

            if ind % self.gen_freq == 0:  # Label-free-predicion-of-Cell-Painting論文では，この制約あり
                # generate fake image
                fake_imgs = model_G(input_real)

                # create input sets for discriminator
                fake_concat_with_input = torch.cat((fake_imgs, input_real), 1)
                fake_concat_with_input = fake_concat_with_input.to(torch.device(self.device))

                # discriminate
                fake_d = model_D(fake_concat_with_input).mean()

                # loss
                loss_G = self.gen_criterion(fake_d, fake_imgs, real_imgs, epoch)
                loss_G.backward()

                # update params in generator
                self.optimizer_G.step()

            ### 3. evaluate output
            ssims, mses, maes, psnrs = self._evaluate(fake_imgs, real_imgs)

            # save results
            loss_D_list.append(loss_D.to(torch.device('cpu')).detach().numpy())
            loss_G_list.append(loss_G.to(torch.device('cpu')).detach().numpy())
            ssim_list += ssims
            mse_list += mses
            mae_list += maes
            psnr_list += psnrs

            # delete loss_D grad object to avoid memory leak
            del loss_D
            torch.cuda.empty_cache()
            gc.collect()

        # if scheduler
        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()

        loss_G_mean = float(abs(np.mean(loss_G_list)))
        loss_D_mean = float(abs(np.mean(loss_D_list)))

        evaluates_dict = {
            "ssim": ssim_list,
            "mse": mse_list,
            "mae": mae_list,
            "psnr": psnr_list,
        }

        elapsed_time = time.time() - start

        print("[train] loss G: {:.4f}, D: {:.4f} | {}, elapsed time: {} s". \
              format(loss_G_mean, loss_D_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time))))

        return loss_G_list, loss_D_list, evaluates_dict

    def _calculate_gradient_penalty(self, model_D, real_images, fake_images):
        # print(torch.random.get_rng_state()[:10])
        # print('cuda',torch.cuda.get_rng_state()[-10:])
        # generate random eta
        eta = torch.rand(self.batch_size, 1, 1, 1, device=torch.device('cpu')) # self.device
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(torch.device(self.device))
        # print(torch.random.get_rng_state()[:10])
        # print('cuda',torch.cuda.get_rng_state()[-10:])
        # print('eta',eta[0][0][0][0])
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

        # delete grad_penalty to avoid memory leak
        del prob_interpolated, interpolated, gradients
        torch.cuda.empty_cache()
        gc.collect()
        return grad_penalty

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

    def _evaluate(self, fake_imgs, real_imgs):
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
        return ssims, mses, maes, psnrs

    def _save_log(self, epoch, out_train, out_val):
        # parse
        loss_G_list_train, loss_D_list_train, evaluates_dict_train = out_train
        loss_G_list_val, loss_D_list_val, evaluates_dict_val = out_val

        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_G_train'] = float(np.mean(loss_G_list_train))
        result_each_epoch['loss_G_train_list'] = [float(l) for l in loss_G_list_train]
        result_each_epoch['loss_D_train'] = float(np.mean(loss_D_list_train))
        result_each_epoch['loss_D_train_list'] = [float(l) for l in loss_D_list_train]

        # ssim, mse, mae, psnr in train
        key_list = list(evaluates_dict_train.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_train[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result_each_epoch[f'{key}_train'] = float(np.mean(data_list))
            result_each_epoch[f'{key}_train_list'] = data_list
            result_each_epoch[f'{key}_train_list_channels'] = data_list_channels

        # loss in validation
        result_each_epoch['loss_G_validation'] = float(np.mean(loss_G_list_val))
        result_each_epoch['loss_G_validation_list'] = [float(l) for l in loss_G_list_val]
        result_each_epoch['loss_D_validation'] = float(np.mean(loss_D_list_val))
        result_each_epoch['loss_D_validation_list'] = [float(l) for l in loss_D_list_val]

        # ssim, mse, mae, psnr in validation
        key_list = list(evaluates_dict_val.keys())
        for key in key_list:
            data_list = []  # channel平均
            data_list_channels = []  # channelごと
            for l in evaluates_dict_val[key]:
                data_list_channels.append([float(ll) for ll in l])
                data_list.append(float(np.mean(l)))

            result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
            result_each_epoch[f'{key}_validation_list'] = data_list
            result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)

    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model_G, model_D, out_val, epoch):
        # calculate statics
        key_list, eval_results_val = self._calculate_statics(out_val=out_val)

        # check best_eval
        for key in key_list:
            flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model_G.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_G.pth'))
                model_G.to(torch.device(self.device))
                torch.save(model_D.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_D.pth'))
                model_D.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]
                save_data_for_best['metrics'] = eval_results_val

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

        return eval_results_val

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        assert eval_metric in eval_results, \
            "Evaluation doesn't contain metric '{}'." \
            .format(eval_metric)

        current_val = eval_results[eval_metric]

        maximize_ind = self.eval_metrics.index(eval_metric)
        maximize_bool = self.eval_maximizes[maximize_ind]

        if maximize_bool:
            if current_val >= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False
        else:
            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model_G, model_D, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model_G.train()
        model_D.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model_G': model_G.state_dict(),
            'model_D': model_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model_G.to(torch.device(self.device))
        model_D.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)


'''
guided-I2I
'''
class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class guidedI2ITrainer(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']

        self.device = kwargs['device']
        self.seed = kwargs['seed']

        self.last_epoch = 0
        self._best_vals = {}
        self._best_epochs = {}
        if self.eval_metrics is not None and self.eval_maximizes is not None:
            for ind, metric in enumerate(self.eval_metrics):
                self._best_epochs[metric] = 0
                if self.eval_maximizes[ind]:
                    self._best_vals[metric] = 0.0
                else:
                    self._best_vals[metric] = 10.0**13
        else:
            self._best_epochs['loss'] = 0
            self._best_vals['loss'] = 10.0**13

        self.results = {}

        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        if "ema_start" in kwargs and "ema_iter"  in kwargs and "ema_decay"  in kwargs:
            self.ema_scheduler = {"ema_start": kwargs["ema_start"],
                                  "ema_iter": kwargs["ema_iter"],
                                  "ema_decay": kwargs["ema_decay"]}
        else:
            self.ema_scheduler = None

        self.iter = 0
        self.sample_num = kwargs['sample_num']
        self.task = kwargs['task']

    def train(self, model, train_iterator, validation_iterator):

        # create validator
        validator_args = {
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            'sample_num': self.sample_num,
            'task': self.task,
            'eval_metrics': self.eval_metrics,
            }

        validator = guidedI2ITester(**validator_args)
        # set EMA
        if self.ema_scheduler is not None:
            self.model_EMA = copy.deepcopy(model)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])

        for epoch in range(self.last_epoch, self.epoch_num):
            print("Epoch: {}/{}".format(epoch+1, self.epoch_num))

            '''Train'''
            # turn on network training mode
            model.train()

            # train
            out_train = self._train_step(model, train_iterator)

            '''Validation'''
            # validation
            out_val = validator.test(model, validation_iterator, phase="validation")

            # save logs
            self._save_log(epoch, out_train, out_val)

            # check best model
            self._check_model(model, out_val, epoch)

            # save final model
            self._save_final_model(model, train_iterator, epoch)

        show_loss(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
                show_metrics_validation(eval_metrics=metric,
                                        log_path=os.path.join(self.save_dir, "log.json"),
                                        save_dir=save_dir_each_metric,
                                        show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        if self.eval_metrics is not None:
            print('=' * 100)
            for metric in self.eval_metrics:
                text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
                print(text)
            print('='*100)

    def _parse_data(self, data):
        image_source, image_target, weak_label = data
        cond_image = image_source.to(torch.device(self.device))
        gt_image = image_target.to(torch.device(self.device))
        weak_label = weak_label.to(self.device)
        return cond_image, gt_image, weak_label, self.batch_size

    def _train_step(self, model, data_iterator):
        start = time.time()

        loss_list = []

        for data in data_iterator:
            # parse data
            cond_image, gt_image, weak_label, batch_size = self._parse_data(data=data)
            self.iter += batch_size

            # set gradient zero
            self.optimizer.zero_grad()

            # loss
            loss = model(gt_image, weak_label, cond_image, mask=None)
            loss.backward()

            # update params in discriminator
            self.optimizer.step()

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())

            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.model_EMA, model)

        # if scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        loss_mean = float(abs(np.mean(loss_list)))

        elapsed_time = time.time() - start

        print("[train] loss: {:.4f}, elapsed time: {} s". \
              format(loss_mean, int(np.floor(elapsed_time))))

        return loss_list, None

    def _save_log(self, epoch, out_train, out_val):
        loss_list_train, _ = out_train
        if self.eval_metrics is not None:
            loss_list_val, evaluates_dict_val = out_val
        else:
            loss_list_val = out_val
        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_train'] = float(np.mean(loss_list_train))
        result_each_epoch['loss_train_list'] = [float(l) for l in loss_list_train]

        # loss in validation
        result_each_epoch['loss_validation'] = float(np.mean(loss_list_val))
        result_each_epoch['loss_validation_list'] = [float(l) for l in loss_list_val]

        if self.eval_metrics is not None:
            # ssim, mse, mae, psnr in validation
            key_list = list(evaluates_dict_val.keys())
            for key in key_list:
                data_list = []  # channel平均
                data_list_channels = []  # channelごと
                for l in evaluates_dict_val[key]:
                    data_list_channels.append([float(ll) for ll in l])
                    data_list.append(float(np.mean(l)))

                result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
                result_each_epoch[f'{key}_validation_list'] = data_list
                result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)


    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model, out_val, epoch):
        if self.eval_metrics is not None:
            # calculate statics
            key_list, eval_results_val = self._calculate_statics(out_val=out_val)

            # check best_eval
            for key in key_list:
                flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
                if flag:
                    save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                    torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                    model.to(torch.device(self.device))

                    save_data_for_best = dict()
                    save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                    save_data_for_best['best epoch'] = self._best_epochs[key]
                    save_data_for_best['metrics'] = eval_results_val

                    with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                        json.dump(save_data_for_best, f, indent=4)
        else:
            loss_val = float(abs(np.mean(out_val)))
            key = 'loss'
            flag = self._best_eval_result(eval_metric=key, eval_results=loss_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                model.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        if self.eval_metrics is not None:
            assert eval_metric in eval_results, \
                "Evaluation doesn't contain metric '{}'." \
                .format(eval_metric)

            current_val = eval_results[eval_metric]

            maximize_ind = self.eval_metrics.index(eval_metric)
            maximize_bool = self.eval_maximizes[maximize_ind]

            if maximize_bool:
                if current_val >= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
            else:
                if current_val <= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
        else:
            if isinstance(eval_results, dict):
                current_val = eval_results[eval_metric]
            elif isinstance(eval_results, float):
                current_val = eval_results
            else:
                raise CustomException(f"Invalid eval_results: {eval_results}")

            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)

'''
I2SB
'''
def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

class I2SBTrainer(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']

        self.device = kwargs['device']
        self.seed = kwargs['seed']

        self._best_vals = {}
        self._best_epochs = {}
        if self.eval_metrics is not None and self.eval_maximizes is not None:
            for ind, metric in enumerate(self.eval_metrics):
                self._best_epochs[metric] = 0
                if self.eval_maximizes[ind]:
                    self._best_vals[metric] = 0.0
                else:
                    self._best_vals[metric] = 10.0**13
        else:
            self._best_epochs['loss'] = 0
            self._best_vals['loss'] = 10.0**13

        self.results = {}

        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.interval = kwargs['interval']
        self.t0 = kwargs['t0']
        self.T = kwargs['T']
        self.cond_x1 = kwargs['cond_x1']
        self.add_x1_noise = kwargs['add_x1_noise']
        self.use_fp16 = kwargs['use_fp16']
        self.ema = kwargs['ema']
        self.global_size = kwargs['global_size']
        self.microbatch = kwargs['microbatch']
        self.ot_ode = kwargs['ot_ode']
        self.beta_max = kwargs['beta_max']
        betas = make_beta_schedule(n_timestep=self.interval,
                                   linear_end=self.beta_max / self.interval)
        betas = np.concatenate([betas[:self.interval//2], np.flip(betas[:self.interval//2])])
        self.diffusion = DiffusionI2SB(betas, self.device)
        self.val_per_epoch = kwargs['val_per_epoch'] if 'val_per_epoch' in kwargs else 1
        self.print_train_loss_per_epoch = kwargs['print_train_loss_per_epoch'] if 'print_train_loss_per_epoch' in kwargs else 1

    def train(self, model, train_iterator, validation_iterator):

        self.emer = ExponentialMovingAverage(model.parameters(), decay=self.ema)

        # create validator
        validator_args = {
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            'eval_metrics': self.eval_metrics,
            'cond_x1': self.cond_x1,
            'add_x1_noise': self.add_x1_noise,
            'ot_ode': self.ot_ode,
            'interval': self.interval,
            'beta_max': self.beta_max,
            'emer': self.emer,
            'diffusion': self.diffusion,
            }

        validator = I2SBTester(**validator_args)

        for epoch in range(self.epoch_num):
            #print("Epoch: {}/{}".format(epoch+1, self.epoch_num))

            '''Train'''
            # turn on network training mode
            model.train()

            # train
            out_train = self._train_step(model, train_iterator)

            if epoch == 0 or epoch % self.print_train_loss_per_epoch == 0:
                today = datetime.datetime.fromtimestamp(time.time())
                print("Epoch: {}/{}, Time: {}".format(epoch + 1, self.epoch_num, today.strftime('%Y/%m/%d %H:%M:%S')))
                loss_mean = float(abs(np.mean(out_train[0])))
                print("[train] loss: {}".format(loss_mean))

            if epoch % self.val_per_epoch == 0:
                '''Validation'''
                # validation
                out_val = validator.test(model, validation_iterator, phase="validation")
            else:
                out_val = None

            # save logs
            self._save_log(epoch, out_train, out_val)

            if out_val is not None:
                # check best model
                self._check_model(model, out_val, epoch)

                # save final model
                #self._save_final_model(model, train_iterator, epoch)

        show_loss_distributed(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
                show_metrics_validation(eval_metrics=metric,
                                        log_path=os.path.join(self.save_dir, "log.json"),
                                        save_dir=save_dir_each_metric,
                                        show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        if self.eval_metrics is not None:
            print('=' * 100)
            for metric in self.eval_metrics:
                text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
                print(text)
            print('='*100)

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

    def _train_step(self, model, data_iterator):
        #start = time.time()

        loss_list = []

        # set gradient zero
        self.optimizer.zero_grad()

        for data in data_iterator:
            # parse data
            # x0 target, x1 source, xt sampled from source
            x0, x1, mask, cond = self.sample_batch(data=data)

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
            loss.backward()

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())

        # update params
        self.optimizer.step()
        self.emer.update()

        # if scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        loss_mean = float(abs(np.mean(loss_list)))

        #elapsed_time = time.time() - start
        # print("[train] loss: {}, elapsed time: {} s". \
        #       format(loss_mean, int(np.floor(elapsed_time))))

        return loss_list, None

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def _save_log(self, epoch, out_train, out_val):
        loss_list_train, _ = out_train
        if out_val is not None:
            if self.eval_metrics is not None:
                loss_list_val, evaluates_dict_val = out_val
            else:
                loss_list_val = out_val

        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_train'] = float(np.mean(loss_list_train))
        result_each_epoch['loss_train_list'] = [float(l) for l in loss_list_train]

        if out_val is not None:
            # loss in validation
            result_each_epoch['loss_validation'] = float(np.mean(loss_list_val))
            result_each_epoch['loss_validation_list'] = [float(l) for l in loss_list_val]

        if out_val is not None and self.eval_metrics is not None:
            # ssim, mse, mae, psnr in validation
            key_list = list(evaluates_dict_val.keys())
            for key in key_list:
                data_list = []  # channel平均
                data_list_channels = []  # channelごと
                for l in evaluates_dict_val[key]:
                    data_list_channels.append([float(ll) for ll in l])
                    data_list.append(float(np.mean(l)))

                result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
                result_each_epoch[f'{key}_validation_list'] = data_list
                result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)


    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model, out_val, epoch):
        if self.eval_metrics is not None:
            # calculate statics
            key_list, eval_results_val = self._calculate_statics(out_val=out_val)

            # check best_eval
            for key in key_list:
                flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
                if flag:
                    save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                    torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                    model.to(torch.device(self.device))

                    save_data_for_best = dict()
                    save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                    save_data_for_best['best epoch'] = self._best_epochs[key]
                    save_data_for_best['metrics'] = eval_results_val

                    with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                        json.dump(save_data_for_best, f, indent=4)
        else:
            loss_val = float(abs(np.mean(out_val)))
            key = 'loss'
            flag = self._best_eval_result(eval_metric=key, eval_results=loss_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                model.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        if self.eval_metrics is not None:
            assert eval_metric in eval_results, \
                "Evaluation doesn't contain metric '{}'." \
                .format(eval_metric)

            current_val = eval_results[eval_metric]

            maximize_ind = self.eval_metrics.index(eval_metric)
            maximize_bool = self.eval_maximizes[maximize_ind]

            if maximize_bool:
                if current_val >= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
            else:
                if current_val <= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
        else:
            if isinstance(eval_results, dict):
                current_val = eval_results[eval_metric]
            elif isinstance(eval_results, float):
                current_val = eval_results
            else:
                raise CustomException(f"Invalid eval_results: {eval_results}")

            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)

class I2SBDistributedTrainer(object):

    def __init__(self, **kwargs):

        self.model_name = kwargs['model']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']

        self.device = kwargs['device']
        self.seed = kwargs['seed']

        self._best_vals = {}
        self._best_epochs = {}
        if self.eval_metrics is not None and self.eval_maximizes is not None:
            for ind, metric in enumerate(self.eval_metrics):
                self._best_epochs[metric] = 0
                if self.eval_maximizes[ind]:
                    self._best_vals[metric] = 0.0
                else:
                    self._best_vals[metric] = 10.0**13
        else:
            self._best_epochs['loss'] = 0
            self._best_vals['loss'] = 10.0**13

        self.results = {}

        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.interval = kwargs['interval']
        self.t0 = kwargs['t0']
        self.T = kwargs['T']
        self.cond_x1 = kwargs['cond_x1']
        self.add_x1_noise = kwargs['add_x1_noise']
        self.use_fp16 = kwargs['use_fp16']
        self.ema = kwargs['ema']
        self.global_size = kwargs['global_size']
        self.microbatch = kwargs['microbatch']
        self.ot_ode = kwargs['ot_ode']
        self.beta_max = kwargs['beta_max']
        betas = make_beta_schedule(n_timestep=self.interval,
                                   linear_end=self.beta_max / self.interval)
        betas = np.concatenate([betas[:self.interval//2], np.flip(betas[:self.interval//2])])
        self.diffusion = DiffusionI2SB(betas, self.device)

    def train(self, args, model, train_dataset, validation_dataset, generator):

        # setting for DDP
        model = DDP(model, device_ids=[int(args.local_rank)])
        self.emer = ExponentialMovingAverage(model.parameters(), decay=self.ema)
        # Initialize an optimizer
        self.optimizer = get_optimizer(args=args, model=model)

        # Initialize an scheduler
        self.scheduler = get_scheduler(args=args, optimizer=self.optimizer)

        train_iterator = setup_loader(train_dataset, int(args.microbatch), generator)
        validation_iterator = setup_loader(validation_dataset, int(args.microbatch), generator)

        # create validator
        validator_args = {
            'distributed': eval(args.distributed),
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            'eval_metrics': self.eval_metrics,
            'cond_x1': self.cond_x1,
            'add_x1_noise': self.add_x1_noise,
            'ot_ode': self.ot_ode,
            'interval': self.interval,
            'beta_max': self.beta_max,
            'emer': self.emer,
            'diffusion': self.diffusion,
            }

        validator = I2SBTester(**validator_args)
        val_per_epoch = int(args.val_per_epoch)

        for epoch in range(self.epoch_num):
            '''Train'''
            # turn on network training mode
            model.train()

            # train
            out_train = self._train_step(model, train_iterator)

            if epoch == 0 or epoch % 1 == 0:
                today = datetime.datetime.fromtimestamp(time.time())
                print("Epoch: {}/{}, Time: {}".format(epoch + 1, self.epoch_num, today.strftime('%Y/%m/%d %H:%M:%S')))
                loss_mean = float(abs(np.mean(out_train[0])))
                print("[train] loss: {}".format(loss_mean))

            if epoch % val_per_epoch == 0:
                torch.distributed.barrier()
                '''Validation'''
                # validation
                out_val = validator.test(model, validation_iterator, phase="validation")
            else:
                out_val = None

            # save logs
            self._save_log(epoch, out_train, out_val)

            if out_val is not None:
                # check best model
                self._check_model(model, out_val, epoch)

                # save final model
                #self._save_final_model(model, train_iterator, epoch)

        show_loss_distributed(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
                show_metrics_validation(eval_metrics=metric,
                                        log_path=os.path.join(self.save_dir, "log.json"),
                                        save_dir=save_dir_each_metric,
                                        show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        if self.eval_metrics is not None:
            print('=' * 100)
            for metric in self.eval_metrics:
                text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
                print(text)
            print('='*100)

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

    def _train_step(self, model, data_iterator):

        loss_list = []

        # set gradient zero
        self.optimizer.zero_grad()

        for data in data_iterator:
            # parse data
            # x0 target, x1 source, xt sampled from source
            x0, x1, mask, cond = self.sample_batch(data=data)

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
            loss.backward()

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())

        # update params
        self.optimizer.step()
        self.emer.update()

        # if scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return loss_list, None

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def _save_log(self, epoch, out_train, out_val):
        loss_list_train, _ = out_train
        if out_val is not None:
            if self.eval_metrics is not None:
                loss_list_val, evaluates_dict_val = out_val
            else:
                loss_list_val = out_val

        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_train'] = float(np.mean(loss_list_train))
        result_each_epoch['loss_train_list'] = [float(l) for l in loss_list_train]

        if out_val is not None:
            # loss in validation
            result_each_epoch['loss_validation'] = float(np.mean(loss_list_val))
            result_each_epoch['loss_validation_list'] = [float(l) for l in loss_list_val]

        if out_val is not None and self.eval_metrics is not None:
            # ssim, mse, mae, psnr in validation
            key_list = list(evaluates_dict_val.keys())
            for key in key_list:
                data_list = []  # channel平均
                data_list_channels = []  # channelごと
                for l in evaluates_dict_val[key]:
                    data_list_channels.append([float(ll) for ll in l])
                    data_list.append(float(np.mean(l)))

                result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
                result_each_epoch[f'{key}_validation_list'] = data_list
                result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)


    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model, out_val, epoch):
        if self.eval_metrics is not None:
            # calculate statics
            key_list, eval_results_val = self._calculate_statics(out_val=out_val)

            # check best_eval
            for key in key_list:
                flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
                if flag:
                    save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                    torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                    model.to(torch.device(self.device))

                    save_data_for_best = dict()
                    save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                    save_data_for_best['best epoch'] = self._best_epochs[key]
                    save_data_for_best['metrics'] = eval_results_val

                    with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                        json.dump(save_data_for_best, f, indent=4)
        else:
            loss_val = float(abs(np.mean(out_val)))
            key = 'loss'
            flag = self._best_eval_result(eval_metric=key, eval_results=loss_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                model.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        if self.eval_metrics is not None:
            assert eval_metric in eval_results, \
                "Evaluation doesn't contain metric '{}'." \
                .format(eval_metric)

            current_val = eval_results[eval_metric]

            maximize_ind = self.eval_metrics.index(eval_metric)
            maximize_bool = self.eval_maximizes[maximize_ind]

            if maximize_bool:
                if current_val >= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
            else:
                if current_val <= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
        else:
            if isinstance(eval_results, dict):
                current_val = eval_results[eval_metric]
            elif isinstance(eval_results, float):
                current_val = eval_results
            else:
                raise CustomException(f"Invalid eval_results: {eval_results}")

            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)

'''
Palette
'''
class EMAPalette():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class PaletteTrainer(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']

        self.device = kwargs['device']
        self.seed = kwargs['seed']

        self.last_epoch = 0
        self._best_vals = {}
        self._best_epochs = {}
        if self.eval_metrics is not None and self.eval_maximizes is not None:
            for ind, metric in enumerate(self.eval_metrics):
                self._best_epochs[metric] = 0
                if self.eval_maximizes[ind]:
                    self._best_vals[metric] = 0.0
                else:
                    self._best_vals[metric] = 10.0**13
        else:
            self._best_epochs['loss'] = 0
            self._best_vals['loss'] = 10.0**13

        self.results = {}

        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        if "ema_start" in kwargs and "ema_iter" in kwargs and "ema_decay" in kwargs:
            self.ema_scheduler = {"ema_start": kwargs["ema_start"],
                                  "ema_iter": kwargs["ema_iter"],
                                  "ema_decay": kwargs["ema_decay"]}
        else:
            self.ema_scheduler = None

        self.iter = 0
        self.sample_num = kwargs['sample_num']
        self.task = kwargs['task']

    def train(self, model, train_iterator, validation_iterator):

        # create validator
        validator_args = {
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            'sample_num': self.sample_num,
            'task': self.task,
            'eval_metrics': self.eval_metrics,
            }

        validator = PaletteTester(**validator_args)

        # set EMA
        if self.ema_scheduler is not None:
            self.model_EMA = copy.deepcopy(model)
            self.EMA = EMAPalette(beta=self.ema_scheduler['ema_decay'])

        for epoch in range(self.last_epoch, self.epoch_num):
            print("Epoch: {}/{}".format(epoch+1, self.epoch_num))

            '''Train'''
            # turn on network training mode
            model.train()

            # train
            out_train = self._train_step(model, train_iterator)

            '''Validation'''
            # validation
            out_val = validator.test(model, validation_iterator, phase="validation")

            # save logs
            self._save_log(epoch, out_train, out_val)

            # check best model
            self._check_model(model, out_val, epoch)

            # save final model
            self._save_final_model(model, train_iterator, epoch)

        show_loss(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
                show_metrics_validation(eval_metrics=metric,
                                        log_path=os.path.join(self.save_dir, "log.json"),
                                        save_dir=save_dir_each_metric,
                                        show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        if self.eval_metrics is not None:
            print('=' * 100)
            for metric in self.eval_metrics:
                text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
                print(text)
            print('='*100)

    def _parse_data(self, data):
        image_source, image_target = data
        image_source = image_source.to(torch.device(self.device))
        image_target = image_target.to(torch.device(self.device))
        return image_source, image_target, self.batch_size

    def _train_step(self, model, data_iterator):
        start = time.time()

        loss_list = []

        for data in data_iterator:
            # parse data
            input_real, real_imgs, batch_size = self._parse_data(data=data)

            # set gradient zero
            self.optimizer.zero_grad()

            # loss
            loss = model(real_imgs, input_real, mask=None)
            loss.backward()

            # update params in discriminator
            self.optimizer.step()

            self.iter += batch_size

            # save results
            loss_list.append(loss.to(torch.device('cpu')).clone().detach().numpy())

            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.model_EMA, model)

        # if scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        loss_mean = float(abs(np.mean(loss_list)))

        elapsed_time = time.time() - start

        print("[train] loss: {:.4f}, elapsed time: {} s". \
              format(loss_mean, int(np.floor(elapsed_time))))

        return loss_list, None

    def _save_log(self, epoch, out_train, out_val):
        loss_list_train, _ = out_train
        if self.eval_metrics is not None:
            loss_list_val, evaluates_dict_val = out_val
        else:
            loss_list_val = out_val
        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_train'] = float(np.mean(loss_list_train))
        result_each_epoch['loss_train_list'] = [float(l) for l in loss_list_train]

        # loss in validation
        result_each_epoch['loss_validation'] = float(np.mean(loss_list_val))
        result_each_epoch['loss_validation_list'] = [float(l) for l in loss_list_val]

        if self.eval_metrics is not None:
            # ssim, mse, mae, psnr in validation
            key_list = list(evaluates_dict_val.keys())
            for key in key_list:
                data_list = []  # channel平均
                data_list_channels = []  # channelごと
                for l in evaluates_dict_val[key]:
                    data_list_channels.append([float(ll) for ll in l])
                    data_list.append(float(np.mean(l)))

                result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
                result_each_epoch[f'{key}_validation_list'] = data_list
                result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)


    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model, out_val, epoch):
        if self.eval_metrics is not None:
            # calculate statics
            key_list, eval_results_val = self._calculate_statics(out_val=out_val)

            # check best_eval
            for key in key_list:
                flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
                if flag:
                    save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                    torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                    model.to(torch.device(self.device))

                    save_data_for_best = dict()
                    save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                    save_data_for_best['best epoch'] = self._best_epochs[key]
                    save_data_for_best['metrics'] = eval_results_val

                    with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                        json.dump(save_data_for_best, f, indent=4)
        else:
            loss_val = float(abs(np.mean(out_val)))
            key = 'loss'
            flag = self._best_eval_result(eval_metric=key, eval_results=loss_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model.pth'))
                model.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        if self.eval_metrics is not None:
            assert eval_metric in eval_results, \
                "Evaluation doesn't contain metric '{}'." \
                .format(eval_metric)

            current_val = eval_results[eval_metric]

            maximize_ind = self.eval_metrics.index(eval_metric)
            maximize_bool = self.eval_maximizes[maximize_ind]

            if maximize_bool:
                if current_val >= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
            else:
                if current_val <= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
        else:
            if isinstance(eval_results, dict):
                current_val = eval_results[eval_metric]
            elif isinstance(eval_results, float):
                current_val = eval_results
            else:
                raise CustomException(f"Invalid eval_results: {eval_results}")

            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False

    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)

'''
cWSB-GP
'''
class cWSBGPTrainer(object):

    def __init__(self, **kwargs):
        self.model_name = kwargs['model']
        self.optimizer_G = kwargs['optimizer_G']
        self.optimizer_D = kwargs['optimizer_D']
        self.scheduler_G = kwargs['scheduler_G']
        self.scheduler_D = kwargs['scheduler_D']
        self.epoch_num = kwargs['epoch']
        self.batch_size = kwargs['batch_size']
        self.gen_freq = kwargs['gen_freq'] if 'gen_freq' in kwargs else 1
        self.mae_loss = nn.L1Loss(reduction='none')

        self.save_dir = kwargs['save_dir']
        self.eval_metrics = kwargs['eval_metrics']
        self.eval_maximizes = kwargs['eval_maximize']
        self.save_model_freq = kwargs['save_model_freq'] if 'save_model_freq' in kwargs else None

        self.device = kwargs['device']
        self.seed = kwargs['seed']
        self.reuse = kwargs['reuse'] if 'reuse' in kwargs else False
        if self.reuse:
            with open(os.path.join(self.save_dir, 'last_epoch.json'), 'r') as f:
                last_data = json.load(f)
            self.last_epoch = last_data['last_epoch']

            with open(os.path.join(self.save_dir, 'log.json'), 'r') as f:
                self.results = json.load(f)

            self._best_vals = {}
            self._best_epochs = {}
            for ind, metric in enumerate(self.eval_metrics):
                save_dir_each_metric = os.path.join(self.save_dir, metric)
                with open(os.path.join(save_dir_each_metric, f'best_{metric}_result.json'), 'r') as f:
                    data = json.load(f)
                self._best_epochs[metric] = data['best epoch']
                self._best_vals[metric] = data['best {}'.format(metric)]
        else:
            self.last_epoch = 0
            self._best_vals = {}
            self._best_epochs = {}
            if self.eval_metrics is not None and self.eval_maximizes is not None:
                for ind, metric in enumerate(self.eval_metrics):
                    self._best_epochs[metric] = 0
                    if self.eval_maximizes[ind]:
                        self._best_vals[metric] = 0.0
                    else:
                        self._best_vals[metric] = 10.0 ** 13
            else:
                self._best_epochs['loss'] = 0
                self._best_vals['loss'] = 10.0 ** 13

            self.results = {}

        self.lamb = torch.tensor(float(kwargs['lamb'])).float().to(torch.device(self.device))
        self.image_dtype = kwargs['image_dtype']
        self.data_range = kwargs['data_range']
        self.normalization = kwargs['normalization']

        self.interval = kwargs['interval']
        self.t0 = kwargs['t0']
        self.T = kwargs['T']
        self.cond_x1 = kwargs['cond_x1']
        self.add_x1_noise = kwargs['add_x1_noise']
        self.use_fp16 = kwargs['use_fp16']
        self.ema = kwargs['ema']
        self.global_size = kwargs['global_size']
        self.microbatch = kwargs['microbatch']
        self.ot_ode = kwargs['ot_ode']
        self.beta_max = kwargs['beta_max']
        betas = make_beta_schedule(n_timestep=self.interval,
                                   linear_end=self.beta_max / self.interval)
        betas = np.concatenate([betas[:self.interval//2], np.flip(betas[:self.interval//2])])
        self.diffusion = DiffusionI2SB(betas, self.device)
        self.val_per_epoch = kwargs['val_per_epoch'] if 'val_per_epoch' in kwargs else 1
        self.print_train_loss_per_epoch = kwargs['print_train_loss_per_epoch'] if 'print_train_loss_per_epoch' in kwargs else 1


    def train(self, model_G, model_D, train_iterator, validation_iterator):

        self.emer = ExponentialMovingAverage(model_G.parameters(), decay=self.ema)

        # create validator
        validator_args = {
            'model': self.model_name,
            'save_dir': self.save_dir,
            'device': self.device,
            'lamb': self.lamb,
            'batch_size': self.batch_size,
            'image_dtype': self.image_dtype,
            'data_range': self.data_range,
            'normalization': self.normalization,
            'eval_metrics': self.eval_metrics,
            'cond_x1': self.cond_x1,
            'add_x1_noise': self.add_x1_noise,
            'ot_ode': self.ot_ode,
            'interval': self.interval,
            'beta_max': self.beta_max,
            'emer': self.emer,
            'diffusion': self.diffusion,
            }

        if self.reuse:
            # final modelでsaveしたrandom_stateは，_calculate_gradient_penalty()のために再度ここで指定する必要がある
            last_data = torch.load(f'{self.save_dir}/last_epoch_object.cpt')

            random.setstate(last_data["random_state"])
            np.random.set_state(last_data['np_random_state'])
            torch.random.set_rng_state(last_data["torch_random_state"])
            if 'torch_cuda_random_state' in last_data:
                torch.cuda.set_rng_state(last_data['torch_cuda_random_state'])

        validator = cWSBGPTester(**validator_args)

        for epoch in range(self.last_epoch, self.epoch_num):
            '''Train'''
            # turn on network training mode
            model_G.train()
            model_D.train()

            # train
            out_train = self._train_step(model_G, model_D, train_iterator, epoch)

            if epoch == 0 or epoch % self.print_train_loss_per_epoch == 0:
                today = datetime.datetime.fromtimestamp(time.time())
                print("Epoch: {}/{}, Time: {}".format(epoch + 1, self.epoch_num, today.strftime('%Y/%m/%d %H:%M:%S')))
                loss_G_mean = float(abs(np.mean(out_train[0])))
                loss_D_mean = float(abs(np.mean(out_train[1])))
                print("[train] loss G: {:.4f}, loss D: {:.4f}".format(loss_G_mean, loss_D_mean))

            if epoch % self.val_per_epoch == 0:
                '''Validation'''
                # validation
                out_val = validator.test(model_G, model_D, validation_iterator, phase="validation")
            else:
                out_val = None

            # save logs
            self._save_log(epoch, out_train, out_val)

            if out_val is not None:
                # check best model
                self._check_model(model_G, model_D,  out_val, epoch)

                # save final model
                #self._save_final_model(model, train_iterator, epoch)

        show_loss_WSB(
            log_path=os.path.join(self.save_dir, "log.json"),
            save_dir=self.save_dir,
            filename="losses",
            show_mode=False
        )
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, metric))
                show_metrics(eval_metrics=metric,
                             log_path=os.path.join(self.save_dir, "log.json"),
                             save_dir=save_dir_each_metric,
                             show_mode=False)

        print("Training was finished")
        print(f"Saved directory:")
        print(str(self.save_dir))
        if self.eval_metrics is not None:
            print('=' * 100)
            for metric in self.eval_metrics:
                text = 'best {}: {} at epoch {}'.format(metric, self._best_vals[metric], self._best_epochs[metric])
                print(text)
            print('='*100)

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

    def _train_step(self, model_G, model_D, data_iterator, epoch):

        loss_G_list = []
        loss_D_list = []

        for ind, data in enumerate(data_iterator):
            # parse data
            # x0 target, x1 source, xt sampled from source
            x0, x1, mask, cond = self.sample_batch(data=data)

            x1 = x1.to(torch.device(self.device))
            x0 = x0.to(torch.device(self.device))

            ### 1. train Discriminator first ###
            # set gradient zero
            self.optimizer_D.zero_grad()

            # generate fake image
            with torch.no_grad():
                step = torch.randint(0, self.interval, (x0.shape[0],))
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
                label = self.compute_label(step, x0, xt)
                pred = model_G(xt, step, cond=cond)

            # create input sets for discriminator
            reals = torch.cat((label, x1), 1)
            fakes = torch.cat((pred, x1), 1)
            reals = reals.to(torch.device(self.device))
            fakes = fakes.to(torch.device(self.device))

            # discriminate
            real_d = model_D(reals).mean()
            fake_d = model_D(fakes).mean()

            # calculate gradient penalty
            gradient_penalty = self._calculate_gradient_penalty(model_D,reals,fakes)
            # loss
            loss_D = fake_d - real_d + self.lamb * gradient_penalty
            loss_D.create_graph = True
            loss_D.backward(retain_graph=False)

            # update params in discriminator
            self.optimizer_D.step()

            ### 2. train Generator second ###
            # set gradient zero
            self.optimizer_G.zero_grad()

            if ind % self.gen_freq == 0:
                # loss
                step = torch.randint(0, self.interval, (x0.shape[0],))
                xt = self.diffusion.q_sample(step, x0, x1, ot_ode=self.ot_ode)
                label = self.compute_label(step, x0, xt)
                pred = model_G(xt, step, cond=cond)
                assert xt.shape == label.shape == pred.shape
                if mask is not None:
                    pred = mask * pred
                    label = mask * label

                # create input sets for discriminator
                fakes = torch.cat((pred, x1), 1)
                fakes = fakes.to(torch.device(self.device))

                # discriminate
                fake_d = model_D(fakes).mean()

                # loss
                adversarial_loss = -torch.mean(fake_d)
                image_loss = self.mae_loss(pred, label)
                loss_G = image_loss.mean() + 0.01 * adversarial_loss / (epoch + 1)
                loss_G.backward()

                # update params
                self.optimizer_G.step()
                self.emer.update()

            # save results
            loss_D_list.append(loss_D.to(torch.device('cpu')).detach().numpy())
            loss_G_list.append(loss_G.to(torch.device('cpu')).detach().numpy())

            # delete loss_D grad object to avoid memory leak
            del loss_D
            torch.cuda.empty_cache()
            gc.collect()

        # if scheduler
        if self.scheduler_G is not None:
            self.scheduler_G.step()
        if self.scheduler_D is not None:
            self.scheduler_D.step()

        # loss_G_mean = float(abs(np.mean(loss_G_list)))
        # loss_D_mean = float(abs(np.mean(loss_D_list)))

        # elapsed_time = time.time() - start
        #
        # print("[train] loss G: {:.4f}, D: {:.4f} | {}, elapsed time: {} s". \
        #       format(loss_G_mean, loss_D_mean, self._print_eval_statics(evaluates_dict), int(np.floor(elapsed_time))))

        return loss_G_list, loss_D_list, None

    def _calculate_gradient_penalty(self, model_D, real_images, fake_images):
        # print(torch.random.get_rng_state()[:10])
        # print('cuda',torch.cuda.get_rng_state()[-10:])
        # generate random eta
        eta = torch.rand(self.batch_size, 1, 1, 1, device=torch.device('cpu')) # self.device
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        eta = eta.to(torch.device(self.device))
        # print(torch.random.get_rng_state()[:10])
        # print('cuda',torch.cuda.get_rng_state()[-10:])
        # print('eta',eta[0][0][0][0])
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

        # delete grad_penalty to avoid memory leak
        del prob_interpolated, interpolated, gradients
        torch.cuda.empty_cache()
        gc.collect()
        return grad_penalty

    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

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

    def _evaluate(self, fake_imgs, real_imgs):
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
        return ssims, mses, maes, psnrs

    def _save_log(self, epoch, out_train, out_val):
        # parse
        loss_G_list_train, loss_D_list_train, _ = out_train
        if out_val is not None:
            loss_mse_list_val, loss_G_list_val, loss_D_list_val, evaluates_dict_val = out_val

        # create dict
        result_each_epoch = {}
        result_each_epoch['epoch'] = epoch + 1

        # loss in train
        result_each_epoch['loss_G_train'] = float(np.mean(loss_G_list_train))
        result_each_epoch['loss_G_train_list'] = [float(l) for l in loss_G_list_train]
        result_each_epoch['loss_D_train'] = float(np.mean(loss_D_list_train))
        result_each_epoch['loss_D_train_list'] = [float(l) for l in loss_D_list_train]

        if out_val is not None:
            # loss in validation
            result_each_epoch['loss_mse_validation'] = float(np.mean(loss_mse_list_val))
            result_each_epoch['loss_mse_validation_list'] = [float(l) for l in loss_mse_list_val]
            result_each_epoch['loss_G_validation'] = float(np.mean(loss_G_list_val))
            result_each_epoch['loss_G_validation_list'] = [float(l) for l in loss_G_list_val]
            result_each_epoch['loss_D_validation'] = float(np.mean(loss_D_list_val))
            result_each_epoch['loss_D_validation_list'] = [float(l) for l in loss_D_list_val]

        if out_val is not None and self.eval_metrics is not None:
            # ssim, mse, mae, psnr in validation
            key_list = list(evaluates_dict_val.keys())
            for key in key_list:
                data_list = []  # channel平均
                data_list_channels = []  # channelごと
                for l in evaluates_dict_val[key]:
                    data_list_channels.append([float(ll) for ll in l])
                    data_list.append(float(np.mean(l)))

                result_each_epoch[f'{key}_validation'] = float(np.mean(data_list))
                result_each_epoch[f'{key}_validation_list'] = data_list
                result_each_epoch[f'{key}_validation_list_channels'] = data_list_channels

        self.results[epoch+1] = result_each_epoch
        with open(os.path.join(self.save_dir, 'log.json'), 'w') as f:
            json.dump(self.results, f, indent=4)

    def _calculate_statics(self, out_val):
        # parse out_val to evaluate if best
        _, _, eval_results_val_list_dict = out_val

        # calculate statics
        eval_results_val = {}
        key_list = list(eval_results_val_list_dict.keys())
        for key, value in eval_results_val_list_dict.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            eval_results_val[key] = float(vals_mean)
        return key_list, eval_results_val

    def _check_model(self, model_G, model_D, out_val, epoch):

        loss_mse_list_val, loss_G_list_val, loss_D_list_val, evaluates_dict_val = out_val

        if self.eval_metrics is not None:
            # calculate statics
            key_list, eval_results_val = self._calculate_statics(out_val=out_val)

            # check best_eval
            for key in key_list:
                flag = self._best_eval_result(eval_metric=key, eval_results=eval_results_val, epoch=epoch)
                if flag:
                    save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                    torch.save(model_G.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_G.pth'))
                    model_G.to(torch.device(self.device))
                    torch.save(model_D.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_D.pth'))
                    model_D.to(torch.device(self.device))

                    save_data_for_best = dict()
                    save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                    save_data_for_best['best epoch'] = self._best_epochs[key]
                    save_data_for_best['metrics'] = eval_results_val

                    with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                        json.dump(save_data_for_best, f, indent=4)
        else:
            loss_val = float(abs(np.mean(loss_mse_list_val)))
            key = 'loss'
            flag = self._best_eval_result(eval_metric=key, eval_results=loss_val, epoch=epoch)
            if flag:
                save_dir_each_metric = check_dir(os.path.join(self.save_dir, key))
                torch.save(model_G.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_G.pth'))
                model_G.to(torch.device(self.device))
                torch.save(model_D.to('cpu'), os.path.join(save_dir_each_metric, f'best_{key}_model_D.pth'))
                model_D.to(torch.device(self.device))

                save_data_for_best = dict()
                save_data_for_best['best {}'.format(key)] = self._best_vals[key]
                save_data_for_best['best epoch'] = self._best_epochs[key]

                with open(os.path.join(save_dir_each_metric, f'best_{key}_result.json'), 'w') as f:
                    json.dump(save_data_for_best, f, indent=4)

    def _best_eval_result(self, eval_metric, eval_results, epoch):
        if self.eval_metrics is not None:
            assert eval_metric in eval_results, \
                "Evaluation doesn't contain metric '{}'." \
                .format(eval_metric)

            current_val = eval_results[eval_metric]

            maximize_ind = self.eval_metrics.index(eval_metric)
            maximize_bool = self.eval_maximizes[maximize_ind]

            if maximize_bool:
                if current_val >= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
            else:
                if current_val <= self._best_vals[eval_metric]:
                    self._best_vals[eval_metric] = current_val
                    self._best_epochs[eval_metric] = epoch + 1
                    return True
                else:
                    return False
        else:
            if isinstance(eval_results, dict):
                current_val = eval_results[eval_metric]
            elif isinstance(eval_results, float):
                current_val = eval_results
            else:
                raise CustomException(f"Invalid eval_results: {eval_results}")

            if current_val <= self._best_vals[eval_metric]:
                self._best_vals[eval_metric] = current_val
                self._best_epochs[eval_metric] = epoch + 1
                return True
            else:
                return False
    def _print_eval_statics(self, results):
        out = []
        for key, value in results.items():
            vals = []
            for channel_value in value:
                val = np.mean(channel_value)
                vals.append(val)
            vals_mean = np.mean(vals)
            text = str(key).replace("", "") + ": " + "{:.4f}".format(vals_mean)
            out.append(text)
        return ", ".join(out)

    def _save_final_model(self, model_G, model_D, train_iterator, epoch):
        # 学習再開用にtrain modeに切り替え
        model_G.train()
        model_D.train()

        # save final model
        last_ckpt = {
            'epoch': epoch + 1,
            'random_state': random.getstate(),
            'np_random_state': np.random.get_state(legacy=False),
            'torch_random_state': torch.random.get_rng_state(),
            'torch_generator_random_state': train_iterator.generator.get_state(),
            'model_G': model_G.state_dict(),
            'model_D': model_D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }

        if self.device == torch.device('cuda') or self.device == torch.device('mps'):
            last_ckpt['torch_cuda_random_state'] = torch.cuda.get_rng_state()
        torch.save(last_ckpt, os.path.join(self.save_dir, f'last_epoch_object.cpt'))

        model_G.to(torch.device(self.device))
        model_D.to(torch.device(self.device))

        save_data_for_last = dict()
        save_data_for_last['last_epoch'] = epoch + 1

        with open(os.path.join(self.save_dir, f'last_epoch.json'), 'w') as f:
            json.dump(save_data_for_last, f, indent=4)