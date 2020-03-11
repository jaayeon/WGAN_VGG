import glob
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch

import torch
import torch.nn as nn
import torch.optim as optim

from prep import printProgressBar
from networks import WGAN_VGG, WGAN_VGG_generator
from metric import compute_measure

from data.loader import ImageDataset
from torch.utils.data import Dataset, DataLoader

import datetime
import json

import os
from skimage.external.tifffile import imsave, imread
import data.make_patches as mp

class Solver(object):
    def __init__(self, args, data_loader, test_img_list):
        
        self.opt = args
        self.resume = args.resume
        self.mode = args.mode
        self.load_mode = args.load_mode
        self.data_loader = data_loader
        self.test_list, self.test_gt_list = test_img_list
        self.start_epoch = 1

        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mse_criterion = nn.MSELoss()
        self.norm_range_min = args.norm_range_min
        self.norm_range_max = args.norm_range_max
        self.trunc_min = args.trunc_min
        self.trunc_max = args.trunc_max

        self.checkpoint_dir = args.checkpoint_dir
        self.multi_gpu = args.multi_gpu

        self.num_epochs = args.num_epochs
        self.print_iters = args.print_iters
        self.decay_iters = args.decay_iters
        self.save_iters = args.save_iters
        self.test_iters = args.test_iters
        self.result_fig = args.result_fig

        self.n_d_train = args.n_d_train

        self.patch_n = args.patch_n
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size

        self.lr = args.lr
        self.lambda_ = args.lambda_

        self.WGANVGG = WGAN_VGG(input_size=args.patch_size if args.patch_n else 512)

        if (self.multi_gpu) and (torch.cuda.device_count() > 1):
            print('Use {} GPUs'.format(torch.cuda.device_count()))
            self.WGANVGG = nn.DataParallel(self.WGANVGG)

        self.WGANVGG.to(self.device)

        self.criterion_perceptual = nn.L1Loss()
        self.optimizer_g = optim.Adam(self.WGANVGG.generator.parameters(), self.lr)
        self.optimizer_d = optim.Adam(self.WGANVGG.discriminator.parameters(), self.lr)


    def select_checkpoint_dir(self):
        checkpoint_dir = self.opt.checkpoint_dir
        print(checkpoint_dir)
        dirs = os.listdir(checkpoint_dir)

        for i, d in enumerate(dirs, 0):
            print("(%d) %s" % (i, d))
        d_idx = input("Select directory that you want to load: ")

        path_opt = dirs[int(d_idx)]
        self.opt.path_opt = path_opt

        checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, dirs[int(d_idx)]))
        self.opt.checkpoint_dir = checkpoint_dir
        print("checkpoint_dir is: {}".format(checkpoint_dir))

        return checkpoint_dir

    # def save_model(self, iter_, loss_=None):
    #     f = os.path.join(self.checkpoint_dir, 'WGANVGG_{}iter.ckpt'.format(iter_))
    #     if not os.path.exists(f):
    #         os.makedirs(f)
    #     torch.save(self.WGANVGG.state_dict(), f)
    #     if loss_:
    #         f_loss = os.path.join(self.checkpoint_dir, 'WGANVGG_loss_{}iter.npy'.format(iter_))
    #         if not os.path.exists(f_loss):
    #             os.makedirs(f_loss)
    #         np.save(f_loss, np.array(loss_))

    def save_model(self, epoch, loss_=None):
        f = os.path.join(self.opt.checkpoint_dir, 'WGANVGG_%depoch_loss_%.5f.pth'%(epoch, loss_))
        # if not os.path.exists(f):
        #     os.makedirs(f)
        torch.save(self.WGANVGG.state_dict(), f)
        # if loss_:
        #     f_loss = os.path.join(self.checkpoint_dir, 'WGANVGG_loss_{}iter.npy'.format(iter_))
        #     if not os.path.exists(f_loss):
        #         os.makedirs(f_loss)
        #     np.save(f_loss, np.array(loss_))


    def load_model(self):
        checkpoint_dir = self.select_checkpoint_dir()
        # f = os.path.join(self.checkpoint_dir, 'WGANVGG_{}iter.ckpt'.format(iter_))

        checkpoint_list = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
        checkpoint_list.sort()

        #set log file
        log_file = glob.glob(os.path.join(checkpoint_dir, '*.csv'))
        self.opt.log_file = str(log_file[0])
        
        if self.opt.resume : # resume 
            f = checkpoint_list[len(checkpoint_list)-1]
            self.start_epoch = int(f.split('_')[1][:-5])
            model = torch.load(f)
            self.WGANVGG.load_state_dict(model)

        if self.opt.mode == 'test':
            if self.opt.resume_best : # resume_best
                loss_list = list(map(lambda x : float(os.path.basename(x).split('_')[-1][:-4]), checkpoint_list))
                best_loss_idx = loss_list.index(min(loss_list))
                f =checkpoint_list[best_loss_idx]
            else : # last checkpoint
                f = checkpoint_list[len(checkpoint_list)-1]

            generator_w = {k[10:]:torch.load(f)[k] for k in  list(torch.load(f).keys()) if 'generator' in k}
            
            if self.multi_gpu and (torch.cuda.device_count() > 1):
                state_d = OrderedDict()
                for k, v in generator_w:
                    n = k[7:]
                    state_d[n] = v
                self.WGANVGG_G.load_state_dict(state_d)
            else:
                self.WGANVGG_G.load_state_dict(generator_w)
            

    def set_checkpoint_dir(self):
        dt = datetime.datetime.now()
        date = dt.strftime("%Y%m%d")
        model_opt = self.opt.dataset + "-" + date + "-" + self.opt.model + '-lr' + str(self.opt.lr) 

        self.opt.checkpoint_dir = os.path.join(self.opt.checkpoint_dir, model_opt)
        if not os.path.exists(self.opt.checkpoint_dir):
            os.makedirs(self.opt.checkpoint_dir)
        log_file = os.path.join(self.opt.checkpoint_dir, self.opt.model + '_log.csv')
        self.opt.log_file = log_file
    

    def save_config(self):
        config_file = os.path.join(self.opt.checkpoint_dir, "config.txt")
        with open(config_file, 'w') as f:
            json.dump(self.opt.__dict__, f, indent=2)

    def lr_decay(self):
        lr = self.lr * 0.5
        for param_group in self.optimizer_g.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_d.param_groups:
            param_group['lr'] = lr


    def denormalize_(self, image):
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def normalize_(self, image):
        min_t = self.trunc_min
        max_t = self.trunc_max
        image = (image-min_t)/(max_t-min_t)
        return image


    def trunc(self, mat):
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat


    def save_fig(self, x, y, pred, fig_name, original_result, pred_result):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.checkpoint_dir, 'fig', 'result_{}.png'.format(fig_name)))
        plt.close()


    def train(self):
        train_losses = []
        total_iters = 0
        start_time = time.time()

        if not self.resume : 
            self.set_checkpoint_dir()
            with open(self.opt.log_file, mode = 'w') as f :
                f.write('epoch, train__G_loss, train__P_loss, train__D_loss, train__GP_loss, PSNR, SSIM\n')
            self.save_config()
        else : 
            # self.set_checkpoint_dir()
            self.load_model()

        for epoch in range(self.start_epoch, self.num_epochs):

            total_d_loss = 0.0
            total_g_loss = 0.0
            total_p_loss = 0.0
            total_gp_loss = 0.0

            for iter_, (x, y) in enumerate(self.data_loader):
                total_iters += 1

                x = x.float().to(self.device)
                y = y.float().to(self.device)

                # add 1 channel
                # x = x.unsqueeze(0).float().to(self.device)
                # y = y.unsqueeze(0).float().to(self.device)
                # # patch training
                # if self.patch_size:
                #     x = x.view(-1, 1, self.patch_size, self.patch_size)
                #     y = y.view(-1, 1, self.patch_size, self.patch_size)

                # discriminator
                self.optimizer_d.zero_grad()
                self.WGANVGG.discriminator.zero_grad()
                for _ in range(self.n_d_train):
                    d_loss, gp_loss = self.WGANVGG.d_loss(x, y, gp=True, return_gp=True)
                    d_loss.backward()
                    self.optimizer_d.step()

                # generator, perceptual loss
                self.optimizer_g.zero_grad()
                self.WGANVGG.generator.zero_grad()
                g_loss, p_loss = self.WGANVGG.g_loss(x, y, perceptual=True, return_p=True)
                g_loss.backward()
                self.optimizer_g.step()

                train_losses.append([g_loss.item()-p_loss.item(), p_loss.item(),
                                     d_loss.item()-gp_loss.item(), gp_loss.item()])

                # print
                if total_iters % self.print_iters == 0:
                    print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}], TIME [{:.1f}s] >>> G_LOSS: {:.8f}, P_LOSS: {:.8f}, D_LOSS: {:.8f}, GD_LOSS: {:.8f}".format(total_iters, epoch, self.num_epochs, iter_ + 1, len(self.data_loader), time.time() - start_time, 
                            g_loss.item()-p_loss.item()*0.1, p_loss.item(), d_loss.item()-gp_loss.item(), gp_loss.item()))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.lr_decay()
                # save model
                # if total_iters % self.save_iters == 0:
                #     self.save_model(total_iters, g_loss.item())
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                total_p_loss += p_loss.item()
                total_gp_loss += gp_loss.item()

            #save model
            self.save_model(epoch, g_loss.item())

            pred = self.WGANVGG.generator(x)
            original_result, pred_result = compute_measure(x, y, pred, 1)

            op, oos, _ = original_result
            pp, ps, _ = pred_result
            print("((ORIGIN)) PSNR : {:.5f}, SSIM : {:.5f}, ((PREP)) PSNR : {:.5f}, SSIM : {:.5f}".format(op,oos,pp,ps))

            total_d_loss = total_d_loss/iter_
            total_g_loss  = total_g_loss/iter_
            total_p_loss = total_p_loss/iter_
            total_gp_loss = total_gp_loss/iter_

            with open(self.opt.log_file, mode = 'a')as f :
                f.write("{:d},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f},{:.8f}\n".format(
                    epoch, 
                    total_g_loss,
                    total_p_loss,
                    total_d_loss,
                    total_gp_loss,
                    pp,
                    ps)
                )

    """ #no trunc
    def test(self):
        del self.WGANVGG
        # load
        self.WGANVGG_G = WGAN_VGG_generator().to(self.device)
        self.load_model()

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg= 0, 0
        pred_psnr_avg, pred_ssim_avg = 0, 0

        with torch.no_grad():
            num_total_img = len(self.test_list)
            for img_idx, img_path in enumerate(self.test_list):
                img_name = os.path.basename(img_path)
                img_path = os.path.abspath(img_path)
                print("[{}/{}] processing {}".format(img_idx, num_total_img, os.path.abspath(img_path)))
                
                gt_img_path = self.test_gt_list[img_idx]
                gt_img = imread(gt_img_path)
                input_img = imread(img_path)
                img_patch_dataset = ImageDataset(self.opt, input_img)
                img_patch_dataloader = DataLoader(dataset=img_patch_dataset,
                                            batch_size=self.opt.batch_size,
                                            shuffle=False)

                img_shape = img_patch_dataset.get_img_shape()
                pad_img_shape = img_patch_dataset.get_padded_img_shape()
                
                out_list =[]

                for i, x in enumerate(img_patch_dataloader):

                    x = x.float().to(self.device)

                    pred = self.WGANVGG_G(x)
                    pred = pred.to('cpu').detach().numpy()
                    out_list.append(pred)

                out = np.concatenate(out_list, axis = 0)
                out = out.squeeze()

                img_name = 'out-'+img_name
                base_name = os.path.basename(self.opt.checkpoint_dir)
                test_result_dir = os.path.join(self.opt.test_result_dir, base_name)
                if not os.path.exists(test_result_dir):
                    os.makedirs(test_result_dir)
                dst_img_path = os.path.join(test_result_dir, img_name)

                out_img = mp.recon_patches(out, pad_img_shape[1], pad_img_shape[0], self.opt.patch_size, self.opt.patch_offset)
                out_img = mp.unpad_img(out_img, self.opt.patch_offset, img_shape)

                input_img = torch.from_numpy(input_img).float().to(self.device)
                gt_img = torch.from_numpy(gt_img).float().to(self.device)
                original_result, pred_result = compute_measure(input_img, gt_img, out_img, 1)

                op, oos, _ = original_result
                pp, ps, _ = pred_result

                ori_psnr_avg += op
                ori_ssim_avg += oos
                pred_psnr_avg += pp
                pred_ssim_avg += ps

                imsave(dst_img_path, out_img)

            aop = ori_psnr_avg/(img_idx+1)
            aos = ori_ssim_avg/(img_idx+1)
            app = pred_psnr_avg/(img_idx+1)
            aps = pred_ssim_avg/(img_idx+1)
            print("((ORIGIN)) PSNR : {:.5f}, SSIM : {:.5f}, ((PREP)) PSNR : {:.5f}, SSIM : {:.5f}".format(aop,aos,app,aps))
    """
    #trunc
    def test(self):
        del self.WGANVGG
        # load
        self.WGANVGG_G = WGAN_VGG_generator().to(self.device)
        self.load_model()

        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg= 0, 0
        pred_psnr_avg, pred_ssim_avg = 0, 0

        with torch.no_grad():
            num_total_img = len(self.test_list)
            for img_idx, img_path in enumerate(self.test_list):
                img_name = os.path.basename(img_path)
                img_path = os.path.abspath(img_path)
                print("[{}/{}] processing {}".format(img_idx, num_total_img, os.path.abspath(img_path)))
                
                gt_img_path = self.test_gt_list[img_idx]
                gt_img = imread(gt_img_path)
                input_img = imread(img_path)
                img_patch_dataset = ImageDataset(self.opt, input_img)
                img_patch_dataloader = DataLoader(dataset=img_patch_dataset,
                                            batch_size=self.opt.batch_size,
                                            shuffle=False)

                img_shape = img_patch_dataset.get_img_shape()
                pad_img_shape = img_patch_dataset.get_padded_img_shape()
                
                out_list =[]

                for i, x in enumerate(img_patch_dataloader):

                    x = x.float().to(self.device)

                    pred = self.WGANVGG_G(x)
                    pred = pred.to('cpu').detach().numpy()
                    out_list.append(pred)

                out = np.concatenate(out_list, axis = 0)
                out = out.squeeze()

                img_name = 'out-'+img_name
                base_name = os.path.basename(self.opt.checkpoint_dir)
                test_result_dir = os.path.join(self.opt.test_result_dir, base_name)
                if not os.path.exists(test_result_dir):
                    os.makedirs(test_result_dir)
                dst_img_path = os.path.join(test_result_dir, img_name)

                out_img = mp.recon_patches(out, pad_img_shape[1], pad_img_shape[0], self.opt.patch_size, self.opt.patch_offset)
                out_img = mp.unpad_img(out_img, self.opt.patch_offset, img_shape)

                input_img = torch.Tensor(input_img)
                out_img = torch.Tensor(out_img)
                gt_img = torch.Tensor(gt_img)
                input_img = self.trunc(self.denormalize_(input_img).cpu().detach())
                out_img = self.trunc(self.denormalize_(out_img).cpu().detach())
                gt_img = self.trunc(self.denormalize_(gt_img).cpu().detach())

                # x = self.trunc(self.denormalize_(x))
                # out_img = self.trunc(self.denormalize_(out_img))
                # gt_img = self.trunc(self.denormalize_(gt_img))

                data_range = self.trunc_max-self.trunc_min

                original_result, pred_result = compute_measure(input_img, gt_img, out_img, data_range)

                op, oos, _ = original_result
                pp, ps, _ = pred_result

                ori_psnr_avg += op
                ori_ssim_avg += oos
                pred_psnr_avg += pp
                pred_ssim_avg += ps

                out_img = self.normalize_(out_img)
                out_img = out_img.cpu().numpy()
                imsave(dst_img_path, out_img)

            aop = ori_psnr_avg/(img_idx+1)
            aos = ori_ssim_avg/(img_idx+1)
            app = pred_psnr_avg/(img_idx+1)
            aps = pred_ssim_avg/(img_idx+1)
            print("((ORIGIN)) PSNR : {:.5f}, SSIM : {:.5f}, ((PREP)) PSNR : {:.5f}, SSIM : {:.5f}".format(aop,aos,app,aps))

    