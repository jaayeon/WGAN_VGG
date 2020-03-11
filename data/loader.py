import os
from glob import glob
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.patchdata import PatchData
from data.make_patches import make_patches, pad_img, unpad_img
import torch


class ct_dataset(PatchData):

    def __init__(self, args, name='mayo', mode='train', benchmark=False):
        self.thickness = args.thickness
        super(ct_dataset, self).__init__(
            args, name=name, mode=mode, benchmark=benchmark
        )
        # Mayo specific
        
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '**', '*' + self.ext[0]))
        )
        names_lr = sorted(
            glob.glob(os.path.join(self.dir_lr, '**', '*' + self.ext[1]))
        )

        return names_hr, names_lr


    def _set_filesystem(self, data_dir):
        super(ct_dataset, self)._set_filesystem(data_dir)

        full_dose = 'full_{}mm'.format(self.thickness)
        quarter_dose = 'quarter_{}mm'.format(self.thickness)
        self.dir_hr = os.path.join(self.apath, full_dose)
        self.dir_lr = os.path.join(self.apath, quarter_dose)
        self.ext = ('.tiff', '.tiff')


class ImageDataset(Dataset):
    def __init__(self, opt, img):
        super(ImageDataset, self).__init__()

        if opt.n_channels == 3:
            img = img / 255.0

        self.img_shape = img.shape

        self.opt = opt

        padded_img = pad_img(img, opt.patch_size, opt.patch_offset)
        self.pad_img_shape =padded_img.shape
        self.img_patches = make_patches(padded_img, opt.patch_size, opt.patch_offset)

        patches_dims = self.img_patches.shape

        self.img_patches = self.img_patches.reshape(patches_dims[0], 1, patches_dims[1], patches_dims[2])
        # print(self.img_patches.shape)

    def __getitem__(self, idx):
        patch = self.img_patches[idx]
        patch = torch.from_numpy(patch).type(torch.FloatTensor)
        return patch

    def __len__(self):
        return len(self.img_patches)

    def get_img_shape(self):
        return self.img_shape

    def get_padded_img_shape(self):
        return self.pad_img_shape


    # def __init__(self, mode, load_mode, saved_path, test_patient, patch_n=None, patch_size=None, transform=None):
    #     assert mode in ['train', 'test'], "mode is 'train' or 'test'"
    #     assert load_mode in [0,1], "load_mode is 0 or 1"

    #     input_path = sorted(glob(os.path.join(saved_path, '*_input.npy')))
    #     target_path = sorted(glob(os.path.join(saved_path, '*_target.npy')))
    #     self.load_mode = load_mode
    #     self.patch_n = patch_n
    #     self.patch_size = patch_size
    #     self.transform = transform

    #     if mode == 'train':
    #         input_ = [f for f in input_path if test_patient not in f]
    #         target_ = [f for f in target_path if test_patient not in f]
    #         if load_mode == 0: # batch data load
    #             self.input_ = input_
    #             self.target_ = target_
    #         else: # all data load
    #             self.input_ = [np.load(f) for f in input_]
    #             self.target_ = [np.load(f) for f in target_]
    #     else: # mode =='test'
    #         input_ = [f for f in input_path if test_patient in f]
    #         target_ = [f for f in target_path if test_patient in f]
    #         if load_mode == 0:
    #             self.input_ = input_
    #             self.target_ = target_
    #         else:
    #             self.input_ = [np.load(f) for f in input_]
    #             self.target_ = [np.load(f) for f in target_]

    # def __len__(self):
    #     return len(self.target_)

    # def __getitem__(self, idx):
    #     input_img, target_img = self.input_[idx], self.target_[idx]
    #     if self.load_mode == 0:
    #         input_img, target_img = np.load(input_img), np.load(target_img)

    #     if self.transform:
    #         input_img = self.transform(input_img)
    #         target_img = self.transform(target_img)

    #     if self.patch_size:
    #         input_patches, target_patches = get_patch(input_img,
    #                                                   target_img,
    #                                                   self.patch_n,
    #                                                   self.patch_size)
    #         return (input_patches, target_patches)
    #     else:
    #         return (input_img, target_img)


# def get_patch(full_input_img, full_target_img, patch_n, patch_size, drop_background=0.1):
#     assert full_input_img.shape == full_target_img.shape
#     patch_input_imgs = []
#     patch_target_imgs = []
#     h, w = full_input_img.shape
#     new_h, new_w = patch_size, patch_size
#     n = 0
#     while n <= patch_n:
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
#         patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
#         patch_target_img = full_target_img[top:top + new_h, left:left + new_w]

#         if (np.mean(patch_input_img) < drop_background) or \
#             (np.mean(patch_target_img) < drop_background):
#             continue
#         else:
#             n += 1
#             patch_input_imgs.append(patch_input_img)
#             patch_target_imgs.append(patch_target_img)
#     '''
#     for _ in range(patch_n):
#         top = np.random.randint(0, h-new_h)
#         left = np.random.randint(0, w-new_w)
#         patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
#         patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
#         patch_input_imgs.append(patch_input_img)
#         patch_target_imgs.append(patch_target_img)
#     '''
#     return np.array(patch_input_imgs), np.array(patch_target_imgs)


# def get_loader(mode='train', load_mode=0,
#                saved_path=None, test_patient='L506',
#                patch_n=None, patch_size=None,
#                transform=None, batch_size=32, num_workers=6):
#     dataset_ = ct_dataset(mode, load_mode, saved_path, test_patient, patch_n, patch_size, transform)
#     data_loader = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return data_loader

def get_train_loader(opt, mode = 'train'):
    dataset = ct_dataset(opt, name = 'mayo', mode = mode)
    data_loader = DataLoader(dataset = dataset, batch_size = opt.batch_size, shuffle=True, num_workers = opt.num_workers)
    return data_loader

def get_test_list(opt, mode = 'test'):
    opt.img_dir = 'D:/data/denoising/test/mayo/quarter_3mm/L506'
    opt.gt_img_dir = 'D:/data/denoising/test/mayo/full_3mm/L506'
    # opt.img_dir = 'D:/data/denoising/test/mayo/quarter_1mm/L506'
    # opt.gt_img_dir = 'D:/data/denoising/test/mayo/full_1mm/L506'
    img_list = [os.path.join(opt.img_dir, x) for x in os.listdir(opt.img_dir)]
    gt_img_list = [os.path.join(opt.gt_img_dir, x) for x in os.listdir(opt.gt_img_dir)]
    return img_list, gt_img_list
