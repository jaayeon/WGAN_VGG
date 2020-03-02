import os, glob
import time
from shutil import copyfile, move

import numpy as np
from skimage.external.tifffile import imsave, imread


def pad_img(img, patch_size, patch_offset):
    stride = patch_size - 2 * patch_offset
    # print("original image size:", img.shape)
    # print("stride:", stride)
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    # rw = (patch_offset + w) % patch_size
    # rh = (patch_offset + h) % patch_size
    # w_pad_size = patch_size - rw
    # h_pad_size = patch_size - rh
    rw = (patch_offset + w) % stride
    rh = (patch_offset + h) % stride
    w_stride_pad_size = stride - rw
    h_stride_pad_size = stride - rh

    stride_pad_w = patch_offset + w + w_stride_pad_size
    stride_pad_h = patch_offset + h + h_stride_pad_size

    # rw = stride_pad_w % (2 * patch_size)
    # rh = stride_pad_h % (2 * patch_size)
    # w_pad_size = (2 * patch_size) - rw
    # h_pad_size = (2 * patch_size) - rh
    
    w_pad_size = w_stride_pad_size + patch_size
    h_pad_size = h_stride_pad_size + patch_size

    # print("offset({}, {}), Padding({}, {})".format(patch_offset, patch_offset, h_pad_size, w_pad_size))
    if img.ndim == 2:
        npad = ((patch_offset, h_pad_size), (patch_offset, w_pad_size))
    else:
        npad = ((patch_offset, h_pad_size), (patch_offset, w_pad_size), (0, 0))

    img = np.pad(img, npad, 'reflect')
    # print("padded image size:", img.shape)
    return img
    

def unpad_img(img, patch_offset, img_shape):
    if img.ndim == 2:
        h, w = img_shape
        ret = img[patch_offset:patch_offset+h, patch_offset:patch_offset+w]
    else:
        h, w, _ = img_shape
        ret = img[patch_offset:patch_offset+h, patch_offset:patch_offset+w, :]
    # print("img.shape:", img.shape)
    # print("img_shape:", img_shape)
    # print("patch_offset:", patch_offset)
    
    return ret

def make_patches(img, patch_size, patch_offset):
    stride = patch_size - 2 * patch_offset

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    img_dims = img.shape
    img_h, img_w, _ = img_dims

    # print("img_dims: " + str(img_dims))

    mod_h = img_h - np.mod(img_h - patch_size, stride)
    mod_w = img_w - np.mod(img_w - patch_size, stride)
    # mod_h = img_h
    # mod_w = img_w
    
    num_patches = (mod_h // stride) * (mod_w // stride)

    if img.shape[2] == 1:
        patch_arr = np.zeros((num_patches, patch_size, patch_size), dtype=img.dtype)
    else:
        patch_arr = np.zeros((num_patches, patch_size, patch_size, 3), dtype=img.dtype)

    ps = patch_size

    patch_idx = 0
    for y in range(0, mod_h - stride + 1, stride):
        for x in range(0, mod_w - stride + 1, stride):
            # print("({}:{}, {}:{})".format(y, y+patch_size, x, x+patch_size))
            patch = img[y:y+ps, x:x+ps, :]

            patch = patch.squeeze()

            patch_arr[patch_idx] = patch
            patch_idx += 1

    # print("patch_idx: " + str(patch_idx))
    # print("patch_dtype: " + str(patch_arr.dtype))
    
    return patch_arr

def recon_patches(patch_arr, width, height, patch_size, patch_offset):
    stride = patch_size - 2 * patch_offset

    if patch_arr[0].ndim == 2:
        img = np.zeros((height, width, 1), dtype=patch_arr[0].dtype)
    else:
        img = np.zeros((height, width, 3), dtype=patch_arr[0].dtype)

    # img_dims = img.shape
    # img_h, img_w = img_dims

    # print("img_dims: " + str(img_dims))

    mod_h = height - np.mod(height - 2 * patch_offset, stride)
    mod_w = width - np.mod(width - 2 * patch_offset, stride)

    ps = patch_size
    po = patch_offset

    patch_idx = 0
    for y in range(0, mod_h - (patch_size - patch_offset) + 1, stride):
        for x in range(0, mod_w - (patch_size - patch_offset) + 1, stride):
            if patch_arr[patch_idx].ndim == 2:
                patch = np.expand_dims(patch_arr[patch_idx], axis=2)
            else:
                patch = patch_arr[patch_idx]
                
            img[y+po:y+ps-po, x+po:x+ps-po, :] = patch[po:-po, po:-po, :]
            patch_idx += 1

    img = img.squeeze()
    return img
