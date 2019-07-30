
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image
import os

# This is base on https://www.kaggle.com/witwitchayakarn/u-net-with-pytorch


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    ends = starts + lengths
    im = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        im[lo:hi] = 1
    return im.reshape(shape).T


def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs[::2] -= 1
    return ' '.join(str(x) for x in runs)


class ImgMaskUnetDataset(Dataset):
    def __init__(self,
                 img_fnames,
                 img_transform,
                 injection=None,
                 mask_transform=None
                 ):
        self.img_fnames = img_fnames
        self.img_transform = img_transform
        self.injection = injection
        self.mask_transform = mask_transform

    def __getitem__(self, i):
        img = Image.open(self.injection.get_path(self.img_fnames[i]))
        #print(self.mask_transform)
        if self.mask_transform is None:
            raise ValueError(
                'If mask_dpath is not None, mask_size and mask_transform must not be None.')
        # このへんでinjection patternを使う
        masks = self.injection.get_mask(self.img_fnames[i], np.array(img))
        #masks = self.mask_transform(mask)
        return torch.from_numpy(np.array(img).reshape(3, 256, 1600).astype(np.float)).float(), torch.from_numpy(np.array(masks, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)


def get_each_unet_dataloader(img_size,
                             batch_size,
                             num_workers,
                             injection_train_val,
                             train_fnames,
                             val_fnames,
                             test_fnames):
    train_transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.RandomRotation(360),
                                          transforms.ToTensor(), ])
    # transforms.Normalize(channel_means, channel_stds)])
    val_transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.ToTensor(), ])
    # transforms.Normalize(channel_means, channel_stds)])
    mask_transform = transforms.Compose([transforms.Resize(img_size),
                                         transforms.RandomRotation(360)])

    train_dataloader = DataLoader(ImgMaskUnetDataset(
        train_fnames,
        train_transform,
        injection_train_val,
        mask_transform),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers)
    val_dataloader = DataLoader(ImgMaskUnetDataset(
        val_fnames,
        val_transform,
        injection_train_val,
        mask_transform),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers)
    test_dataloader = DataLoader(ImgMaskUnetDataset(
        test_fnames,
        val_transform),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers)
    return (train_dataloader, val_dataloader, test_dataloader)
