
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
                 img_dpath,
                 img_fnames,
                 img_transform,
                 mask_encodings=None,
                 mask_size=None,
                 mask_transform=None):
        self.img_dpath = img_dpath
        self.img_fnames = img_fnames
        self.img_transform = img_transform
        self.mask_encodings = mask_encodings
        self.mask_size = mask_size
        self.mask_transform = mask_transform

    def __getitem__(self, i):
        # https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        seed = np.random.randint(2147483647)
        fname = self.img_fnames[i]
        fpath = os.path.join(self.img_dpath, fname)
        img = Image.open(fpath)
        if self.img_transform is not None:
            random.seed(seed)
            img = self.img_transform(img)
        if self.mask_encodings is None:
            return img, fname
        if self.mask_size is None or self.mask_transform is None:
            raise ValueError(
                'If mask_dpath is not None, mask_size and mask_transform must not be None.')
        mask = np.zeros(self.mask_size, dtype=np.uint8)
        # NaN doesn't equal to itself
        if self.mask_encodings[fname][0] == self.mask_encodings[fname][0]:
            for encoding in self.mask_encodings[fname]:
                mask += rle_decode(encoding, self.mask_size)
        mask = np.clip(mask, 0, 1)
        mask = Image.fromarray(mask)
        random.seed(seed)
        mask = self.mask_transform(mask)
        return img, torch.from_numpy(np.array(mask, dtype=np.int64))

    def __len__(self):
        return len(self.img_fnames)


def get_each_unet_dataloader(channel_means, channel_stds, img_size, batch_size, num_workers,
                             original_img_size,
                             train_dpath, train_fnames, train_encodings,
                             val_dpath, val_fnames, val_encodings,
                             test_dpath, test_fnames):
    train_transform = transforms.Compose([transforms.Resize(img_size),
                                          transforms.RandomRotation(360),
                                          transforms.ToTensor(),
                                          transforms.Normalize(channel_means, channel_stds)])
    val_transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(channel_means, channel_stds)])
    mask_transform = transforms.Compose([transforms.Resize(img_size),
                                         transforms.RandomRotation(360)])

    train_dataloader = DataLoader(ImgMaskUnetDataset(train_dpath,
                                                     train_fnames,
                                                     train_transform,
                                                     train_encodings,
                                                     original_img_size,
                                                     mask_transform),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  num_workers=num_workers)
    val_dataloader = DataLoader(ImgMaskUnetDataset(val_dpath,
                                                   val_fnames,
                                                   val_transform,
                                                   val_encodings,
                                                   original_img_size,
                                                   mask_transform),
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=num_workers)
    test_dataloader = DataLoader(ImgMaskUnetDataset(test_dpath,
                                                    test_fnames,
                                                    val_transform),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=torch.cuda.is_available(),
                                 num_workers=num_workers)
    return (train_dataloader, val_dataloader, test_dataloader)
