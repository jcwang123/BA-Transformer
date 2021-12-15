import os
import glob
import random
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F

import albumentations as A


def norm01(x):
    return np.clip(x, 0, 255) / 255


def filter_image(p):
    label_data = np.load(p.replace('image', 'label'))
    return np.max(label_data) == 1


class myDataset(data.Dataset):
    def __init__(self, split, aug=False):
        super(myDataset, self).__init__()

        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        root_dir = '/raid/wjc/data/skin_lesion/isic2016/'
        if split == 'train':
            self.image_paths = glob.glob(root_dir + '/Train/Image/*.npy')
            self.label_paths = glob.glob(root_dir + '/Train/Label/*.npy')
            self.point_paths = glob.glob(root_dir + '/Train/Point/*.npy')
        elif split == 'valid':
            self.image_paths = glob.glob(root_dir + '/Validation/Image/*.npy')
            self.label_paths = glob.glob(root_dir + '/Validation/Label/*.npy')
            self.point_paths = glob.glob(root_dir + '/Validation/Point/*.npy')
        elif split == 'test':
            self.image_paths = glob.glob(root_dir + '/Test/Image/*.npy')
            self.label_paths = glob.glob(root_dir + '/Test/Label/*.npy')
            self.point_paths = glob.glob(root_dir + '/Test/Point/*.npy')
        self.image_paths.sort()
        self.label_paths.sort()
        self.point_paths.sort()

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug

        self.transf = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate()
        ])

    def __getitem__(self, index):

        image_data = np.load(self.image_paths[index])
        label_data = np.load(self.label_paths[index]) > 0.5
        point_data = np.load(self.point_paths[index])

        if self.aug:
            mask = np.concatenate([
                label_data[..., np.newaxis].astype('uint8'),
                point_data[..., np.newaxis]
            ],
                                  axis=-1)
            #             print(mask.shape)
            tsf = self.transf(image=image_data.astype('uint8'), mask=mask)
            image_data, mask_aug = tsf['image'], tsf['mask']
            label_data = mask_aug[:, :, 0]
            point_data = mask_aug[:, :, 1]

        image_data = norm01(image_data)
        label_data = np.expand_dims(label_data, 0)
        point_data = np.expand_dims(point_data, 0)
        image_data = torch.from_numpy(image_data).float()
        label_data = torch.from_numpy(label_data).float()
        point_data = torch.from_numpy(point_data).float()

        image_data = image_data.permute(2, 0, 1)
        return {
            'image_path': self.image_paths[index],
            'label_path': self.label_paths[index],
            'point_path': self.point_paths[index],
            'image': image_data,
            'label': label_data,
            'point': point_data
        }

    def __len__(self):
        return self.num_samples