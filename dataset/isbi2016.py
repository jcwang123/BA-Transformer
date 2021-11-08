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

        indexes = [
            l[:-4] for l in os.listdir(
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Image/'
            )
        ]
        test_indexes = [
            l[:-4] for l in os.listdir(
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Test/Image/'
            )
        ]
        temp = int(0.2 * len(indexes))
        # 8:2
        random.shuffle(indexes)

        train_indexes = indexes[:-temp]
        valid_indexes = indexes[-temp:]
        print('train: {} valid: {}'.format(len(train_indexes),
                                           len(valid_indexes)))

        if split == 'train':
            self.image_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Image/{}.npy'
                .format(_id) for _id in train_indexes
            ]
            self.label_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Label/{}.npy'
                .format(_id) for _id in train_indexes
            ]
            self.point_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Point/{}.npy'
                .format(_id) for _id in train_indexes
            ]
        elif split == 'valid':
            self.image_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Image/{}.npy'
                .format(_id) for _id in valid_indexes
            ]
            self.label_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Label/{}.npy'
                .format(_id) for _id in valid_indexes
            ]
            self.point_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Train/Point/{}.npy'
                .format(_id) for _id in valid_indexes
            ]
        elif split == 'test':
            self.image_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Test/Image/{}.npy'
                .format(_id) for _id in test_indexes
            ]
            self.label_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Test/Label/{}.npy'
                .format(_id) for _id in test_indexes
            ]
            self.point_paths = [
                '/data2/cf_data/skinlesion_segment/ISBI2016_rawdata/ISBI_2016/Test/Label/{}.npy'
                .format(_id) for _id in test_indexes
            ]
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