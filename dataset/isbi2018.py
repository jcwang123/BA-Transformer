import os
import glob
import json
import torch
import random
import torch.nn as nn
import numpy as np
import torch.utils.data
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F

import albumentations as A


def norm01(x):
    return np.clip(x, 0, 255) / 255


seperable_indexes = json.load(open('/raid/wl/ISBI_2018/data_split.json', 'r'))


# cross validation
class myDataset(data.Dataset):
    def __init__(self, fold, split, aug=False):
        super(myDataset, self).__init__()
        self.split = split

        # load images, label, point
        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        indexes = [
            l[:-4] for l in os.listdir('/raid/wl/ISBI_2018/Train/Image/')
        ]
        valid_indexes = seperable_indexes[fold]
        train_indexes = list(filter(lambda x: x not in valid_indexes, indexes))
        print('Fold {}: train: {} valid: {}'.format(fold, len(train_indexes),
                                                    len(valid_indexes)))

        if split == 'train':
            self.image_paths = [
                '/raid/wl/ISBI_2018/Train/Image/{}.npy'.format(_id)
                for _id in train_indexes
            ]
            self.label_paths = [
                '/raid/wl/ISBI_2018/Train/Label/{}.npy'.format(_id)
                for _id in train_indexes
            ]
            self.point_paths = [
                '/raid/wl/ISBI_2018/Train/Point/{}.npy'.format(_id)
                for _id in train_indexes
            ]
        elif split == 'valid':
            self.image_paths = [
                '/raid/wl/ISBI_2018/Train/Image/{}.npy'.format(_id)
                for _id in valid_indexes
            ]
            self.label_paths = [
                '/raid/wl/ISBI_2018/Train/Label/{}.npy'.format(_id)
                for _id in valid_indexes
            ]
            self.point_paths = [
                '/raid/wl/ISBI_2018/Train/Point/{}.npy'.format(_id)
                for _id in valid_indexes
            ]

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug

        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            #             A.RandomBrightnessContrast(p=p),
        ])

    def __getitem__(self, index):

        image_data = np.load(self.image_paths[index])
        label_data = np.load(self.label_paths[index]) > 0.5
        point_data = np.load(self.point_paths[index]) > 0.5

        if self.aug and self.split == 'train':
            mask = np.concatenate([
                label_data[..., np.newaxis].astype('uint8'),
                point_data[..., np.newaxis]
            ],
                                  axis=-1)
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


def random_seperate_dataset():
    indexes = [l[:-4] for l in os.listdir('/raid/wl/ISBI_2018/Train/Image/')]
    random.shuffle(indexes)
    names = [
        indexes[:500], indexes[500:1000], indexes[1000:1500],
        indexes[1500:2000], indexes[2000]
    ]
    with open('/raid/wl/ISBI_2018/data_split.json', 'w') as f:
        json.dump(names, f)
    return


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = myDataset(fold=0, split='train', aug=False)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=8,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    for d in train_loader:
        pass
