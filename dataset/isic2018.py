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
from sklearn.model_selection import KFold


def norm01(x):
    return np.clip(x, 0, 255) / 255


seperable_indexes = json.load(open('dataset/data_split.json', 'r'))


# cross validation
class myDataset(data.Dataset):
    def __init__(self, fold, split, aug=False):
        super(myDataset, self).__init__()
        self.split = split
        root_data_dir = '/raid/wjc/data/skin_lesion/isic2018/'

        # load images, label, point
        self.image_paths = []
        self.label_paths = []
        self.point_paths = []
        self.dist_paths = []

        indexes = [l[:-4] for l in os.listdir(root_data_dir + 'Image/')]
        valid_indexes = seperable_indexes[fold]

        train_indexes = list(filter(lambda x: x not in valid_indexes, indexes))
        print('Fold {}: train: {} valid: {}'.format(fold, len(train_indexes),
                                                    len(valid_indexes)))

        indexes = train_indexes if split == 'train' else valid_indexes

        self.image_paths = [
            root_data_dir + '/Image/{}.npy'.format(_id) for _id in indexes
        ]
        self.label_paths = [
            root_data_dir + '/Label/{}.npy'.format(_id) for _id in indexes
        ]
        self.point_paths = [
            root_data_dir + '/Point/{}.npy'.format(_id) for _id in indexes
        ]
        # self.point_All_paths = [
        #     '/data2/cf_data/skinlesion_segment/ISIC2018_rawdata/ISBI_2018/Train/Point_All/{}.npy'
        #     .format(_id) for _id in indexes
        # ]

        print('Loaded {} frames'.format(len(self.image_paths)))
        self.num_samples = len(self.image_paths)
        self.aug = aug

        p = 0.5
        self.transf = A.Compose([
            A.GaussNoise(p=p),
            A.HorizontalFlip(p=p),
            A.VerticalFlip(p=p),
            A.ShiftScaleRotate(p=p),
            # A.RandomBrightnessContrast(p=p),
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


def dataset_kfold(dataset_dir, save_path, k=5):
    indexes = [l[:-4] for l in os.listdir(dataset_dir)]

    kf = KFold(k, shuffle=True)  #k折交叉验证
    val_index = dict()
    for i in range(k):
        val_index[str(i)] = []

    for i, (tr, val) in enumerate(kf.split(indexes)):
        for item in val:
            val_index[str(i)].append(indexes[item])
        print('fold:{},train_len:{},val_len:{}'.format(i, len(tr), len(val)))

    with open(save_path, 'w') as f:
        json.dump(val_index, f)


def random_seperate_dataset():
    #    indexes = [l[:-4] for l in os.listdir('/raid/wl/ISBI_2018/Train/Image/')]
    #    random.shuffle(indexes)
    #    names = {'0':indexes[:500], '1':indexes[500:1000], '2':indexes[1000:1500], '3':indexes[1500:2000],'4':indexes[2000:]}
    #    names = [indexes[:500], indexes[500:1000], indexes[1000:1500], indexes[1500:2000], indexes[2000:]]
    #    with open('/raid/wl/ISBI_2018/data_split.json','w') as f:
    #        json.dump(names, f)
    return


if __name__ == '__main__':
    from tqdm import tqdm
    dataset = myDataset(fold='0', split='train', aug=True)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=8,
                                               shuffle=False,
                                               num_workers=2,
                                               pin_memory=True,
                                               drop_last=True)
    for d in train_loader:
        pass
