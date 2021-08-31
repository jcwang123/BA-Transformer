import cv2
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_dir_path = '/raid/wl/2018_raw_data/ISIC2018_Task1-2_Training_Input'
    mask_dir_path = '/raid/wl/2018_raw_data/ISIC2018_Task1_Training_GroundTruth'
    image_path_list = os.listdir(image_dir_path)
    mask_path_list = os.listdir(mask_dir_path)
    image_path_list.sort()
    mask_path_list.sort()

    # ISBI Dataset
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'png' or image_path[-3:] == 'jpg':
            #             i = path[5:12]
            assert os.path.basename(image_path)[:-4].split(
                '_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
            _id = os.path.basename(image_path)[:-4].split('_')[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)

            dim = (512, 512)
            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_NEAREST)

            save_dir_path = '/raid/wl/ISBI_2018/Train/Image'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

            save_dir_path = '/raid/wl/ISBI_2018/Train/Label'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)

#             break

# PH2 Dataset
    for path in path_list:
        label_path = os.path.join('/home/wl/ISBI/PH2', path, path + '_lesion',
                                  path + '_lesion.bmp')
        label = plt.imread(label_path)
        label = label[:, :, 0]

        dim = (512, 512)
        label_new = cv2.resize(label, dim, interpolation=cv2.INTER_NEAREST) > 0

        label_save_path = os.path.join('/home/wl/ISBI/Test/Label',
                                       path + '_label.npy')

        np.save(label_save_path, label_new)
