import cv2
import os
import random
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    image_dir_path = '/data2/cf_data/skinlesion_segment/ISIC2017_rawdata/ISIC-2017_Test_v2_Data'
    mask_dir_path = '/data2/cf_data/skinlesion_segment/ISIC2017_rawdata/ISIC-2017_Test_v2_Part1_GroundTruth'
    mask_path_list = os.listdir(mask_dir_path)

    image_path_list = []
    for FileName in os.listdir(image_dir_path):
        if FileName[-3:] == 'jpg':
            image_path_list.append(FileName)

    image_path_list.sort()
    mask_path_list.sort()

    # ISBI Dataset
    for image_path, mask_path in zip(image_path_list, mask_path_list):
        if image_path[-3:] == 'jpg':  #image_path[-3:] == 'png'
            print(image_path)
            assert os.path.basename(image_path)[:-4].split(
                '_')[1] == os.path.basename(mask_path)[:-4].split('_')[1]
            _id = os.path.basename(image_path)[:-4].split('_')[1]
            image_path = os.path.join(image_dir_path, image_path)
            mask_path = os.path.join(mask_dir_path, mask_path)
            image = plt.imread(image_path)
            mask = plt.imread(mask_path)

            dim = (512, 512)
            image_new = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            mask_new = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)

            save_dir_path = '/data2/cf_data/skinlesion_segment/ISIC2017_rawdata/ISBI_2017/Test/Image'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), image_new)

            save_dir_path = '/data2/cf_data/skinlesion_segment/ISIC2017_rawdata/ISBI_2017/Test/Label'
            os.makedirs(save_dir_path, exist_ok=True)
            np.save(os.path.join(save_dir_path, _id + '.npy'), mask_new)
