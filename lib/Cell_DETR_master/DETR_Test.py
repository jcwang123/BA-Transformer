import os,argparse, math
import sys
import torch
import torch.nn
import tqdm
import logging
import numpy as np
from glob import glob
from HNgtv_dataset import norm01,myDataset,crop_array
from medpy.metric.binary import hd, hd95, dc, jc, assd
sys.path.append('Ours/Cell_DETR_master/')

parser = argparse.ArgumentParser()
parser.add_argument('--crop_size', type=int,default=128)
# parser.add_argument('--with_BPB', type=int, default=0,choices = [0,1])
parse_config = parser.parse_args()
print(parse_config)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torch.load('./logs/test5_aaai_sdm_loss/model/best.pkl')
dir_path = "./logs/test_transformer_loss_0_ver_1/"
model = torch.load(dir_path + 'model/best.pkl')
# model = torch.load(dir_path + 'model/latest.pkl')
txt_path = os.path.join(dir_path + 'parameter.txt')

def test():
    dice_value = 0
    hd_value = 0
    hd95_value = 0
    jc_value = 0
    assd_value = 0
    numm = 0
   
    logging.basicConfig(filename = txt_path, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    for num in range(41,51):
        prediction = []
        labels = []
        img_dir = os.path.join( '/home/wl/gtv_data/Patient' + str(num).zfill(3)+ '/')
        n = len(glob(img_dir + 'image_*.npy'))
        for i in range(n):
            img = np.load(img_dir + 'image_{:03d}.npy'.format(i))
            label = np.load(img_dir + 'label_{:03d}.npy'.format(i))
            point = np.load(img_dir + 'point_{:03d}.npy'.format(i))
            point = np.expand_dims(point,0)
            label_sum = np.sum(label)
            img = norm01(img)
            img = crop_array(img, parse_config.crop_size)
#             img = np.repeat(img,3,0)
            label = crop_array(label, parse_config.crop_size)
            labels.append(label)
            img = torch.from_numpy(img).unsqueeze(0).float().cuda()
            
            with torch.no_grad():
#                 if parse_config.with_BPB == 0:
#                     output = model(img)
#                 if parse_config.with_BPB == 1:
#                     output,maps = model(img)
#                 output = torch.sigmoid(output)[0]
                
                output = model(img)
                output = torch.max(output, dim=0, keepdim=True).values
                output = output.cpu().numpy()>0.5
                
#             writer.add_image('val_label', label, val_num)
#             writer.add_image('val_point', point, val_num)
#             writer.add_image('val_output', output, val_num)
            
#             if parse_config.with_BPB == 1:
#                 for j, m in enumerate(maps):
#                     if m is not None:
#                         writer.add_image('val_m{}_img'.format(j+1), maps[j][0,...], val_num)
#             val_num += 1
            prediction.append(output)
        
        prediction = np.array(prediction)
#         prediction = prediction.squeeze(1)
        labels = np.array(labels)
        
        assert(prediction.shape==labels.shape)
        # calculate metric
        dice_ave = dc(prediction, labels)
        hd_ave = hd(prediction, labels)
        hd95_ave = hd95(prediction, labels)
        jc_ave = jc(prediction, labels)
        assd_ave = assd(prediction, labels)
        
        logging.info('patient %d : dice_value  : %f' % (num, dice_ave))
        logging.info('patient %d : hd_value   : %f' % (num, hd_ave))
        logging.info('patient %d : hd95_value  : %f' % (num, hd95_ave))
        logging.info('patient %d : jc_value   : %f' % (num, jc_ave))
        logging.info('patient %d : assd_value  : %f' % (num, assd_ave))
        
#         print("Dice value for patient{} = ".format(num),dice_ave)
#         print("HD value for patient{} = ".format(num),hd_ave)
        dice_value += dice_ave
        hd_value += hd_ave
        hd95_value += hd95_ave
        jc_value += jc_ave
        assd_value += assd_ave
        numm += 1
        
    dice_average = dice_value / numm
    hd_average = hd_value / numm
    hd95_average = hd95_value / numm
    jc_average = jc_value / numm
    assd_average = assd_value / numm
    
    logging.info('Dice value of test dataset  : %f' % (dice_average))
    logging.info('HD value of test dataset   : %f' % (hd_average))
    logging.info('HD95 value of test dataset  : %f' % (hd95_average))
    logging.info('JC value of test dataset   : %f' % (jc_average))
    logging.info('ASSD value of test dataset  : %f' % (assd_average))
#     print("Average dice value of evaluation dataset = ",dice_average)
    return dice_average

if __name__ == '__main__':
    test()