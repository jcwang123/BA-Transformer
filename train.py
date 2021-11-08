import os, argparse, math
import numpy as np
from glob import glob
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from medpy.metric.binary import hd, dc, assd, jc
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter

from src.LoadModel import load_model
from loss.losses import dice_loss, ce_loss, structure_loss, focal_loss

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='Transformer')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--net_layer', type=int, default=50)
parser.add_argument('--dataset', type=str, default='isbi2018')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--lr_seg', type=float, default=0.0003)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--bt_size', type=int, default=4)
parser.add_argument('--seg_loss', type=int, default=0, choices=[0, 1])
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--patience', type=int, default=50)

# using transformer
parser.add_argument('--trans', type=int, default=1)

# using key-patch map
parser.add_argument('--point_pred', type=int, default=0)

parse_config = parser.parse_args()
print(parse_config)

exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
    parse_config.seg_loss) + '_aug_' + str(
        parse_config.aug) + '/' + parse_config.arch + '/fold_' + str(
            parse_config.fold)

os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
writer = SummaryWriter('logs/{}/log'.format(exp_name))
save_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

EPOCHS = parse_config.n_epochs
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device_ids = range(torch.cuda.device_count())

# define the data loader

if parse_config.dataset == 'isbi2018':
    from dataset.isbi2018 import norm01, myDataset
    dataset = myDataset(fold=parse_config.fold,
                        split='train',
                        aug=parse_config.aug)
    dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)

elif parse_config.dataset == 'isbi2016':
    from dataset.isbi2016 import norm01, myDataset
    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           shuffle=True,
                                           num_workers=2,
                                           pin_memory=True,
                                           drop_last=True)
valid_loader = torch.utils.data.DataLoader(dataset2,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=2,
                                           pin_memory=True,
                                           drop_last=False)

# define the model

if parse_config.arch is 'Transformer':
    if parse_config.trans == 1 and parse_config.point_pred == 1:
        from models.BAT import BAT
        model = BAT(1, parse_config.net_layer).cuda()
    elif parse_config.trans == 1 and parse_config.point_pred == 0:
        from models.Transformer_Seg import Transformer
        model = Transformer(1, parse_config.net_layer).cuda()
    else:
        from models.Base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()
else:
    raise NotImplementedError

if len(device_ids) > 1:  # training with multiple GPUs
    model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

criteon = [dice_loss, ce_loss][parse_config.seg_loss]

##############################


def train(epoch):
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()

        if parse_config.net_layer == 18:
            point = nn.functional.max_pool2d(point,
                                             kernel_size=(8, 8),
                                             stride=(8, 8))
        else:
            point = nn.functional.max_pool2d(point,
                                             kernel_size=(16, 16),
                                             stride=(16, 16))

        if parse_config.arch == 'transfuse':
            lateral_map_4, lateral_map_3, lateral_map_2 = model(data)

            loss4 = structure_loss(lateral_map_4, label)
            loss3 = structure_loss(lateral_map_3, label)
            loss2 = structure_loss(lateral_map_2, label)

            loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('loss/loss_4', loss4,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/loss_3', loss3,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/loss_2', loss2,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/loss', loss,
                                  batch_idx + epoch * len(train_loader))
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\t[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'
                    .format(epoch, batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss2.item(),
                            loss3.item(), loss4.item()))

        elif parse_config.point_pred == 0:
            output = model(data)
            output = torch.sigmoid(output)

            loss = criteon(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('loss/loss', loss,
                                  batch_idx + epoch * len(train_loader))
                writer.add_image('label', label[0],
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('output', output[0] > 0.5,
                                 batch_idx + epoch * len(train_loader))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        elif parse_config.point_pred == 1:
            output, point_maps_pre = model(data)
            output = torch.sigmoid(output)

            assert (output.shape == label.shape)
            loss_dc = criteon(output, label)
            assert (point_maps_pre[-1].shape == point.shape)
            point_loss = 0.
            for i in range(len(point_maps_pre) - 1):
                point_loss += focal_loss(point_maps_pre[i], point)
            point_loss = 1.0 / len(point_maps_pre) * (
                point_loss + focal_loss(point_maps_pre[-1], point))

            loss = loss_dc + 3 * point_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('loss/dc_loss', loss_dc,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/point_loss', point_loss,
                                  batch_idx + epoch * len(train_loader))
                writer.add_scalar('loss/loss', loss,
                                  batch_idx + epoch * len(train_loader))

                writer.add_image('label', label[0],
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('output', output[0] > 0.5,
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('point_pre', point_maps_pre[-1][0],
                                 batch_idx + epoch * len(train_loader))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    dice_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        point = nn.functional.max_pool2d(point,
                                         kernel_size=(16, 16),
                                         stride=(16, 16))

        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
                loss_fuse = structure_loss(output, label)
            elif parse_config.point_pred == 0:
                output = model(data)
            elif parse_config.point_pred == 1:
                output, point_maps_pre = model(data)
                point_loss = 0.
                for i in range(len(point_maps_pre) - 1):
                    point_loss += focal_loss(point_maps_pre[i], point)
                point_loss = 1.0 / len(point_maps_pre) * (
                    point_loss + focal_loss(point_maps_pre[-1], point))

            output = torch.sigmoid(output)

            loss_dc = criteon(output, label)

            if parse_config.arch == 'transfuse':
                loss = loss_fuse
            elif parse_config.arch == 'transunet':
                loss = 0.5 * loss_dc + 0.5 * ce_loss(output, label)
            elif parse_config.point_pred == 0:
                loss = loss_dc
            elif parse_config.point_pred == 1:
                loss = loss_dc + 3 * point_loss

            output = output.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        dice_ave = dc(output, label)
        dice_value += dice_ave
        numm += 1

    dice_average = dice_value / numm
    writer.add_scalar('valid_dice', dice_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    return dice_average, loss


max_dice = 0
best_ep = 0
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    dice, valid_loss = evaluation(epoch, valid_loader)
    scheduler.step(valid_loss)

    if dice > max_dice:
        max_dice = dice
        best_ep = epoch
        torch.save(model.state_dict(), save_path)
    else:
        if epoch - best_ep >= parse_config.patience:
            print('Early stopping!')
            break
    torch.save(model.state_dict(), latest_path)