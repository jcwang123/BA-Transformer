from ast import parse
import os, argparse, math, sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)
import numpy as np
from glob import glob
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from medpy.metric.binary import hd, dc, assd, jc
from src.utils import load_model
from src.losses import dice_loss

from scipy.ndimage import distance_transform_edt as distance
from torch.utils.tensorboard import SummaryWriter

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import time

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='BAT')
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--net_layer', type=int, default=50)
parser.add_argument('--dataset', type=str, default='isic2016')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--fold', type=str, default='0')
parser.add_argument('--lr_seg', type=float, default=1e-4)  #0.0003
parser.add_argument('--n_epochs', type=int, default=200)  #100
parser.add_argument('--bt_size', type=int, default=8)  #36
parser.add_argument('--seg_loss', type=int, default=0, choices=[0, 1])
parser.add_argument('--aug', type=int, default=1)
parser.add_argument('--patience', type=int, default=500)  #50

# pre-train
parser.add_argument('--pre', type=int, default=0)

# transformer
parser.add_argument('--trans', type=int, default=1)

# point constrain
parser.add_argument('--point_pred', type=int, default=1)
parser.add_argument('--ppl', type=int, default=6)

# cross-scale framework
parser.add_argument('--cross', type=int, default=0)

parse_config = parser.parse_args()
print(parse_config)

if parse_config.arch == 'BAT':
    parse_config.exp_name += '_{}_{}_{}_e{}'.format(parse_config.trans,
                                                    parse_config.point_pred,
                                                    parse_config.cross,
                                                    parse_config.ppl)
exp_name = parse_config.dataset + '/' + parse_config.exp_name + '_loss_' + str(
    parse_config.seg_loss) + '_aug_' + str(parse_config.aug) + '/fold_' + str(
        parse_config.fold)

os.makedirs('logs/{}'.format(exp_name), exist_ok=True)
os.makedirs('logs/{}/model'.format(exp_name), exist_ok=True)
writer = SummaryWriter('logs/{}/log'.format(exp_name))
save_path = 'logs/{}/model/best.pkl'.format(exp_name)
latest_path = 'logs/{}/model/latest.pkl'.format(exp_name)

EPOCHS = parse_config.n_epochs
os.environ['CUDA_VISIBLE_DEVICES'] = parse_config.gpu
device_ids = range(torch.cuda.device_count())

torch.set_num_threads(8)

if parse_config.dataset == 'isic2018':
    from dataset.isic2018 import norm01, myDataset
    dataset = myDataset(fold=parse_config.fold,
                        split='train',
                        aug=parse_config.aug)
    dataset2 = myDataset(fold=parse_config.fold, split='valid', aug=False)
#     dataset = myDataset(fold=parse_config.fold, split='train', aug=parse_config.aug)
elif parse_config.dataset == 'isic2016':
    from dataset.isic2016 import norm01, myDataset
    dataset = myDataset(split='train', aug=parse_config.aug)
    dataset2 = myDataset(split='valid', aug=False)

train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=parse_config.bt_size,
                                           shuffle=True,
                                           num_workers=2,
                                           pin_memory=True,
                                           drop_last=True)
if parse_config.arch == 'BAT':
    if parse_config.trans == 1:
        from Ours.Base_transformer import BAT
        model = BAT(1, parse_config.net_layer, parse_config.point_pred,
                    parse_config.ppl).cuda()
    else:
        from Ours.base import DeepLabV3
        model = DeepLabV3(1, parse_config.net_layer).cuda()

if len(device_ids) > 1:  # 多卡训练
    model = torch.nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=parse_config.lr_seg)

#scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)

scheduler = CosineAnnealingLR(optimizer, T_max=20)


def ce_loss(pred, gt):
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    return (-gt * torch.log(pred) - (1 - gt) * torch.log(1 - pred)).mean()


def structure_loss(pred, mask):
    """            TransFuse train loss        """
    """            Without sigmoid             """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.6,  #0.8
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="mean")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


criteon = [focal_loss, ce_loss][parse_config.seg_loss]

##############################


def train(epoch):
    model.train()
    iteration = 0
    for batch_idx, batch_data in enumerate(train_loader):
        #         print(epoch, batch_idx)
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        #point_All = (batch_data['point_All'] > 0).cuda().float()

        if parse_config.net_layer == 18:
            point_c4 = nn.functional.max_pool2d(point,
                                                kernel_size=(16, 16),
                                                stride=(16, 16))
            point = nn.functional.max_pool2d(point,
                                             kernel_size=(8, 8),
                                             stride=(8, 8))
        else:
            point_c5 = nn.functional.max_pool2d(point,
                                                kernel_size=(32, 32),
                                                stride=(32, 32))

            point_c4 = nn.functional.max_pool2d(point,
                                                kernel_size=(16, 16),
                                                stride=(16, 16))

        if parse_config.point_pred == 1:
            output, point_maps_pre = model(data)
            output = torch.sigmoid(output)

            #print("point_pre shape:{}, point shape:{}".format(point_pre.shape,point.shape))
            assert (output.shape == label.shape)
            loss_dc = dice_loss(output, label)
            # print(point_maps_pre[-1].shape, point_c4.shape)
            assert (point_maps_pre[-1].shape == point_c4.shape)

            point_loss = 0.
            for i in range(len(point_maps_pre)):
                point_loss += criteon(point_maps_pre[i], point_c4)
            point_loss = point_loss / len(point_maps_pre)

            loss = loss_dc + point_loss  # point_loss weight: 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration = iteration + 1

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
                writer.add_image('point', point_c4[0],
                                 batch_idx + epoch * len(train_loader))
                writer.add_image('point_pre', point_maps_pre[-1][0],
                                 batch_idx + epoch * len(train_loader))

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    print("Iteration numbers: ", iteration)


val_loader = torch.utils.data.DataLoader(
    dataset2,
    batch_size=1,  #parse_config.bt_size
    shuffle=False,  #True
    num_workers=2,
    pin_memory=True,
    drop_last=False)  #True


def evaluation(epoch, loader):
    model.eval()
    dice_value = 0
    iou_value = 0
    dice_average = 0
    iou_average = 0
    numm = 0
    for batch_idx, batch_data in enumerate(loader):
        data = batch_data['image'].cuda().float()
        label = batch_data['label'].cuda().float()
        point = (batch_data['point'] > 0).cuda().float()
        point_c5 = nn.functional.max_pool2d(point,
                                            kernel_size=(32, 32),
                                            stride=(32, 32))
        point_c4 = nn.functional.max_pool2d(point,
                                            kernel_size=(16, 16),
                                            stride=(16, 16))

        with torch.no_grad():
            if parse_config.arch == 'transfuse':
                _, _, output = model(data)
                loss_fuse = structure_loss(output, label)
            elif parse_config.point_pred == 0:
                output = model(data)
            elif parse_config.cross == 1 and parse_config.point_pred == 1:
                output, point_maps_pre_1, point_maps_pre_2 = model(data)
                point_loss_c4 = 0.
                for i in range(len(point_maps_pre_1) - 1):
                    point_loss_c4 += criteon(point_maps_pre_1[i], point_c4)
                point_loss_c4 = 1.0 / len(point_maps_pre_1) * (
                    point_loss_c4 + criteon(point_maps_pre_1[-1], point_c4))
                point_loss_c5 = 0.
                for i in range(len(point_maps_pre_2) - 1):
                    point_loss_c5 += criteon(point_maps_pre_2[i], point_c5)
                point_loss_c5 = 1.0 / len(point_maps_pre_2) * (
                    point_loss_c5 + criteon(point_maps_pre_2[-1], point_c5))
                point_loss = 0.5 * (point_loss_c4 + point_loss_c5)
            elif parse_config.point_pred == 1:
                output, point_maps_pre = model(data)
                point_loss = 0.
                for i in range(len(point_maps_pre) - 1):
                    point_loss += criteon(point_maps_pre[i], point_c4)
                point_loss = 1.0 / len(point_maps_pre) * (
                    point_loss + criteon(point_maps_pre[-1], point_c4))

            output = torch.sigmoid(output)

            loss_dc = dice_loss(output, label)

            if parse_config.arch == 'transfuse':
                loss = loss_fuse
            elif parse_config.arch == 'transunet':
                loss = 0.5 * loss_dc + 0.5 * ce_loss(output, label)
            elif parse_config.point_pred == 0:
                loss = loss_dc
            elif parse_config.cross == 1 and parse_config.point_pred == 1:
                loss = loss_dc + point_loss
            elif parse_config.point_pred == 1:
                loss = loss_dc + 3 * point_loss

            output = output.cpu().numpy() > 0.5

        label = label.cpu().numpy()
        assert (output.shape == label.shape)
        dice_ave = dc(output, label)
        iou_ave = jc(output, label)
        dice_value += dice_ave
        iou_value += iou_ave
        numm += 1

    dice_average = dice_value / numm
    iou_average = iou_value / numm
    writer.add_scalar('val_metrics/val_dice', dice_average, epoch)
    writer.add_scalar('val_metrics/val_iou', iou_average, epoch)
    print("Average dice value of evaluation dataset = ", dice_average)
    print("Average iou value of evaluation dataset = ", iou_average)
    return dice_average, iou_average, loss


max_dice = 0
max_iou = 0
best_ep = 0

min_loss = 10
min_epoch = 0
# evaluation(0, val_loader)
for epoch in range(1, EPOCHS + 1):
    #打印学习率 lr

    this_lr = optimizer.state_dict()['param_groups'][0]['lr']
    writer.add_scalar('Learning Rate', this_lr, epoch)
    start = time.time()
    train(epoch)
    dice, iou, loss = evaluation(epoch, val_loader)
    #scheduler.step(loss)
    scheduler.step()

    if loss < min_loss:
        min_epoch = epoch
        min_loss = loss
    else:
        if epoch - min_epoch >= parse_config.patience:
            print('Early stopping!')
            break
    #if dice > max_dice:
    #    max_dice = dice
    #    best_ep = epoch
    #    torch.save(model.state_dict(), save_path)
    if iou > max_iou:
        max_iou = iou
        best_ep = epoch
        torch.save(model.state_dict(), save_path)
    else:
        if epoch - best_ep >= parse_config.patience:
            print('Early stopping!')
            break
    torch.save(model.state_dict(), latest_path)
    time_elapsed = time.time() - start
    print('Training and evaluating on epoch:{} complete in {:.0f}m {:.0f}s'.
          format(epoch, time_elapsed // 60, time_elapsed % 60))
