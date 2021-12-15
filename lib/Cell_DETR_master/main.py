from argparse import ArgumentParser
import os
import setproctitle

# Manage command line arguments
parser = ArgumentParser()
parser.add_argument("--train", default=False, action="store_true",
                    help="Binary flag. If set training will be performed.")
parser.add_argument("--val", default=False, action="store_true",
                    help="Binary flag. If set validation will be performed.")
parser.add_argument("--test", default=False, action="store_true",
                    help="Binary flag. If set testing will be performed.")
parser.add_argument("--cuda_devices", default="0", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If multi GPU training should be utilized set flag.")
parser.add_argument("--cpu", default=False, action="store_true",
                    help="Binary flag. If set all operations are performed on the CPU.")
parser.add_argument("--epochs", default=200, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr_schedule", default=False, action="store_true",
                    help="Binary flag. If set the learning rate will be reduced after epoch 50 and 100.")
parser.add_argument("--ohem", default=False, action="store_true",
                    help="Binary flag. If set online heard example mining is utilized.")
parser.add_argument("--ohem_fraction", default=0.75, type=float,
                    help="Ohem fraction to be applied when performing ohem.")
parser.add_argument("--batch_size", default=4, type=int,
                    help="Batch size to be utilized while training.")
parser.add_argument("--path_to_data", default="../../BCS_Data/Cell_Instance_Segmentation_Regular_Traps", type=str,
                    help="Path to dataset.")
parser.add_argument("--augmentation_p", default=0.6, type=float,
                    help="Probability that data augmentation is applied on training data sample.")
parser.add_argument("--lr_main", default=1e-04, type=float,
                    help="Learning rate of the detr model (excluding backbone).")
parser.add_argument("--lr_backbone", default=1e-05, type=float,
                    help="Learning rate of the backbone network.")
parser.add_argument("--lr_segmentation_head", default=1e-06, type=float,
                    help="Learning rate of the segmentation head, only applied when seg head is trained exclusively.")
parser.add_argument("--no_pac", default=False, action="store_true",
                    help="Binary flag. If set no pixel adaptive convolutions will be utilized in the segmentation head.")
parser.add_argument("--load_model", default="", type=str,
                    help="Path to model to be loaded.")
parser.add_argument("--dropout", default=0.05, type=float,
                    help="Dropout factor to be used in model.")
parser.add_argument("--three_classes", default=False, action="store_true",
                    help="Binary flag, If set three classes (trap, cell of interest and add. cells) will be utilized.")
parser.add_argument("--softmax", default=False, action="store_true",
                    help="Binary flag, If set a softmax will be applied to the segmentation prediction instead sigmoid.")
parser.add_argument("--only_train_segmentation_head_after_epoch", default=150, type=int,
                    help="Number of epoch where only the segmentation head is trained.")
parser.add_argument("--no_deform_conv", default=False, action="store_true",
                    help="Binary flag. If set no deformable convolutions will be utilized.")
parser.add_argument("--no_pau", default=False, action="store_true",
                    help="Binary flag. If set no pade activation unit is utilized, however, a leaky ReLU is utilized.")

# Get arguments
args = parser.parse_args()

# Set device type
device = "cpu" if args.cpu else "cuda"

# Set cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

setproctitle.setproctitle("Cell-DETR")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modules.modulated_deform_conv import ModulatedDeformConvPack
from pade_activation_unit.utils import PAU

# Avoid data loader bug
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2 ** 12, rlimit[1]))

from detr import CellDETR
from dataset import CellInstanceSegmentation, collate_function_cell_instance_segmentation
from lossfunction import InstanceSegmentationLoss, SegmentationLoss, MultiClassSegmentationLoss
from model_wrapper import ModelWrapper
from segmentation import ResFeaturePyramidBlock, ResPACFeaturePyramidBlock

if __name__ == '__main__':
    # Init detr
    detr = CellDETR(num_classes=3 if args.three_classes else 2,
                    segmentation_head_block=ResPACFeaturePyramidBlock if not args.no_pac else ResFeaturePyramidBlock,
                    segmentation_head_final_activation=nn.Softmax if args.softmax else nn.Sigmoid,
                    backbone_convolution=nn.Conv2d if args.no_deform_conv else ModulatedDeformConvPack,
                    segmentation_head_convolution=nn.Conv2d if args.no_deform_conv else ModulatedDeformConvPack,
                    transformer_activation=nn.LeakyReLU if args.no_pau else PAU,
                    backbone_activation=nn.LeakyReLU if args.no_pau else PAU,
                    bounding_box_head_activation=nn.LeakyReLU if args.no_pau else PAU,
                    classification_head_activation=nn.LeakyReLU if args.no_pau else PAU,
                    segmentation_head_activation=nn.LeakyReLU if args.no_pau else PAU)
    if args.load_model != "":
        detr.load_state_dict(torch.load(args.load_model))
    # Print network
    print(detr)
    # Print number of parameters
    print("# DETR parameters", sum([p.numel() for p in detr.parameters()]))
    # Init optimizer
    detr_optimizer = torch.optim.AdamW(detr.get_parameters(lr_main=args.lr_main, lr_backbone=args.lr_backbone),
                                       weight_decay=1e-06)
    detr_segmentation_optimizer = torch.optim.AdamW(detr.get_segmentation_head_parameters(lr=args.lr_segmentation_head),
                                                    weight_decay=1e-06)
    # Init data parallel if utilized
    if args.data_parallel:
        detr = torch.nn.DataParallel(detr)
    # Init learning rate schedule if utilized
    if args.lr_schedule:
        learning_rate_schedule = torch.optim.lr_scheduler.MultiStepLR(detr_optimizer, milestones=[50, 100], gamma=0.1)
    else:
        learning_rate_schedule = None
    # Init datasets
    training_dataset = DataLoader(
        CellInstanceSegmentation(path=os.path.join(args.path_to_data, "train"),
                                 augmentation_p=args.augmentation_p, two_classes=not args.three_classes),
        collate_fn=collate_function_cell_instance_segmentation, batch_size=args.batch_size, num_workers=20,
        shuffle=True)
    validation_dataset = DataLoader(
        CellInstanceSegmentation(path=os.path.join(args.path_to_data, "val"),
                                 augmentation_p=0.0, two_classes=not args.three_classes),
        collate_fn=collate_function_cell_instance_segmentation, batch_size=1, num_workers=1, shuffle=False)
    test_dataset = DataLoader(
        CellInstanceSegmentation(path=os.path.join(args.path_to_data, "test"),
                                 augmentation_p=0.0, two_classes=not args.three_classes),
        collate_fn=collate_function_cell_instance_segmentation, batch_size=1, num_workers=1, shuffle=False)
    # Model wrapper
    model_wrapper = ModelWrapper(detr=detr,
                                 detr_optimizer=detr_optimizer,
                                 detr_segmentation_optimizer=detr_segmentation_optimizer,
                                 training_dataset=training_dataset,
                                 validation_dataset=validation_dataset,
                                 test_dataset=test_dataset,
                                 loss_function=InstanceSegmentationLoss(
                                     segmentation_loss=SegmentationLoss(),
                                     ohem=args.ohem,
                                     ohem_faction=args.ohem_fraction),
                                 device=device)
    # Perform training
    if args.train:
        model_wrapper.train(epochs=args.epochs,
                            optimize_only_segmentation_head_after_epoch=args.only_train_segmentation_head_after_epoch)
    # Perform validation
    if args.val:
        model_wrapper.validate(number_of_plots=30)
    # Perform testing
    if args.test:
        model_wrapper.test()
