# ------------------------------------------
# ------------------------------------------
# IMPORTS
# ------------------------------------------
# ------------------------------------------

# import pandas as pd
import numpy as np

import glob

# import pdb

# from tqdm import tqdm
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
# from torchvision import models
from torch import nn # optim
# from torch.optim import lr_scheduler
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss
import torch.nn.functional as F

from pipeline.train_test_split import split_regions

from PIL import Image
import random
import math
import pdb

from FastFCN import encoding
from FastFCN.encoding import dilated as resnet
from FastFCN.encoding.utils import batch_pix_accuracy, batch_intersection_union


# ---- Image Utitilies ----


def mytransform(image, mask):
    '''
    Custom Pytorch randomized preprocessing of training image and mask.
    '''
    image = transforms.functional.pad(image, padding=0, padding_mode='reflect')
    crop_loc = np.random.randint(0, 728, 2)
    image = transforms.functional.crop(image, *crop_loc, 512, 512)
    mask = transforms.functional.crop(mask, *crop_loc, 512, 512)
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    return image, mask


# ---- Dataset Class ----


class MyDataset(Dataset):
    '''
    Custom PyTorch Dataset class.
    '''
    def __init__(self, path='tmp', transforms=None, load_test=False, split=None, batch_trim=False):

        self.transforms = transforms
        self.load_test = load_test
        
        if self.load_test:
            print('Loading Test')
            self.path = 'submission_data/test'
            self.images = os.listdir(self.path)
        else:
            self.path = path
            self.images = glob.glob(os.path.join(path, 'images','*.jpg'))
            self.masks = glob.glob(os.path.join(path, 'masks','*.jpg'))

            if split is not None:
                self.images, self.masks = split_regions(self.images, self.masks, split=split)
                pdb.set_trace()

            # self.images = [fn for fn in self.images if fn.endswith('.jpg')]
            # self.masks = [fn for fn in self.masks if fn.endswith('.jpg')]
            self.coordinates = None
        
        # Option to dataset for speedy development training to subset of batches
        if batch_trim:
            if batch_trim * 16 > len(self.images):
                pass
            else:
                self.images, self.masks = self.images[:int(trim)*16], self.masks[:int(trim)*16]

    def __getitem__(self, index):
        # print(index)
        if self.load_test:
            img_name = self.images[index]
            image = Image.open(os.path.join(self.path, img_name, img_name + '.tif'))
            image_tensor = transforms.functional.to_tensor(image)[:3]
            return image_tensor, img_name
        else:
            image = Image.open(self.images[index])
            mask = Image.open(self.masks[index])
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return (image, mask)

    def __len__(self):
        return len(self.images)


# ---- Load Dataset ----

def get_dataloader(path=None, load_test=False, batch_size=16, batch_trim=False):
    '''
    Load pytorch batch data loader only
    '''
    if path is None:
        path = 'tmp'
    print('Load test:', load_test)
    batch_loader = DataLoader(
        MyDataset(path, transforms=mytransform, load_test=load_test, batch_trim=False),
        shuffle=True, batch_size=batch_size
        )
    print('Dataset Loaded:')

    # sample_batch = next(iter(batch_loader))[0]
    # print('Batch Shape -', sample_batch[0].shape)
    # print('Single Image Shape -', sample_batch[0][0].shape)
    # print('Single Mask Shape –', sample_batch[0][1].shape)
    
    return batch_loader

# ------------------------------------------
# ------------------------------------------
# Optimizer and Loss Functions
# ------------------------------------------
# ------------------------------------------

# ---- Segmentation Loss ----

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target.detach())
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass, requires_grad=True)
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect


class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`
        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplementedError
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


# ------------------------------------------
# ------------------------------------------
# Network Modules
# ------------------------------------------
# ------------------------------------------


# ----- Base Network ----- #
UP_KWARGS = {'mode': 'bilinear', 'align_corners': True}

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, jpu=True, dilated=False, norm_layer=None,
                 base_size=520, crop_size=480, mean=[.485, .456, .406],
                 std=[.229, .224, .225], root='~/.encoding/models', **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.pretrained = resnet.resnet50(pretrained=True, dilated=dilated,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet.resnet101(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet.resnet152(pretrained=True, dilated=dilated,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # don't keep track of gradients in pretrained model!
        for param in self.pretrained.parameters():
            param.requires_grad = False
        # bilinear upsample options
        self._up_kwargs = UP_KWARGS
        self.backbone = backbone
        self.jpu = JPU([512, 1024, 2048], width=512, norm_layer=norm_layer, up_kwargs=self._up_kwargs) if jpu else None

    def base_forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        if self.jpu:
            return self.jpu(c1, c2, c3, c4)
        else:
          return c1, c2, c3, c4

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(pred.data, target.data, self.nclass)
        return correct, labeled, inter, union

# ----- EncNet Module ----- #
class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True,
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead([512, 1024, 2048], self.nclass, se_loss=se_loss, jpu=kwargs['jpu'],
                            lateral=kwargs['lateral'], norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


# ----- ENCHEAD ------

class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                  norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels[-1], 512, 1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU(inplace=True)) if jpu else \
                      nn.Sequential(nn.Conv2d(in_channels[-1], 512, 3, padding=1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU(inplace=True))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[0], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
                nn.Sequential(
                    nn.Conv2d(in_channels[1], 512, kernel_size=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True)),
            ])
            self.fusion = nn.Sequential(
                    nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
        self.encmodule = EncModule(512, out_channels, ncodes=32,
            se_loss=se_loss, norm_layer=norm_layer)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                    nn.Conv2d(512, out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c2 = self.connect[0](inputs[1])
            c3 = self.connect[1](inputs[2])
            feat = self.fusion(torch.cat([feat, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)

# ----- FCN Head ----- #
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

# ----- Enc Module ----- #

class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(inplace=True),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)

# ----- Separable Convolutions ------

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


# ----- Joint Pyramid Upsampling (JPU) ------

class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None, up_kwargs=None):
        super(JPU, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       norm_layer(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat



# ------------------------------------------
# ------------------------------------------
# Model Compiler
# ------------------------------------------
# ------------------------------------------



def get_model(args):
    ''' Return Modified EncNet / FastFCN, with ResNet backbone.'''

    # UP_KWARGS = {'mode': 'bilinear', 'align_corners': True}        

    
    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'citys': 240,
            'pascal_voc': 50,
            'pascal_aug': 50,
            'pcontext': 80,
            'ade20k': 120,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.batch_size is None:
        args.batch_size = 16
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
    if args.lr is None:
        lrs = {
            'coco': 0.01,
            'citys': 0.01,
            'pascal_voc': 0.0001,
            'pascal_aug': 0.001,
            'pcontext': 0.001,
            'ade20k': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / 16 * args.batch_size

    torch.manual_seed(args.seed)
    return EncNet(2, backbone=args.backbone, root='FastFCN/encoding/models',
                        dilated = args.dilated, lateral=args.lateral, jpu=args.jpu, aux=args.aux,
                        se_loss = args.se_loss, norm_layer = nn.BatchNorm2d,
                        base_size = args.base_size, crop_size=args.crop_size)