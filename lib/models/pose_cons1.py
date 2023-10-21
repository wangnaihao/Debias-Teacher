# ------------------------------------------------------------------------------
# Written by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from torchvision import transforms
import torch
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from collections import OrderedDict
import numpy as np
import cv2
import random
from torch.autograd import Function
from .pose_hrnet import PoseHighResolutionNet
from core.inference import get_max_preds_tensor
import copy
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)
class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff= 1.):
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output) :
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start
        The forward and backward behaviours are:
        .. math::
            \mathcal{R}(x) = x,
            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.
        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:
        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- 伪 \dfrac{i}{N})} - (hi-lo) + lo
        where :math:`i` is the iteration step.
        Args:
            alpha (float, optional): :math:`伪`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha = 1.0, lo= 0.0, hi = 0.1,
                 max_iters= 1000, auto_step= False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_CAFFE(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_CAFFE, self).__init__()
        # add stride to conv1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.proj_output_dim = 24576
        self.proj_hidden_dim = 2048
        self.projector = nn.Sequential(nn.Linear(self.proj_output_dim, self.proj_hidden_dim),
                                       nn.BatchNorm1d(self.proj_hidden_dim),
                                       nn.ReLU())
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fea = self.layer4(x)
        feature = fea.view(-1, 24576)
        feature = self.projector(feature)
        fea = self.deconv_layers(fea)
        ht = self.final_layer(fea)
        '''feature = fea.view(-1, 24576)
        print(feature.shape)
        feature = self.projector(feature)'''
        return fea, ht

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> loading pretrained model {}'.format(pretrained))
            checkpoint = torch.load(pretrained, map_location='cpu')
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict_old = checkpoint['state_dict']
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith('module.'):
                        state_dict[key[7:]] = state_dict_old[key]
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                raise RuntimeError(
                    'No state_dict found in checkpoint file {}'.format(pretrained))

            if list(state_dict.keys())[0][:6] == 'resnet':
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def cutmix_based_on_keypoint_perception(image, joints, MASK_JOINT_NUM=4):
    image_tmp = image.clone()

    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, height, width = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = 0 + re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    center_x = copy.deepcopy(re_joints[:, :, 1])
    center_y = copy.deepcopy(re_joints[:, :, 0])

    x0 = re_joints[:, :, 1] - size[:, :, 1]
    y0 = re_joints[:, :, 0] - size[:, :, 0]

    x1 = re_joints[:, :, 1] + size[:, :, 1]
    y1 = re_joints[:, :, 0] + size[:, :, 0]

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)

    for i in range(N):
        ind = np.random.choice(J, MASK_JOINT_NUM)
        ind_2 = np.random.choice(J, MASK_JOINT_NUM)  ######
        img_id = np.random.randint(0, N)  #####
        ##

        for idx in range(len(ind)):
            j = ind[idx]
            j2 = ind_2[idx]

            x_start = center_x[i, j] - abs(x0[img_id, j2] - center_x[img_id, j2])  #######
            x_end = center_x[i, j] + abs(x1[img_id, j2] - center_x[img_id, j2])
            y_start = center_y[i, j] - abs(y0[img_id, j2] - center_y[img_id, j2])  ######
            y_end = center_y[i, j] + abs(y1[img_id, j2] - center_y[img_id, j2])

            x_start = torch.clamp(x_start, 0, width - 1)
            x_end = torch.clamp(x_end, 0, width - 1)
            y_start = torch.clamp(y_start, 0, height - 1)
            y_end = torch.clamp(y_end, 0, height - 1)

            offset_y = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-2] - \
                           image[i, :, y_start: y_end, x_start: x_end].shape[-2])
            offset_x = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-1] - \
                           image[i, :, y_start: y_end, x_start: x_end].shape[-1])
            offset_y_start = 0
            offset_x_start = 0
            offset_y_end = 0
            offset_x_end = 0
            if y_start == 0:
                offset_y_start = offset_y
            if x_start == 0:
                offset_x_start = offset_x
            if y_end == height - 1:
                offset_y_end = offset_y
            if x_end == width - 1:
                offset_x_end = offset_x

            if image[i, :, y_start: y_end, x_start: x_end].shape[-1] == 0 or \
                    image[i, :, y_start: y_end, x_start: x_end].shape[-2] == 0:  ##
                pass

            elif image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                 x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-1] == 0 or \
                    image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-2] == 0:  ##
                image[i, :, y_start: y_end, x_start: x_end] = 0

            elif image[i, :, y_start: y_end, x_start: x_end].shape != image[img_id, :,
                                                                      y0[img_id, j2] + offset_y_start: y1[
                                                                                                           img_id, j2] - offset_y_end,
                                                                      x0[img_id, j2] + offset_x_start: x1[
                                                                                                           img_id, j2] - offset_x_end].shape:
                image[i, :, y_start: y_end, x_start: x_end] = 0

            else:  ### keypoint cutmix
                image[i, :, y_start: y_end, x_start: x_end] = \
                    image_tmp[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end]

    return image

def cutmix_based_on_hard_easy(image, joints, pre,MASK_JOINT_NUM=4):
    # confi id
    confi_id = np.where(pre == 1)[0].tolist()
    # unconfi id
    unconfi_id = np.where(pre == 0)[0].tolist()
    image_tmp = image.clone()

    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, height, width = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = 0 + re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    center_x = copy.deepcopy(re_joints[:, :, 1])
    center_y = copy.deepcopy(re_joints[:, :, 0])

    x0 = re_joints[:, :, 1] - size[:, :, 1]
    y0 = re_joints[:, :, 0] - size[:, :, 0]

    x1 = re_joints[:, :, 1] + size[:, :, 1]
    y1 = re_joints[:, :, 0] + size[:, :, 0]

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)

    for i in unconfi_id:
        ind = np.random.choice(J, 2)
        ind_2 = np.random.choice(J, 2)  ######
        img_id = np.random.choice(confi_id,1)  #####
        ##

        for idx in range(len(ind)):
            j = ind[idx]
            j2 = ind_2[idx]

            x_start = center_x[i, j] - abs(x0[img_id, j2] - center_x[img_id, j2])  #######
            x_end = center_x[i, j] + abs(x1[img_id, j2] - center_x[img_id, j2])
            y_start = center_y[i, j] - abs(y0[img_id, j2] - center_y[img_id, j2])  ######
            y_end = center_y[i, j] + abs(y1[img_id, j2] - center_y[img_id, j2])

            x_start = torch.clamp(x_start, 0, width - 1)
            x_end = torch.clamp(x_end, 0, width - 1)
            y_start = torch.clamp(y_start, 0, height - 1)
            y_end = torch.clamp(y_end, 0, height - 1)

            offset_y = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-2] - \
                           image[i, :, y_start: y_end, x_start: x_end].shape[-2])
            offset_x = abs(image[img_id, :, y0[img_id, j2]: y1[img_id, j2], x0[img_id, j2]: x1[img_id, j2]].shape[-1] - \
                           image[i, :, y_start: y_end, x_start: x_end].shape[-1])
            offset_y_start = 0
            offset_x_start = 0
            offset_y_end = 0
            offset_x_end = 0
            if y_start == 0:
                offset_y_start = offset_y
            if x_start == 0:
                offset_x_start = offset_x
            if y_end == height - 1:
                offset_y_end = offset_y
            if x_end == width - 1:
                offset_x_end = offset_x

            if image[i, :, y_start: y_end, x_start: x_end].shape[-1] == 0 or \
                    image[i, :, y_start: y_end, x_start: x_end].shape[-2] == 0:  ##
                pass

            elif image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                 x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-1] == 0 or \
                    image[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end].shape[-2] == 0:  ##
                image[i, :, y_start: y_end, x_start: x_end] = 0

            elif image[i, :, y_start: y_end, x_start: x_end].shape != image[img_id, :,
                                                                      y0[img_id, j2] + offset_y_start: y1[
                                                                                                           img_id, j2] - offset_y_end,
                                                                      x0[img_id, j2] + offset_x_start: x1[
                                                                                                           img_id, j2] - offset_x_end].shape:
                image[i, :, y_start: y_end, x_start: x_end] = 0

            else:  ### keypoint cutmix
                image[i, :, y_start: y_end, x_start: x_end] = \
                    image_tmp[img_id, :, y0[img_id, j2] + offset_y_start: y1[img_id, j2] - offset_y_end,
                    x0[img_id, j2] + offset_x_start: x1[img_id, j2] - offset_x_end]
    for i in confi_id:
        ind = np.random.choice(J, MASK_JOINT_NUM)
        for j in ind:
            image[i, :, x0[i, j]:x1[i, j], y0[i, j]:y1[i, j]] = 0
    return image

def mask_joint(image, joints, l):
    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, width, height = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    x0 = re_joints[:, :, 0] - size[:, :, 0]
    y0 = re_joints[:, :, 1] - size[:, :, 1]

    x1 = re_joints[:, :, 0] + size[:, :, 0]
    y1 = re_joints[:, :, 1] + size[:, :, 1]

    torch.clamp_(x0, 0, width)
    torch.clamp_(x1, 0, width)
    torch.clamp_(y0, 0, height)
    torch.clamp_(y1, 0, height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        count = int(l[i] / 24 * 8)
        if count == 0:
            count = 2
        ind = np.random.choice(J, count)
        for j in ind:
            image[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0
    return image

def mask_joint1(image, joints, MASK_JOINT_NUM=4):
    ## N,J,2 joints
    N, J = joints.shape[:2]
    _, _, width, height = image.shape
    re_joints = joints[:, :, :2] + torch.randn((N, J, 2)).cuda() * 10
    re_joints = re_joints.int()
    size = torch.randint(10, 20, (N, J, 2)).int().cuda()

    x0 = re_joints[:, :, 0] - size[:, :, 0]
    y0 = re_joints[:, :, 1] - size[:, :, 1]

    x1 = re_joints[:, :, 0] + size[:, :, 0]
    y1 = re_joints[:, :, 1] + size[:, :, 1]

    torch.clamp_(x0, 0, width)
    torch.clamp_(x1, 0, width)
    torch.clamp_(y0, 0, height)
    torch.clamp_(y1, 0, height)

    for i in range(N):
        # num = np.random.randint(MASK_JOINT_NUM)
        # ind = np.random.choice(J, num)
        ind = np.random.choice(J, MASK_JOINT_NUM)

        for j in ind:
            image[i, :, y0[i, j]:y1[i, j], x0[i, j]:x1[i, j]] = 0
    return image

def cutmix_based_on_confi(image, joints, pre , l , MASK_JOINT_NUM=4):
    easy = np.sum(pre == 1)
    hard = np.sum(pre == 0)
    if easy > 0 and hard > 0:
        image = cutmix_based_on_hard_easy(image, joints, pre,MASK_JOINT_NUM=4)
    else:
        image = mask_joint(image, joints, l)
    return image

'''def mixup_data(sup_x, unsup_x, alpha=0.75):
        #Returns mixed inputs, pairs of targets, and lambda
    batch_size = sup_x.shape[0]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    lam = max(lam, 1-lam)
    mixed_x = lam * sup_x + (1 - lam) * unsup_x
        #y_a, y_b = y, y[index]
    return mixed_x, lam'''


class PoseCons(nn.Module):

    def __init__(self, resnet, cfg, resnet_tch=None, resnet_seg=None, alpha=1, **kwargs):
        super(PoseCons, self).__init__()
        self.resnet = resnet
        extra = cfg.MODEL.EXTRA

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        # self.pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(in_features=24, out_features=24, bias=False)
        '''self.fc_layer = nn.Conv2d(
            in_channels=512,
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        self.line = nn.Sequential(nn.Linear(512, 512), nn.Linear(512, 24))'''
        self.image_size = cfg.MODEL.IMAGE_SIZE

        self.cfg = cfg
        # self.alpha = alpha
        self.module_list = []
        for n, m in self.resnet.named_modules():
            print(n, type(n))
            self.module_list.append(m)
            if len(self.module_list) > 40:
                break
        # self.lam = np.random.beta(self.alpha, self.alpha)
        self.lam = 0.355
        # new add by wnh
        import copy
        # self.pesudo_head = copy.deepcopy(resnet.final_layer)
        self.worst_head = copy.deepcopy(resnet.final_layer)
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=8000,
                                                       auto_step=False)
        self.ema_model = copy.deepcopy(self.resnet).cuda()
        self.decay = 0.99

    def step(self):
        self.grl_layer.step()

    @torch.no_grad()
    def momentum_update_ema(self):
        for param_train, param_eval in zip(self.resnet.parameters(), self.ema_model.parameters()):
            param_eval.copy_(param_eval * self.decay + param_train.detach() * (1 - self.decay))
        for buffer_train, buffer_eval in zip(self.resnet.buffers(), self.ema_model.buffers()):
            buffer_eval.copy_(buffer_eval * self.decay + buffer_train * (1 - self.decay))

    def hook_modify(self, module, input, output):
        output = self.lam * output + (1 - self.lam) * output[self.indices]
        return output

    def get_batch_affine_transform(self, batch_size):

        sf = self.scale_factor
        rf = self.rotation_factor
        batch_trans = []
        for b in range(batch_size):
            r = 0
            s = 1
            c = self.image_size / 2
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.8 else 0

            trans = cv2.getRotationMatrix2D((0, 0), r, s)
            batch_trans.append(trans)

        batch_trans = np.stack(batch_trans, 0)
        batch_trans = torch.from_numpy(batch_trans).cuda()

        return batch_trans

    def forward(self, x, meta=None):
        if type(x) != list:
            x = x.cuda()
            if self.cfg.USE_EMA:
                return self.ema_model(x)[1]
            else:
                return self.resnet(x)[1]
        else:
            self.momentum_update_ema()
            # RandAug
            if self.cfg.CONS_RAND_AUG:
                sup_x, unsup_x, aug_unsup_x = [ele.cuda() for ele in x]
                cons_x = aug_unsup_x
                #add
                con_x = unsup_x
            else:
                sup_x, unsup_x = [ele.cuda() for ele in x]
                cons_x = unsup_x.clone()
                con_x = unsup_x.clone()
            batch_size = sup_x.shape[0]
            sup_ft, sup_ht = self.resnet(sup_x)
            # 做一个梯度翻转
            # print(f"sup_ft size is {sup_ft.shape}")
            sup_ft_adv = self.grl_layer(sup_ft)
            # print(f"sup_ft_adv size is {sup_ft_adv.shape}")
            sup_worstht = self.worst_head(sup_ft_adv)


            # Teacher

            k = np.random.randint(-1, len(self.module_list))
            if k == 0:
                k = len(self.module_list) - 1
            self.indices = torch.randperm(unsup_x.size(0)).cuda()
            if k == -1:
                con_x = con_x * self.lam + con_x[self.indices] * (1 - self.lam)
                con_ft, con_ht = self.resnet(con_x)
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                con_ft, con_ht = self.resnet(con_x)
                modifier_hook.remove()

            with torch.no_grad():
                unsup_ft, unsup_ht = self.resnet(unsup_x)
            with torch.no_grad():
                # Strong Augmentation #1
                # Joints Cutout
                if self.cfg.MASK_JOINT_NUM > 0:
                    preds, maxvals = get_max_preds_tensor(unsup_ht.detach())
                    vals, index = maxvals.sort(axis=1)
                    vals = vals.data.cpu().numpy()
                    vals_each = vals.reshape(1,-1)
                    vals_each = (vals_each.max() - vals_each) / (vals_each.max() - vals_each.min())
                    vals_each = vals_each.reshape(-1,1)
                    gmm1 = GaussianMixture(n_components=2, max_iter=30, tol=1e-2, reg_covar=5e-4)
                    gmm1.fit(vals_each)
                    prob1 = gmm1.predict_proba(vals_each)
                    confi_weight = prob1[:, gmm1.means_.argmin()]
                    confi_weight = torch.tensor(confi_weight).reshape(batch_size,-1).unsqueeze(-1)


                    # print(vals.shape)#[32, 24]
                    n = []
                    # add each keypoint to easy or hard
                    # [batch * NUM_JOINTS ,-1] :[544,1]
                    sum_vals = np.sum(vals,axis = 1)
                    sum_vals = (sum_vals.max() - sum_vals) / (sum_vals.max() - sum_vals.min())
                    sum_vals = sum_vals.reshape(-1,1)
                    # add gmm to divide into easy and hard
                    gmm = GaussianMixture(n_components=2, max_iter=30, tol=1e-2, reg_covar=5e-4)
                    gmm.fit(sum_vals)
                    prob = gmm.predict_proba(sum_vals)
                    prob = prob[:, gmm.means_.argmin()]
                    pre = np.zeros(len(prob))
                    # 1 means high confi -> easy , 0 is low confi - > hard
                    pre[prob > 0.5] = 1
                    for i in range(batch_size):
                        max = vals[i][self.cfg.MODEL.NUM_JOINTS - 1]
                        max_sub = max - vals[i][0]
                        for j in range(self.cfg.MODEL.NUM_JOINTS - 1):
                            sub = max - vals[i][j]
                            norm = sub / max_sub
                            if max < 0.05:
                                norm = 1
                            n.append(norm)
                    n = np.array(n).reshape(-1, self.cfg.MODEL.NUM_JOINTS - 1)
                    l = np.count_nonzero(n < 0.55, axis=1)
                    cons_x = cutmix_based_on_confi(cons_x,preds*4,pre,l,4)
                    # cons_x = mask_joint(cons_x, preds * 4, l)
                    # cons_x = cutmix_based_on_keypoint_perception(cons_x, preds * 4, 4)

            # Transfor
            # Apply Affine Transformation again for hard augmentation
            if self.cfg.UNSUP_TRANSFORM:
                theta = self.get_batch_affine_transform(sup_x.shape[0])
                grid = F.affine_grid(theta, cons_x.size()).float()
                cons_x = F.grid_sample(cons_x, grid)

                ht_grid = F.affine_grid(theta, unsup_ht.size()).float()
                unsup_ht_trans = F.grid_sample(unsup_ht, ht_grid)
            else:
                unsup_ht_trans = unsup_ht.detach()
            # Student
            cons_ft, cons_ht = self.resnet(cons_x)
            # 对强增强数据同样做梯度反转
            unsup_ft_adv = self.grl_layer(cons_ft)
            unsup_worstht = self.worst_head(unsup_ft_adv)
            # [32, 24, 1, 1]
            #[监督热图，弱增强热图（无梯度），强增强热图（无梯度），增强预测，混合预测。。]
            # count clean and hard num and return
            num_clean = np.sum(pre == 1)
            num_hard = np.sum(pre == 0)
            return [sup_ht, unsup_ht, unsup_ht_trans, cons_ht, con_ht, self.lam, self.indices, (num_clean,num_hard,confi_weight),sup_worstht,unsup_worstht]


def get_pose_net(cfg, is_train, **kwargs):
    if cfg.MODEL.BACKBONE == 'resnet':
        num_layers = cfg.MODEL.EXTRA.NUM_LAYERS
        style = cfg.MODEL.STYLE
        block_class, layers = resnet_spec[num_layers]

        if style == 'caffe':
            block_class = Bottleneck_CAFFE
        resnet = PoseResNet(block_class, layers, cfg, **kwargs)
    elif cfg.MODEL.BACKBONE == 'hrnet':
        resnet = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        resnet.init_weights(cfg.MODEL.PRETRAINED)

    model = PoseCons(resnet, cfg)

    return model
