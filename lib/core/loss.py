# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from core.inference import get_max_preds, get_max_preds_tensor


class WorstCaseEstimationLoss(nn.Module):
    def __init__(self, eta_prime=2, use_target_weight=None, cfg=None):
        super(WorstCaseEstimationLoss, self).__init__()
        self.eta_prime = eta_prime
        self.mse_criterion = JointsMSELoss(use_target_weight, cfg)

    def forward(self, y_l, y_l_adv, y_l_weight, y_u, y_u_adv, unsup_target_weight):
        # 正常的监督训练
        loss_l = self.eta_prime * self.mse_criterion(y_l_adv, y_l.detach(), y_l_weight.detach())

        # 用强增强的热图监督弱增强
        loss_u = self.mse_criterion(y_u_adv, y_u.detach(), unsup_target_weight.detach())
        return loss_l + loss_u


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.cfg = cfg

    def forward(self, output, target, target_weight=None, meta=None):
        batch_size = output.size(0)
        num_joints = output.size(1)
        height = output.size(2)
        width = output.size(3)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class FCMSELoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(FCMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.cfg = cfg

    def forward(self, output, target, target_weight=None, meta=None):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        # heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze(-1)
            # print("heatmap", heatmap_pred.shape)
            heatmap_gt = heatmaps_gt[idx].squeeze(-1)
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
        return loss / num_joints


class PoseCoLoss(nn.Module):
    def __init__(self, use_target_weight, cfg):
        super(PoseCoLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight, cfg)

        self.use_target_weight = use_target_weight

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = 24
        self.target_type = 'gaussian'

        self.cfg = cfg

    def forward(self, output, target, target_weight, meta):
        if type(target) == list:
            sup_target, unsup_target = target
            sup_target_weight, unsup_target_weight = target_weight
            sup_meta, upsup_meta = meta
        else:
            sup_target = target
            sup_target_weight = target_weight
            sup_meta = meta

        batch_size, joint_num, ht_height, ht_width = sup_target.shape
        pseudo_target = 0
        # add
        sup_ht1, sup_ht2, unsup_ht1, unsup_ht2, unsup_ht_trans1, unsup_ht_trans2, cons_ht1, cons_ht2, out_dic, y_l, y_ul = output

        batch_size = sup_ht1.size(0)
        num_joints = sup_ht1.size(1)

        loss_pose = 0.5 * self.mse_criterion(sup_ht1, sup_target, sup_target_weight)
        loss_pose += 0.5 * self.mse_criterion(sup_ht2, sup_target, sup_target_weight)

        loss_cons = self.mse_criterion(cons_ht1, unsup_ht_trans2.detach(), unsup_target_weight)
        loss_cons += self.mse_criterion(cons_ht2, unsup_ht_trans1.detach(), unsup_target_weight)

        pseudo_target = [unsup_ht_trans2.detach().cpu(), unsup_ht_trans1.detach().cpu()]

        loss = loss_pose + loss_cons
        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
        }

        return loss, loss_dic, pseudo_target


class PoseDisLoss(nn.Module):
    def __init__(self, use_target_weight, cfg=None):
        super(PoseDisLoss, self).__init__()
        self.mse_criterion = JointsMSELoss(use_target_weight, cfg)
        self.fc_criterion = FCMSELoss(use_target_weight, cfg)
        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.num_joints = 24
        self.target_type = 'gaussian'
        self.l = cfg.MODEL.L
        self.cfg = cfg
        # add
        self.worst_loss = WorstCaseEstimationLoss(2, use_target_weight, cfg)

    def forward(self, output, target, target_weight, meta):
        # unpackage
        sup_ht, unsup_ht, unsup_ht_trans, cons_ht, con_ht, lam, indices, _, y_l, y_ul = output

        batch_size, joint_num, ht_height, ht_width = sup_ht.shape

        sup_target, unsup_target = target
        sup_target_weight, unsup_target_weight = target_weight

        # JointsMSELoss of supervised sample
        # Loss Pose
        loss_pose = self.mse_criterion(sup_ht, sup_target, sup_target_weight)

        # preds, maxvals = get_max_preds_tensor(unsup_ht.detach())
        # loss_fea = torch.mean(-torch.sum(F.normalize(cons_feature, p=2, dim=1)*F.normalize(unsup_feature.detach(), p=2, dim=1), dim=1) + 1)
        # JointsMSELoss of unsupervised sample
        loss_cons = self.mse_criterion(cons_ht, unsup_ht_trans.detach(), unsup_target_weight)
        loss_mix = lam * self.mse_criterion(con_ht, unsup_ht.detach(), unsup_target_weight) + (
                    1 - lam) * self.mse_criterion(con_ht, unsup_ht[indices].detach(), unsup_target_weight)
        loss_worst = self.worst_loss(sup_target, y_l, sup_target_weight, unsup_ht_trans.detach(), y_ul,
                                     unsup_target_weight)
        # dst loss
        loss = float(self.cfg.WEIGHT.LOSS_POSE) * loss_pose + float(self.cfg.WEIGHT.LOSS_CONS) * loss_cons + float(
            self.cfg.WEIGHT.LOSS_MIX) * loss_mix + float(self.cfg.WEIGHT.LOSS_WORST) * loss_worst
        # no dst loss
        # loss = loss_pose + 0.7 * loss_cons + 0.7 * loss_mix
        loss_dic = {
            'loss_pose': loss_pose,
            'loss_cons': loss_cons,
            'loss_fea': loss_mix,
            'loss_worst': loss_worst
        }

        return loss, loss_dic


