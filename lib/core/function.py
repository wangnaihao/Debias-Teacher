# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Rongchang Xie (rongchangxie@pku.edu.cn) 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import wandb
import logging
import time
import copy
import os
import math
import numpy as np
import torch.nn as nn
import torch
from torch.cuda.amp import autocast, GradScaler
from core.config import get_model_name
from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.loss import JointsMSELoss
import cv2
import torch.nn.functional as F

logger = logging.getLogger(__name__)



def loss_dist(net1_param, net2_param,mu):
    loss = 0
    for param1, param2 in zip(net1_param, net2_param):
        loss += torch.nn.functional.mse_loss(param1, param2)
    return loss ** mu

def train(config, train_loader, model, criterion, optimizer, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    scaler = GradScaler()
    # switch to train mode
    model.train()
    end = time.time()
    dis_batches = []
    clean = 0
    noise = 0
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # input, target, target_weight, meta = input.cuda(), target.cuda(), target_weight.cuda(), meta.cuda()
        # compute output
        output = model(input)
        sup_ht, unsup_ht, unsup_ht_trans, cons_ht, con_ht, lam, indices, (clean_tmp,noise_tmp,confi), y_l, y_ul = output
        clean += clean_tmp
        noise += noise_tmp
        '''params = list(model.parameters())
        k = 0
        for p in params:
            l = 1
            print(str(list(p.size())))
            for j in p.size():
                l *= j
            print(str(l))
            k = k + l
        print("sum parameters" + str(k))
'''


        if type(target) == list:
            target = [t.cuda(non_blocking=True) for t in target]
            assert len(target_weight[1]) == len(confi)
            # print(confi.shape)
            if config.REWEIGHT:
                target_weight[1] = torch.mul(target_weight[1],confi)
            target_weight = [w.cuda(non_blocking=True) for w in target_weight]
        else:
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

        with autocast():
            loss = criterion(output, target, target_weight, meta)
            if config.TRAIN.WARM_EPOCH > 0 :
                if epoch <= config.TRAIN.WARM_EPOCH:
                    # print("warm!!!!")
                    if type(loss) == tuple:
                        sum_loss = config.WEIGHT.LOSS_POSE * loss[1]['loss_pose']
                        loss_dic = loss[1]
                        if config.MODEL.NAME in ['pose_dual']:
                            pseudo_target = loss[2]
                    else:
                        sum_loss = loss
                        loss_dic = {}
                else:
                    if type(loss) == tuple:
                        sum_loss = loss[0]
                        loss_dic = loss[1]
                        if config.MODEL.NAME in ['pose_dual']:
                            pseudo_target = loss[2]
                    else:
                        sum_loss = loss
                        loss_dic = {}
            else:
                if type(loss) == tuple:
                    sum_loss = loss[0]
                    loss_dic = loss[1]
                    if config.MODEL.NAME in ['pose_dual']:
                        pseudo_target = loss[2]
                else:
                    sum_loss = loss
                    loss_dic = {}

        # compute gradient and do update step
        optimizer.zero_grad()
        # sum_loss.backward()
        # optimizer.step()
        scaler.scale(sum_loss).backward()
        # scaler 更新参数，会先自动unscale梯度
        # 如果有nan或inf，自动跳过
        scaler.step(optimizer)
        # scaler factor更新
        scaler.update()
        model.step()
               

        # Get the supervised samples
        if config.MODEL.NAME in ['pose_dual','pose_cons','pose_cons1']:
            if type(target)==list:
                input, target, target_weight, meta = [input[0], target[0], target_weight[0], meta[0]]
            outputs = output[0]
            #lam = output[5]
        # measure accuracy and record loss
        losses.update(sum_loss.item(), input.size(0))

        hm_type='gaussian'
        
        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                            target.detach().cpu().numpy(), hm_type)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % config.PRINT_FREQ == 0 or i == len(train_loader)-1 ) and config.LOCAL_RANK==0:
            # print(list(mapping_model.parameters())[0])
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses, acc=acc)


            for key in loss_dic.keys():
                msg = msg + '\t{}: {:.6f}'.format(key, loss_dic[key])

            logger.info(msg)
            if config.DEBUG.DEBUG:
                if config.MODEL.NAME in ['pose_resnet']:

                    dirs = os.path.join(output_dir,'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'train'), i)
                    save_debug_images(config, input, meta, target, pred*4, outputs,
                                        prefix, None)
    print(f"clean num : {clean}")
    print(f"noise num : {noise}")

def mse_klloss(output, target):
    criterion = nn.MSELoss(reduction='mean')
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
        loss += 0.5 * criterion(heatmap_pred, heatmap_gt)
    return loss / num_joints

def cotrain(config, train_loader, model1,model2,criterion, optimizer1,optimizer2, epoch,
          output_dir, tb_log_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    scaler = GradScaler()

    # switch to train mode
    model1.train()
    model2.train()
    end = time.time()
    dis_batches = []
    clean = 0
    noise = 0
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # input, target, target_weight, meta = input.cuda(), target.cuda(), target_weight.cuda(), meta.cuda()
        # compute output
        output1 = model1(input)
        output2 = model2(input)
        sup_ht1, unsup_ht1, unsup_ht_trans1, cons_ht1, con_ht1, lam1, indices1, (clean_tmp1, noise_tmp1,_), y_l1, y_ul1 = output1
        sup_ht2, unsup_ht2, unsup_ht_trans2, cons_ht2, con_ht2, lam2, indices2, (clean_tmp2, noise_tmp2,_), y_l2, y_ul2 = output2
        clean += clean_tmp1
        noise += noise_tmp1
        '''params = list(model.parameters())
        k = 0
        for p in params:
            l = 1
            print(str(list(p.size())))
            for j in p.size():
                l *= j
            print(str(l))
            k = k + l
        print("sum parameters" + str(k))
'''

        if type(target) == list:
            target = [t.cuda(non_blocking=True) for t in target]
            target_weight = [w.cuda(non_blocking=True) for w in target_weight]
        else:
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)


        loss1 = criterion(output1, target, target_weight, meta)
        loss2 = criterion(output2, target, target_weight, meta)
        if config.TRAIN.WARM_EPOCH > 0:
            if epoch <= config.TRAIN.WARM_EPOCH:
                if type(loss1) == tuple:
                    sum_loss1 = config.WEIGHT.LOSS_POSE * loss1[1]['loss_pose']
                    loss_dic1 = loss1[1]
                    sum_loss2 = config.WEIGHT.LOSS_POSE * loss2[1]['loss_pose']
                    loss_dic2 = loss2[1]
                else:
                    sum_loss1 = loss1
                    sum_loss2 = loss2
                    loss_dic = {}
            else:
                if type(loss1) == tuple:
                    sum_loss1 = config.WEIGHT.LOSS_POSE * loss1[1]['loss_pose']
                    loss_dic1 = loss1[1]
                    sum_loss2 = config.WEIGHT.LOSS_POSE * loss2[1]['loss_pose']
                    loss_dic2 = loss2[1]
                else:
                    sum_loss1 = loss1
                    sum_loss2 = loss2
                    loss_dic = {}
        else:
            if type(loss1) == tuple:
                sum_loss1 = config.WEIGHT.LOSS_POSE * loss1[1]['loss_pose']
                loss_dic1 = loss1[1]
                sum_loss2 = config.WEIGHT.LOSS_POSE * loss2[1]['loss_pose']
                loss_dic2 = loss2[1]
            else:
                sum_loss1 = loss1
                sum_loss2 = loss2
                loss_dic1 = {}

        # compute gradient and do update step
        # optimizer.zero_grad()
        # sum_loss.backward()
        # optimizer.step()
        # model.step()
        optimizer1.zero_grad()
        sum_loss1.backward(retain_graph=True)
        optimizer1.step()
        optimizer2.zero_grad()
        sum_loss2.backward()
        optimizer2.step()
        model1.step()
        model2.step()

        # Get the supervised samples
        if config.MODEL.NAME in ['pose_dual', 'pose_cons', 'pose_cons1']:
            if type(target) == list:
                input, target, target_weight, meta = [input[0], target[0], target_weight[0], meta[0]]
            outputs = output1[0]
            # lam = output[5]
        # measure accuracy and record loss
        losses.update(sum_loss1.item(), input.size(0))

        hm_type = 'gaussian'

        _, avg_acc, cnt, pred = accuracy(outputs.detach().cpu().numpy(),
                                         target.detach().cpu().numpy(), hm_type)
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % config.PRINT_FREQ == 0 or i == len(train_loader) - 1) and config.LOCAL_RANK == 0:
            # print(list(mapping_model.parameters())[0])
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)

            for key in loss_dic1.keys():
                msg = msg + '\t{}: {:.6f}'.format(key, loss_dic1[key])

            logger.info(msg)
            if config.DEBUG.DEBUG:
                if config.MODEL.NAME in ['pose_resnet']:
                    dirs = os.path.join(output_dir, 'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'train'), i)
                    save_debug_images(config, input, meta, target, pred * 4, outputs1,
                                      prefix, None)
    print(f"clean num : {clean}")
    print(f"noise num : {noise}")


def validate(config, val_loader, val_dataset, model, criterion, output_dir,
             tb_log_dir, writer_dict=None,wandb = None):
    print("begin eval")
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    criterion = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT, config)

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_preds_2 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                           dtype=np.float32)
    all_preds_3 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                           dtype=np.float32)

    all_boxes = np.zeros((num_samples, 10))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if config.MODEL.NAME in ['pose_dual']:
                output_list = model(input)
                output = output_list[0]
                output_2 = output_list[1]
            else:
                output = model(input)

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                if config.MODEL.NAME in ['pose_dual']:
                    output_flipped_2 = output_flipped[1]
                    output_flipped = output_flipped[0]

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                else:
                    # Else Affine Transform the heatmap to shift 3/4 pixel
                    batch_size, joint_num, height, width = output_flipped.shape
                    shift_x = 1.5 / width
                    trans = cv2.getRotationMatrix2D((0, 0), 0, 1)
                    trans[0, -1] -= shift_x
                    trans = trans[np.newaxis, :]
                    trans = np.repeat(trans, batch_size, 0)
                    theta = torch.from_numpy(trans).cuda()

                    grid = F.affine_grid(theta, output_flipped.size()).float()
                    output_flipped = F.grid_sample(output_flipped, grid)

                output = (output + output_flipped) * 0.5
                # output = output_flipped

                if config.MODEL.NAME in ['pose_dual']:
                    output_flipped_2 = flip_back(output_flipped_2.cpu().numpy(),
                                                 val_dataset.flip_pairs)
                    output_flipped_2 = torch.from_numpy(output_flipped_2.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped_2[:, :, :, 1:] = \
                            output_flipped_2.clone()[:, :, :, 0:-1]
                    else:
                        # Else Affine Transform the heatmap to shift 3/4 pixel
                        batch_size, joint_num, height, width = output_flipped_2.shape
                        shift_x = 1.5 / width
                        trans = cv2.getRotationMatrix2D((0, 0), 0, 1)
                        trans[0, -1] -= shift_x
                        trans = trans[np.newaxis, :]
                        trans = np.repeat(trans, batch_size, 0)
                        theta = torch.from_numpy(trans).cuda()

                        grid = F.affine_grid(theta, output_flipped_2.size()).float()
                        output_flipped_2 = F.grid_sample(output_flipped_2, grid)

                    output_2 = (output_2 + output_flipped_2) * 0.5
                    # output_2 = output_flipped_2

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output, target, target_weight)

            hm_type = 'gaussian'

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy(), hm_type)

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            r = meta['rotation'].numpy()
            score = meta['score'].numpy()
            box = meta['raw_box'].numpy()

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s, r, hm_type)

            if config.MODEL.NAME in ['pose_dual']:
                preds_2, maxvals_2 = get_final_preds(
                    config, output_2.clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_2[idx:idx + num_images, :, 0:2] = preds_2[:, :, 0:2]
                all_preds_2[idx:idx + num_images, :, 2:3] = maxvals_2

                preds_3, maxvals_3 = get_final_preds(
                    config, (0.5 * output + 0.5 * output_2).clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_3[idx:idx + num_images, :, 0:2] = preds_3[:, :, 0:2]
                all_preds_3[idx:idx + num_images, :, 2:3] = maxvals_3

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]

            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            # print(box)
            all_boxes[idx:idx + num_images, 6:] = np.array(box)
            image_path.extend(meta['image'])
            if config.DATASET.TEST_DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if (i % config.PRINT_FREQ == 0 or i == len(val_loader) - 1) and config.LOCAL_RANK == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)
                if i % 100 == 0:
                    with open('loss.txt', 'a') as f:
                        f.write(msg)
                        f.write('\n')
                if config.DEBUG.DEBUG:
                    # input = model.module.x
                    dirs = os.path.join(output_dir, 'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                    save_debug_images(config, input, meta, target, pred * 4, output, prefix)

                    if config.MODEL.NAME in ['pose_dual']:
                        dirs = os.path.join(output_dir, 'sup_2')
                        checkdir(dirs)
                        prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                        save_debug_images(config, input, meta, target, pred * 4, output_2, prefix)

        np.save(os.path.join(output_dir, 'all_preds.npy'), all_preds)

        if config.LOCAL_RANK != 0:
            return 0
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, prefix='eval_1')

        _, full_arch_name = get_model_name(config)
        logger.info('The Predictions of Net 1')

        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, full_arch_name)
            if 'AP' in name_value.keys():
                wandb.log({
                    "AP": name_value['AP']
                })
        else:
            _print_name_value(name_values, full_arch_name)
            wandb.log({
                'AP' : name_values['AP']
            })
        if config.MODEL.NAME in ['pose_dual']:
            name_values_2, perf_indicator_2 = val_dataset.evaluate(
                config, all_preds_2, output_dir, all_boxes, image_path,
                'head2_pred.mat', imgnums, prefix='eval_2')
            logger.info('The Predictions of Net 2')
            if isinstance(name_values_2, list):

                for name_value in name_values_2:
                    if 'AP' in name_value.keys():
                        wandb.log({
                            "AP":name_value['AP']
                        })
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values_2, full_arch_name)
                wandb.log({
                    "AP": name_value['AP']
                })

            name_values_3, perf_indicator_3 = val_dataset.evaluate(
                config, all_preds_3, output_dir, all_boxes, image_path,
                'ensemble_pred.mat', imgnums, prefix='eval_3')

            logger.info('Ensemble Predictions')
            _print_name_value(name_values_3, full_arch_name)

    return perf_indicator


def co_validate(config, val_loader, val_dataset, model1,model2, criterion, output_dir,
             tb_log_dir, writer_dict=None,wandb = None):
    print("begin eval")
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()


    criterion = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT,config)

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_preds_2 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_preds_3 = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)

    all_boxes = np.zeros((num_samples, 10))
    image_path = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            if config.MODEL.NAME in ['pose_dual']:
                output_list = model(input)
                output = output_list[0]
                output_2 = output_list[1]
            else:
                output1 = model1(input)
                output2 = model2(input)
                output_mean = (output1 + output2) / 2

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                output_flipped = model(input_flipped)
                if config.MODEL.NAME in ['pose_dual']:
                    output_flipped_2 = output_flipped[1]
                    output_flipped = output_flipped[0]
                    
                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                else:
                # Else Affine Transform the heatmap to shift 3/4 pixel
                    batch_size,joint_num,height,width = output_flipped.shape
                    shift_x = 1.5/width
                    trans = cv2.getRotationMatrix2D((0,0), 0, 1)
                    trans[0,-1] -= shift_x
                    trans = trans[np.newaxis, :]
                    trans = np.repeat(trans,batch_size,0)
                    theta = torch.from_numpy(trans).cuda()

                    grid = F.affine_grid(theta, output_flipped.size()).float()
                    output_flipped = F.grid_sample(output_flipped, grid)

                output = (output + output_flipped) * 0.5
                # output = output_flipped

                if config.MODEL.NAME in ['pose_dual']:
                    output_flipped_2 = flip_back(output_flipped_2.cpu().numpy(),
                                            val_dataset.flip_pairs)
                    output_flipped_2 = torch.from_numpy(output_flipped_2.copy()).cuda()

                    # feature is not aligned, shift flipped heatmap for higher accuracy
                    if config.TEST.SHIFT_HEATMAP:
                        output_flipped_2[:, :, :, 1:] = \
                            output_flipped_2.clone()[:, :, :, 0:-1]
                    else:
                    # Else Affine Transform the heatmap to shift 3/4 pixel
                        batch_size,joint_num,height,width = output_flipped_2.shape
                        shift_x = 1.5/width
                        trans = cv2.getRotationMatrix2D((0,0), 0, 1)
                        trans[0,-1] -= shift_x
                        trans = trans[np.newaxis, :]
                        trans = np.repeat(trans,batch_size,0)
                        theta = torch.from_numpy(trans).cuda()

                        grid = F.affine_grid(theta, output_flipped_2.size()).float()
                        output_flipped_2 = F.grid_sample(output_flipped_2, grid)

                    output_2 = (output_2 + output_flipped_2) * 0.5
                    # output_2 = output_flipped_2

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = criterion(output_mean, target, target_weight)

            hm_type='gaussian'

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output_mean.cpu().numpy(),
                                             target.cpu().numpy(),hm_type)

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            r = meta['rotation'].numpy()
            score = meta['score'].numpy()
            box = meta['raw_box'].numpy()

            preds1, maxvals1 = get_final_preds(
                config, output1.clone().cpu().numpy(), c, s, r, hm_type)

            preds2, maxvals2 = get_final_preds(
                config, output2.clone().cpu().numpy(), c, s, r, hm_type)

            preds_mean, maxvals_mean = get_final_preds(
                config, output_mean.clone().cpu().numpy(), c, s, r, hm_type)

            if config.MODEL.NAME in ['pose_dual']:
                preds_2, maxvals_2 = get_final_preds(
                        config, output_2.clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_2[idx:idx + num_images, :, 0:2] = preds_2[:, :, 0:2]
                all_preds_2[idx:idx + num_images, :, 2:3] = maxvals_2

                preds_3, maxvals_3 = get_final_preds(
                        config, (0.5*output+0.5*output_2).clone().cpu().numpy(), c, s, r, hm_type)
                all_preds_3[idx:idx + num_images, :, 0:2] = preds_3[:, :, 0:2]
                all_preds_3[idx:idx + num_images, :, 2:3] = maxvals_3


            all_preds_2[idx:idx + num_images, :, 0:2] = preds1[:, :, 0:2]
            all_preds_2[idx:idx + num_images, :, 2:3] = maxvals1

            all_preds_3[idx:idx + num_images, :, 0:2] = preds2[:, :, 0:2]
            all_preds_3[idx:idx + num_images, :, 2:3] = maxvals2

            all_preds[idx:idx + num_images, :, 0:2] = preds_mean[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals_mean

            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]

            all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
            all_boxes[idx:idx + num_images, 5] = score

            # print(box)
            all_boxes[idx:idx + num_images, 6:] = np.array(box)
            image_path.extend(meta['image'])
            if config.DATASET.TEST_DATASET == 'posetrack':
                filenames.extend(meta['filename'])
                imgnums.extend(meta['imgnum'].numpy())

            idx += num_images

            if (i % config.PRINT_FREQ == 0 or i == len(val_loader) - 1 ) and config.LOCAL_RANK==0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acc)
                logger.info(msg)
                if i % 100 == 0:
                    with open('loss.txt','a') as f:
                        f.write(msg)
                        f.write('\n')
                if config.DEBUG.DEBUG:
                    # input = model.module.x
                    dirs = os.path.join(output_dir,'heatmap')
                    checkdir(dirs)
                    prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                    save_debug_images(config, input, meta, target, pred*4, output, prefix)

                    if config.MODEL.NAME in ['pose_dual']:
                        dirs = os.path.join(output_dir,'sup_2')
                        checkdir(dirs)
                        prefix = '{}_{}'.format(os.path.join(dirs, 'val'), i)
                        save_debug_images(config, input, meta, target, pred*4, output_2,prefix)

        np.save(os.path.join(output_dir,'all_preds.npy'), all_preds)

        if config.LOCAL_RANK!=0:
            return 0
        name_values1, perf_indicator1 = val_dataset.evaluate(
            config, all_preds_2, output_dir, all_boxes, image_path,
            filenames, imgnums, prefix = 'eval_1')

        name_values2, perf_indicator2 = val_dataset.evaluate(
            config, all_preds_3, output_dir, all_boxes, image_path,
            filenames, imgnums, prefix='eval_2')

        name_values_mean, perf_indicator_mean = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_path,
            filenames, imgnums, prefix='eval_mean')

        _, full_arch_name = get_model_name(config)
        logger.info('The Predictions of Net 1')

        if isinstance(name_values1, list):
            for name_value in name_values1:
                _print_name_value(name_value, full_arch_name)
                if 'AP' in name_value.keys():
                    wandb.log({
                        "AP1": name_value['AP']
                    })
        else:
            _print_name_value(name_values1, full_arch_name)
            wandb.log({
                "AP1": name_values1['AP']
            })

        if isinstance(name_values2, list):
            for name_value in name_values2:
                _print_name_value(name_value, full_arch_name)
                if 'AP' in name_value.keys():
                    wandb.log({
                        "AP2": name_value['AP']
                    })
        else:
            _print_name_value(name_values2, full_arch_name)
            wandb.log({
                "AP2": name_values2['AP']
            })

        if isinstance(name_values_mean, list):
            for name_value in name_values_mean:
                _print_name_value(name_value, full_arch_name)
                if 'AP' in name_value.keys():
                    wandb.log({
                        "APmean": name_value['AP']
                    })
        else:
            _print_name_value(name_values_mean, full_arch_name)
            wandb.log({
                "APmean": name_values_mean['AP']
            })

        if config.MODEL.NAME in ['pose_dual']:
            name_values_2, perf_indicator_2 = val_dataset.evaluate(
                config, all_preds_2, output_dir, all_boxes, image_path,
                'head2_pred.mat', imgnums, prefix = 'eval_2')
            logger.info('The Predictions of Net 2')
            if isinstance(name_values_2, list):
                for name_value in name_values_2:
                    _print_name_value(name_value, full_arch_name)
            else:
                _print_name_value(name_values_2, full_arch_name)

            name_values_3, perf_indicator_3 = val_dataset.evaluate(
                config, all_preds_3, output_dir, all_boxes, image_path,
                'ensemble_pred.mat', imgnums, prefix = 'eval_3')
                
            logger.info('Ensemble Predictions')
            _print_name_value(name_values_3, full_arch_name)

    return perf_indicator_mean

# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.4f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def checkdir(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
