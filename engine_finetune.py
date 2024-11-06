# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np




# def misc_measures(confusion_matrix):
    
#     acc = []
#     sensitivity = []
#     specificity = []
#     precision = []
#     G = []
#     F1_score_2 = []
#     mcc_ = []
    
#     for i in range(1, confusion_matrix.shape[0]):
#         cm1=confusion_matrix[i]
#         acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
#         sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
#         sensitivity.append(sensitivity_)
#         specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
#         specificity.append(specificity_)
#         precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
#         precision.append(precision_)
#         G.append(np.sqrt(sensitivity_*specificity_))
#         F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
#         mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
#         mcc_.append(mcc)
        
#     acc = np.array(acc).mean()
#     sensitivity = np.array(sensitivity).mean()
#     specificity = np.array(specificity).mean()
#     precision = np.array(precision).mean()
#     G = np.array(G).mean()
#     F1_score_2 = np.array(F1_score_2).mean()
#     mcc_ = np.array(mcc_).mean()
    
#     return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_


def misc_measures(confusion_matrix):
    try:
        # Verify the input is a numpy array
        if not isinstance(confusion_matrix, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        # Ensure the input is not empty
        if confusion_matrix.size == 0:
            raise ValueError("Input confusion matrix array is empty.")
        
        # Check if each sub-matrix has the expected 2x2 shape
        if any(cm.shape != (2, 2) for cm in confusion_matrix):
            raise ValueError("Each confusion matrix should be of shape 2x2.")
        
        acc, sensitivity, specificity, precision, G, F1_score_2, mcc_ = [], [], [], [], [], [], []
        
        for i, cm1 in enumerate(confusion_matrix):
            # Handle accuracy calculation
            try:
                acc_value = (cm1[0, 0] + cm1[1, 1]) / np.sum(cm1)
                acc.append(acc_value)
            except Exception as e:
                print(f"Warning: An error occurred with accuracy calculation for confusion matrix {i}: {e}")
                acc.append(float('nan'))

            # Handle sensitivity calculation
            try:
                sensitivity_ = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
                sensitivity.append(sensitivity_)
            except Exception as e:
                print(f"Warning: An error occurred with sensitivity calculation for confusion matrix {i}: {e}")
                sensitivity.append(float('nan'))

            # Handle specificity calculation
            try:
                specificity_ = cm1[0, 0] / (cm1[0, 1] + cm1[0, 0])
                specificity.append(specificity_)
            except Exception as e:
                print(f"Warning: An error occurred with specificity calculation for confusion matrix {i}: {e}")
                specificity.append(float('nan'))

            # Handle precision calculation
            try:
                precision_ = cm1[1, 1] / (cm1[1, 1] + cm1[0, 1])
                precision.append(precision_)
            except Exception as e:
                print(f"Warning: An error occurred with precision calculation for confusion matrix {i}: {e}")
                precision.append(float('nan'))

            # Handle G calculation
            try:
                G_value = np.sqrt(sensitivity_ * specificity_)
                G.append(G_value)
            except Exception as e:
                print(f"Warning: An error occurred with G calculation for confusion matrix {i}: {e}")
                G.append(float('nan'))

            # Handle F1 score calculation
            try:
                F1_score_2_value = 2 * precision_ * sensitivity_ / (precision_ + sensitivity_)
                F1_score_2.append(F1_score_2_value)
            except Exception as e:
                print(f"Warning: An error occurred with F1 score calculation for confusion matrix {i}: {e}")
                F1_score_2.append(float('nan'))

            # Handle MCC calculation
            try:
                mcc_value = (cm1[0, 0] * cm1[1, 1] - cm1[0, 1] * cm1[1, 0]) / \
                            np.sqrt((cm1[0, 0] + cm1[0, 1]) * (cm1[0, 0] + cm1[1, 0]) *
                                    (cm1[1, 1] + cm1[1, 0]) * (cm1[1, 1] + cm1[0, 1]))
                mcc_.append(mcc_value)
            except Exception as e:
                print(f"Warning: An error occurred with MCC calculation for confusion matrix {i}: {e}")
                mcc_.append(float('nan'))
        
        # Calculate the mean for each metric, ignoring NaN values
        acc = np.nanmean(acc)
        sensitivity = np.nanmean(sensitivity)
        specificity = np.nanmean(specificity)
        precision = np.nanmean(precision)
        G = np.nanmean(G)
        F1_score_2 = np.nanmean(F1_score_2)
        mcc_ = np.nanmean(mcc_)
        
        return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None, None


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    
    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))
        # print(f"output of model: {output}")
        # print(f"target: {target}")
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    print(f"true labels: {true_label_decode_list}")
    print(f"predicted labels: {prediction_decode_list}")
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    print(f"confusion_matrix : {confusion_matrix}")
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    try:
        auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    except Exception as e:
        print(f"auc_roc calculation failed: {e}")
        auc_roc = np.nan
        
    try:
        auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')
    except Exception as e:
        print(f"auc_pr calculation failed: {e}")
        auc_pr = np.nan
              
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f} Sensitivity: {:.4f} Specifity: {:.4f} Precision: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc, sensitivity, specificity, precision)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc,F1, sensitivity, specificity, precision

