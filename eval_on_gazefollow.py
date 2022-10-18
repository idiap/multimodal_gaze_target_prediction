# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
from torchvision import transforms
import torch.nn as nn

from model import BaselineModel, AttentionModelCombined
from dataset import GazeFollow
from config import *
from utils import imutils, evaluation
from transforms import _get_transform, _get_transform_modality

import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
import warnings
import pickle
from matplotlib import pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


def test(model_weights, val_loader=None, batch_size=48, device=0, mode='dict', save_path=None):

    # Prepare data
    if val_loader is None:
        print("Loading Data")
        transform = _get_transform()
        transform_modality = _get_transform_modality()
    
        val_dataset = GazeFollow(gazefollow_val_label, transform, transform_modality, input_size=input_resolution, output_size=output_resolution, test=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4)
    
    # Define device
    device = torch.device('cuda', device)

    # Load model
    print("Constructing model")
    if mode=='pt':
        pretrained_dict = torch.load(model_weights)
    elif mode=='dict':
        pretrained_dict = model_weights
    
    if pretrained_dict['modality'] == 'attention':
        model = AttentionModelCombined(cone_mode=pretrained_dict['cone_mode'], pred_inout=pretrained_dict['pred_inout'])
    else:
        model = BaselineModel(pretrained_dict['backbone_name'], pretrained_dict['modality'], cone_mode=pretrained_dict['cone_mode'], pred_inout=pretrained_dict['pred_inout'])
    model.cuda().to(device)
    
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict['model'])
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)
    gt_gaze = []; pred_hm = []; image_size = [] ; paths = []; pred_att = []; directions = []
    with torch.no_grad():
        for val_batch, (val_img, val_face, val_pose, val_depth, val_gaze_field, val_gt_direction, val_head_channel, val_gaze_heatmap, cont_gaze, imsize, path) in tqdm(enumerate(val_loader), total=len(val_loader)):
            
            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head_channels = val_head_channel.cuda().to(device)
            val_gaze_fields = val_gaze_field.cuda().to(device)
            val_depth_maps = val_depth.cuda().to(device)
            val_pose_maps = val_pose.cuda().to(device)
            val_gt_direction = val_gt_direction.cuda().to(device)
            
            # choose input modality
            if pretrained_dict['modality'] == 'image':
                model_input = val_images
            elif pretrained_dict['modality'] == 'pose':
                model_input = val_pose_maps
            elif pretrained_dict['modality'] == 'depth':
                model_input = val_depth_maps
            elif pretrained_dict['modality'] == 'attention':
                model_input = [val_images, val_depth_maps, val_pose_maps]
            if pretrained_dict['modality'] == 'attention':
                val_gaze_heatmap_pred, direction, val_inout_pred, val_att = model(model_input, val_faces, val_gaze_fields, val_head_channels)
                pred_att.extend(val_att.squeeze().cpu().numpy())
            else:
                val_gaze_heatmap_pred, direction, val_inout_pred = model(model_input, val_faces, val_gaze_fields, val_head_channels)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1)

            gt_gaze.extend(cont_gaze)
            pred_hm.extend(val_gaze_heatmap_pred.cpu().numpy())
            image_size.extend(imsize)
            paths.extend(path)
            directions.extend(direction.cpu().numpy())
            
    
    AUC, min_dist, avg_dist = compute_metrics(pred_hm, gt_gaze, image_size)
    if save_path is not None:
        output = {}
        if pretrained_dict['modality'] == 'attention':
            output['pred_att'] = pred_att
        output['pred_hm'] = pred_hm; output['gt_gaze'] = gt_gaze; output['paths'] = paths
        output['AUC'] = AUC; output['min_dist'] = min_dist; output['avg_dist'] = avg_dist; output['direction'] = directions
        with open(os.path.join(save_path, 'output_gazefollow.pkl'), 'wb') as fp:
            pickle.dump(output, fp)    
            
    final_AUC = torch.mean(torch.tensor(AUC))
    final_min_dist = torch.mean(torch.tensor(min_dist))
    final_avg_dist = torch.mean(torch.tensor(avg_dist))
    print("\tAUC:{:.4f}\tmin dist:{:.4f}\tavg dist:{:.4f}".format(
          final_AUC,
          final_min_dist,
          final_avg_dist))
    
    return final_AUC, final_min_dist, final_avg_dist


def compute_metrics(pred_hm, gt_gaze, image_size):
    
    AUC = []; min_dist = []; avg_dist = []
    # go through each data point and record AUC, min dist, avg dist
    for b_i in tqdm(range(len(gt_gaze))):
        # remove padding and recover valid ground truth points
        valid_gaze = gt_gaze[b_i]        
        valid_gaze = valid_gaze[valid_gaze != -1].view(-1,2)
        # AUC: area under curve of ROC
        pm = pred_hm[b_i]
        multi_hot = imutils.multi_hot_targets(gt_gaze[b_i], image_size[b_i])
        scaled_heatmap = cv2.resize(pm, (image_size[b_i][0].item(), image_size[b_i][1].item()))
        auc_score = evaluation.auc(scaled_heatmap, multi_hot)
        AUC.append(auc_score)
        # min distance: minimum among all possible pairs of <ground truth point, predicted point>
        pred_x, pred_y = evaluation.argmax_pts(pm)
        norm_p = [pred_x/float(output_resolution), pred_y/float(output_resolution)]
        all_distances = []
        for gaze in valid_gaze:
            all_distances.append(evaluation.L2_dist(gaze, norm_p))
        min_dist.append(min(all_distances))
        # average distance: distance between the predicted point and human average point
        mean_gt_gaze = torch.mean(valid_gaze, 0)
        avg_distance = evaluation.L2_dist(mean_gt_gaze, norm_p)
        avg_dist.append(avg_distance)
    
    return np.array(AUC), np.array(min_dist), np.array(avg_dist)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="gpu id")
    parser.add_argument("--model_weights", type=str, help="model weights")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--save_path", type=str, default=None, help="path to save model outputs")
    args = parser.parse_args()

    test(args.model_weights, None, args.batch_size, args.device, mode='pt', save_path=args.save_path)
