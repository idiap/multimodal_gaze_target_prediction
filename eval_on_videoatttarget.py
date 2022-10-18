# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn

from model import BaselineModel, AttentionModelCombined
from dataset import VideoAttTarget_image
from config import *
from transforms import _get_transform, _get_transform_modality
from utils import imutils, evaluation, misc

import argparse
import os
import numpy as np
import cv2
import warnings
from tqdm import tqdm
import pickle

warnings.simplefilter(action='ignore', category=FutureWarning)


def test(model_weights, val_loader, batch_size=48, device=0, mode='dict', save_path=None):

    # Define device
    device = torch.device('cuda', device)

    # Load model
    print("Constructing model")
    if mode=='pt':
        pretrained_dict = torch.load(model_weights)
    elif mode=='dict':
        pretrained_dict = model_weights
    
    if pretrained_dict['modality'] in ['image', 'depth', 'pose']:
        model = BaselineModel(pretrained_dict['backbone_name'], pretrained_dict['modality'], cone_mode=pretrained_dict['cone_mode'], pred_inout=pretrained_dict['pred_inout'])
    elif pretrained_dict['modality'] == 'attention':
        model = AttentionModelCombined(cone_mode=pretrained_dict['cone_mode'], pred_inout=pretrained_dict['pred_inout'])
    
    model.cuda().to(device)
    model.cuda().to(device)
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict['model'])
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)
    AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []; directions = []; pred_att = []
    gt_gaze = []; gt_hm = []; pred_hm = []; image_size = [] ; paths = []
    with torch.no_grad():
        for val_batch, (val_img, val_face, val_pose, val_depth, val_gaze_field, val_gt_direction, val_head_channel, val_gaze_heatmap, cont_gaze, gaze_inside, path) in \
        tqdm(enumerate(val_loader), total=len(val_loader)):
            
            val_images = val_img.cuda().to(device)
            val_faces = val_face.cuda().to(device)
            val_head_channels = val_head_channel.cuda().to(device)
            val_pose_maps = val_pose.cuda().to(device)
            val_depth_maps = val_depth.cuda().to(device)
            gt_hm.extend(val_gaze_heatmap)
            
            val_gaze_fields = val_gaze_field.cuda().to(device)
            val_gaze_heatmap = val_gaze_heatmap.cuda().to(device)
            
            if pretrained_dict['modality'] == 'image':
                model_input = val_images
            elif pretrained_dict['modality'] == 'pose':
                model_input = val_pose_maps
            elif pretrained_dict['modality'] == 'depth':
                model_input = val_depth_maps
            elif pretrained_dict['modality'] == 'attention':
                model_input = [val_images, val_depth_maps, val_pose_maps]
            
            if pretrained_dict['modality'] == 'attention':
                    val_gaze_heatmap_pred, val_direction, val_inout_pred, val_att = model(model_input, val_faces, val_gaze_fields, val_head_channels)
                    pred_att.extend(val_att.squeeze().cpu().numpy())
            else:
                val_gaze_heatmap_pred, val_direction, val_inout_pred = model(model_input, val_faces, val_gaze_fields, val_head_channels)
            val_gaze_heatmap_pred = val_gaze_heatmap_pred.squeeze(1)

            gt_gaze.extend(cont_gaze)
            pred_hm.extend(val_gaze_heatmap_pred.cpu().numpy())
            paths.extend(path)
            directions.extend(val_direction.cpu().numpy())

            # in vs out classification
            in_vs_out_groundtruth.extend(gaze_inside.float().numpy())
            in_vs_out_pred.extend(val_inout_pred.cpu().numpy())
            
    AUC, distance = compute_metrics(pred_hm, gt_hm, gt_gaze)
    if save_path is not None:
        output = {}
        if pretrained_dict['modality'] == 'attention':
            output['pred_att'] = pred_att
        output['pred_hm'] = pred_hm; output['gt_gaze'] = gt_gaze; output['paths'] = paths; output['direction'] = directions
        output['AUC'] = AUC; output['distance'] = distance; output['pred_inout'] = in_vs_out_pred; output['gt_inout'] = in_vs_out_groundtruth
        with open(os.path.join(save_path, 'output_videoatt.pkl'), 'wb') as fp:
            pickle.dump(output, fp)    
            
    final_AUC = torch.mean(torch.tensor(AUC))
    final_distance = torch.mean(torch.tensor(distance))
    final_ap = evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)
    print("\tAUC:{:.4f}\tmin dist:{:.4f}\tin vs out AP:{:.4f}".format(
          final_AUC,
          final_distance,
          final_ap))
    
    return final_AUC, final_distance, final_ap
            

def compute_metrics(pred_hm, gt_hm, gt_gaze):
    AUC = []; distance = [];
    # go through each data point and record AUC, min dist, avg dist
    inout = [gt_gaze[i].mean()==-1 for i in range(len(gt_gaze))]
    print(np.array(inout).sum())
    for b_i in tqdm(range(len(gt_hm))):
        if gt_gaze[b_i].mean()!=-1:
            multi_hot = gt_hm[b_i]
            multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
            multi_hot = misc.to_numpy(multi_hot)

            pm = pred_hm[b_i]
            scaled_heatmap = cv2.resize(pm, (output_resolution, output_resolution))
            auc_score = evaluation.auc(scaled_heatmap, multi_hot)
            AUC.append(auc_score)

            gaze_x, gaze_y = gt_gaze[b_i]
            # distance: L2 distance between ground truth and argmax point
            pred_x, pred_y = evaluation.argmax_pts(pm)
            norm_p = [pred_x/output_resolution, pred_y/output_resolution]
            dist_score = evaluation.L2_dist([gaze_x, gaze_y], norm_p).item()
            distance.append(dist_score)
    
    return np.array(AUC), np.array(distance)


if __name__ == "__main__":
          
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="gpu id")
    parser.add_argument("--model_weights", type=str, default='model_videoatttarget.pt', help="model weights")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--save_path", type=str, default=None, help="path to save model outputs")
    args = parser.parse_args()
    
    # Prepare data
    transform = _get_transform()
    transform_modality = _get_transform_modality()
    print("Loading Data")
    
    ## VideoAtt
    print('Dataset: VideoAttentionTarget')
    print()
    val_dataset = VideoAttTarget_image(videoattentiontarget_val_label, transform, transform_modality,
                                   skip_frame=0, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=10)
    
    test(args.model_weights, val_loader, args.batch_size, args.device, mode='pt', save_path=args.save_path)
