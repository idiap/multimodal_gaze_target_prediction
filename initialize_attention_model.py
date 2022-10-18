# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import argparse
import torch
from model import AttentionModelCombined


parser = argparse.ArgumentParser()
parser.add_argument("--image_weights", type=str, help="input: path to weights of trained image model")
parser.add_argument("--depth_weights", type=str, help="input: path to weights of trained depth model")
parser.add_argument("--pose_weights", type=str, help="input: path to weights of trained pose model")
parser.add_argument("--attention_weights", type=str, help="output: path to weights of initialized attention model")
args = parser.parse_args()

# load weights
image_weights = torch.load(args.image_weights)
depth_weights = torch.load(args.depth_weights)
pose_weights = torch.load(args.pose_weights)

if image_weights['cone_mode']==depth_weights['cone_mode'] and image_weights['cone_mode']==pose_weights['cone_mode']:
    cone_mode = image_weights['cone_mode']
else:
    print('cone mode {early, late} is not the same for all input weights!')
    exit()

if image_weights['pred_inout']==depth_weights['pred_inout'] and image_weights['pred_inout']==pose_weights['pred_inout']:
    pred_inout = image_weights['pred_inout']
else:
    print('pred inout {True, False} is not the same for all input weights!')
    exit()

# initialize the attention model
attention_model = AttentionModelCombined(cone_mode=cone_mode, pred_inout=pred_inout)
model_dict = attention_model.state_dict()
for k in model_dict.keys():
    k_suffix = '.'.join(k.split('.')[1:])
    
    if 'feature_extractor_image' in k:
        model_dict[k] = image_weights['model']['feature_extractor.'+k_suffix]
    
    if 'feature_extractor_depth' in k:
        model_dict[k] = depth_weights['model']['feature_extractor.'+k_suffix]
    
    if 'feature_extractor_pose' in k:
        model_dict[k] = pose_weights['model']['feature_extractor.'+k_suffix]
    
    if 'human_centric' in k:
        model_dict[k] = image_weights['model'][k]

# save the model
checkpoint = {'model': model_dict,
              'modality': 'attention',
              'cone_mode': cone_mode,
              'pred_inout': pred_inout}
torch.save(checkpoint, args.attention_weights)