# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
from torchvision import transforms
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')

from model import BaselineModel, AttentionModelCombined
from dataset import VideoAttTarget_image
from config import *
from transforms import _get_transform, _get_transform_modality
from utils import imutils, evaluation
from eval_on_videoatttarget import test

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
import warnings
import wandb

# initialize wandb project
wandb.init(project="VideoAttentionTarget-gaze2022", config={}, entity="agupta")

# set seeds
np.random.seed(1)
torch.manual_seed(1)

warnings.simplefilter(action='ignore', category=FutureWarning)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default=None, help="initial weights")
parser.add_argument("--resume", action='store_true', help="resume training; requires init_weights")
parser.add_argument("--lr", type=float, default=2.5e-5, help="learning rate")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=1, help="eval every ___ epochs")
parser.add_argument("--save_every", type=int, default=5, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="logs", help="directory to save log files")
parser.add_argument("--skip_frame", type=int, default=3, help="number of frames to skip when sampling from the video (for training)")
parser.add_argument("--backbone_name", type=str, help="name of the efficientnet backbone {'efficientnet-b0', ..., 'efficientnet-b3'}")
parser.add_argument("--modality", type=str, help="input modality {'image', 'depth', 'pose'}")
args = parser.parse_args()

# log hyperparameters
wandb.config.update(args)
wandb.config.update({"cone_mode": cone_mode})
wandb.config.update({"pred_inout": pred_inout})
wandb.config.update({"privacy": privacy})


def train():
    transform = _get_transform()
    transform_modality = _get_transform_modality()

    # Prepare data
    print("Loading Data")    
    train_dataset = VideoAttTarget_image(videoattentiontarget_train_label, transform, transform_modality,  
                                         skip_frame=args.skip_frame, input_size=input_resolution, output_size=output_resolution, modality=args.modality)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=10)

    val_dataset = VideoAttTarget_image(videoattentiontarget_val_label, transform, transform_modality, 
                                       skip_frame=0, input_size=input_resolution, output_size=output_resolution, test=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=10)

    # Set up log dir
    logdir = os.path.join(args.log_dir,
                          datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
    os.makedirs(logdir)
    wandb.config.update({"log_dir": logdir}, allow_val_change=True)

    # Define device
    device = torch.device('cuda', args.device)
    
    # Load model
    print("Constructing model")
    if args.modality in ['image', 'depth', 'pose']:
        model = BaselineModel(args.backbone_name, args.modality, cone_mode=cone_mode, pred_inout=pred_inout)
    elif args.modality == 'attention':
        model = AttentionModelCombined(cone_mode=cone_mode, pred_inout=pred_inout)
    model.cuda().to(device)


    # Optimizer
    increased_lr_list = ['in_vs_out_head']
    if args.modality=='attention':
        decreased_lr_list = ['feature_extractor_image', 'feature_extractor_depth', 'feature_extractor_pose', 'human_centric']
    else:
        decreased_lr_list = []
    params_non_list = []
    params_list1 = []
    params_list2 = []
    for kv in model.named_parameters():
        flag = 0
        for lname in increased_lr_list:
            if lname in kv[0]:
                flag = 1
        for lname in decreased_lr_list:
            if lname in kv[0]:
                flag = 2
        if flag==1:
            params_list1.append(kv[1])
        elif flag==2:
            params_list2.append(kv[1])
        else:
            params_non_list.append(kv[1])
    optimizer = torch.optim.AdamW([{'params': params_non_list},
                                   {'params': params_list1, 'lr': args.lr*10},
                                   {'params': params_list2, 'lr': args.lr/10}], lr=args.lr)
    
    start_ep = 0
    if args.init_weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)
        pretrained_model_dict = pretrained_dict['model']      
        for k, v in model_dict.items():
            model_dict[k] = pretrained_model_dict.get(k, v)
        model.load_state_dict(model_dict)
        
    if args.resume:
        pretrained_opt_dict = pretrained_dict.get('optimizer', None)
        if pretrained_opt_dict is not None:
            optimizer.load_state_dict(pretrained_opt_dict)

        
    # Loss functions
    mse_loss = nn.MSELoss(reduce=False) # not reducing in order to ignore outside cases
    bcelogit_loss = nn.BCEWithLogitsLoss()

    step = 0
    loss_amp_factor = 100 # multiplied to the loss to prevent underflow
    dir_loss_factor = 0.1
    max_steps = len(train_loader)
    optimizer.zero_grad()
    
    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_AUC = 0; best_distance = 1; best_ap = 1
    print("Training in progress ...")
    with torch.cuda.amp.autocast(enabled=use_amp):
        for ep in range(start_ep, args.epochs):
            for batch, (img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, gaze_inside, dropped) in enumerate(train_loader):
                model.train(True)

                images = img.cuda().to(device)
                head_channels = head_channel.cuda().to(device)
                faces = face.cuda().to(device)
                pose_maps = pose.cuda().to(device)
                depth_maps = depth.cuda().to(device)
                gaze_fields = gaze_field.cuda().to(device)
                gaze_heatmap = gaze_heatmap.cuda().to(device)
                gt_direction = gt_direction.cuda().to(device)
                dropped = dropped.cuda().to(device)

                if args.modality == 'image':
                    model_input = images
                elif args.modality == 'pose':
                    model_input = pose_maps
                elif args.modality == 'depth':
                    model_input = depth_maps
                elif args.modality == 'attention':
                    model_input = [images, depth_maps, pose_maps]

                if args.modality == 'attention':
                        gaze_heatmap_pred, direction, inout_pred, att = model(model_input, faces, gaze_fields, head_channels)
                else:
                    gaze_heatmap_pred, direction, inout_pred = model(model_input, faces, gaze_fields, head_channels)
                gaze_heatmap_pred = gaze_heatmap_pred.squeeze(1)

                # Loss
                    # l2 loss computed only for inside case
                l2_loss = mse_loss(gaze_heatmap_pred, gaze_heatmap)*loss_amp_factor
                l2_loss = torch.mean(l2_loss, dim=1)
                l2_loss = torch.mean(l2_loss, dim=1)
                gaze_inside = gaze_inside.cuda(device).to(torch.float)
                l2_loss = torch.mul(l2_loss, gaze_inside) # zero out loss when it's out-of-frame gaze case
                l2_loss = torch.sum(l2_loss)/torch.sum(gaze_inside)

                    # cross entropy loss for in vs out
                Xent_loss = 0
                if pred_inout:
                    Xent_loss = bcelogit_loss(inout_pred.squeeze(), gaze_inside.squeeze())

                    # cosine loss on predicted direction
                dir_loss = 0
                dir_loss = direction * gt_direction
                dir_loss = 1 - dir_loss.sum(axis=1) 
                dir_loss = dir_loss * gaze_inside # zero out loss when it's out-of-frame gaze case
                dir_loss = dir_loss.sum()*dir_loss_factor

                    # attention loss on dropped modalities
                att_loss = 0
                if args.modality=='attention':
                    att_masked = att.squeeze() * dropped
                    att_loss = att_masked.sum()

                total_loss = l2_loss + Xent_loss + dir_loss + att_loss
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                step += 1

                if batch % args.print_every == 0:
                    print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (dir){:.4f} (att){:.4f} (Xent){:.4f}".format(ep, batch+1, max_steps, l2_loss, dir_loss, att_loss, Xent_loss))

                # log loss
                wandb.log({"loss": total_loss})
            
            if (ep % args.eval_every == 0):
                print('Validation in progress ...')
                model.train(False)
                checkpoint = {'model': model.state_dict(),
                              'backbone_name': args.backbone_name, 
                              'modality': args.modality,
                              'cone_mode': cone_mode,
                              'pred_inout': pred_inout}
                final_AUC, final_distance, final_ap = test(checkpoint, val_loader)

                # log results
                wandb.log({"AUC": final_AUC})
                wandb.log({"Dist": final_distance})
                wandb.log({"AP": final_ap})

            if (ep % args.save_every == 0):
                print('saving model ...')
                checkpoint = {'model': model.state_dict(), 'backbone_name': args.backbone_name, 'modality': args.modality}

                # save the model
                checkpoint = {'epoch': ep,
                              'optimizer': optimizer.state_dict(),
                              'model': model.state_dict(),
                              'backbone_name': args.backbone_name, 
                              'modality': args.modality,
                              'cone_mode': cone_mode,
                              'pred_inout': pred_inout}
                torch.save(checkpoint, os.path.join(logdir, 'epoch_'+str(ep)+'.pt'))
            

if __name__ == "__main__":
    train()
