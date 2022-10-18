# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn

from model import AttentionModelCombined, BaselineModel
from dataset import GazeFollow
from config import *
from eval_on_gazefollow import test
from transforms import _get_transform, _get_transform_modality

import argparse
import os
from datetime import datetime
import shutil
import numpy as np
import warnings
import wandb

# initialize wandb project
wandb.init(project="GazeFollow-gaze2022", config={}, entity="agupta")

# set seeds
np.random.seed(1)
torch.manual_seed(1)

warnings.simplefilter(action='ignore', category=FutureWarning)


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--init_weights", type=str, default=None, help="initial weights")
parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=48, help="batch size")
parser.add_argument("--epochs", type=int, default=40, help="number of epochs")
parser.add_argument("--print_every", type=int, default=100, help="print every ___ iterations")
parser.add_argument("--eval_every", type=int, default=5, help="eval every ___ epochs")
parser.add_argument("--save_every", type=int, default=5, help="save every ___ epochs")
parser.add_argument("--log_dir", type=str, default="logs", help="directory to save log files")
parser.add_argument("--backbone_name", type=str, default=None, help="name of the efficientnet backbone {'efficientnet-b0', ..., 'efficientnet-b3'}")
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
    
    # save training config to WnB
    
    # Prepare data
    print("Loading Data")    
    train_dataset = GazeFollow(gazefollow_train_label, transform, transform_modality, input_size=input_resolution, output_size=output_resolution, modality=args.modality)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=10)

    val_dataset = GazeFollow(gazefollow_val_label, transform, transform_modality, input_size=input_resolution, output_size=output_resolution, test=True)
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
    if args.modality == 'attention':
        model = AttentionModelCombined(cone_mode=cone_mode, pred_inout=pred_inout)
    elif args.modality in ['image', 'depth', 'pose']:
        model = BaselineModel(args.backbone_name, args.modality, cone_mode=cone_mode, pred_inout=pred_inout)
    model.cuda().to(device)
    
    # Optimizer
    if args.modality == 'attention':
        reduced_lr_list = ['feature_extractor_image.backbone', 'feature_extractor_depth.backbone', 'feature_extractor_pose.backbone', 'human_centric.backbone']
    else:
        reduced_lr_list = []
    params_non_backbone = []
    params_backbone = []
    for kv in model.named_parameters():
        flag = 0
        for lname in reduced_lr_list:
            if lname in kv[0]:
                flag = 1
        if flag:
            params_backbone.append(kv[1])
        else:
            params_non_backbone.append(kv[1])
    optimizer = torch.optim.AdamW([{'params': params_non_backbone},
                                   {'params': params_backbone, 'lr': args.lr/10}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # load pretrained weights
    if args.init_weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.init_weights)
        pretrained_model_dict = pretrained_dict['model']      
        model_dict.update(pretrained_model_dict)
        model.load_state_dict(model_dict)
        
        pretrained_opt_dict = pretrained_dict.get('optimizer', None)
        if pretrained_opt_dict is not None:
            optimizer.load_state_dict(pretrained_opt_dict)
    start_ep = 0

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
    
    best_AUC = 0; best_min_dist = 1; best_avg_dist = 1
    print("Training in progress ...")
    with torch.cuda.amp.autocast(enabled=use_amp):
        for ep in range(start_ep, args.epochs):
            for batch, (img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, name, gaze_inside, dropped) in enumerate(train_loader):
                model.train(True)        

                images = img.cuda().to(device)
                faces = face.cuda().to(device)
                head_channels = head_channel.cuda().to(device)
                gaze_fields = gaze_field.cuda().to(device)
                depth_maps = depth.cuda().to(device)
                pose_maps = pose.cuda().to(device)
                gaze_heatmap = gaze_heatmap.cuda().to(device)
                gt_direction = gt_direction.cuda().to(device)
                dropped = dropped.cuda().to(device)

                # choose input modality
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
                    # l2 loss for predicted heatmap
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
                # NOTE: Xent_loss is used to train the main model.
                #       No Xent_loss is used to get SOTA on GazeFollow benchmark.

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                if batch % args.print_every == 0:
                    print("Epoch:{:04d}\tstep:{:06d}/{:06d}\ttraining loss: (l2){:.4f} (Xent){:.4f} (dir){:.4f} (att){:.4f}".format(ep, batch+1, max_steps, l2_loss, Xent_loss, dir_loss, att_loss))

                # log loss
                wandb.log({"loss": total_loss})

            if (ep > 0 and ep % args.eval_every == 0):
                print('Validation in progress ...')
                model.train(False)
                checkpoint = {'model': model.state_dict(),
                              'backbone_name': args.backbone_name, 
                              'modality': args.modality,
                              'cone_mode': cone_mode,
                              'pred_inout': pred_inout}
                final_AUC, final_min_dist, final_avg_dist = test(checkpoint, val_loader)

                # log results
                wandb.log({"AUC": final_AUC})
                wandb.log({"Min Dist": final_min_dist})
                wandb.log({"Avg Dist": final_avg_dist})
            
            if (ep > 0 and ep % args.save_every == 0):
                print('saving model ...')                
                # save the model
                checkpoint = {'epoch': ep,
                              'optimizer': optimizer.state_dict(),
                              'model': model.state_dict(),
                              'backbone_name': args.backbone_name, 
                              'modality': args.modality,
                              'cone_mode': cone_mode,
                              'pred_inout': pred_inout}
                torch.save(checkpoint, os.path.join(logdir, 'epoch_'+str(ep)+'.pt'))
            
            scheduler.step()

if __name__ == "__main__":
    train()
