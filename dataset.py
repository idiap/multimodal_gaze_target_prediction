# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import pandas as pd

import matplotlib as mpl

mpl.use('Agg')
from matplotlib import cm
import matplotlib.pyplot as plt
# from scipy.misc import imresize

import os
import glob
import csv
import cv2
import pickle

from utils import imutils, myutils, gazeutils
from config import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class GazeFollow(Dataset):
    def __init__(self, csv_path, transform, transform_modality, input_size=input_resolution, output_size=output_resolution,
                 test=False, modality='image', imshow=False):
        if test:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'meta1', 'meta2']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[['path', 'eye_x', 'eye_y', 'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max',
                    'bbox_y_max']].groupby(['path', 'eye_x'])
            self.keys = list(df.groups.keys())
            self.X_test = df
            self.length = len(self.keys)
        else:
            column_names = ['path', 'idx', 'body_bbox_x', 'body_bbox_y', 'body_bbox_w', 'body_bbox_h', 'eye_x', 'eye_y',
                            'gaze_x', 'gaze_y', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'inout', 'meta1', 'meta2']
            df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
            df = df[df['inout'] != -1]  # only use "in" or "out "gaze. (-1 is invalid, 0 is out gaze)
            df.reset_index(inplace=True)
            self.y_train = df[['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'eye_x', 'eye_y', 'gaze_x',
                               'gaze_y', 'inout']]
            self.X_train = df['path']
            self.length = len(df)        
            
        self.data_dir = gazefollow_data
        self.pose_dir = gazefollow_pose
        self.depth_dir = gazefollow_depth
        self.transform = transform
        self.transform_modality = transform_modality
        self.test = test
        self.modality = modality

        self.input_size = input_size
        self.output_size = output_size
        self.imshow = imshow

    def __getitem__(self, index):
                    
        if self.test:
            g = self.X_test.get_group(self.keys[index])
            cont_gaze = []
            for i, row in g.iterrows():
                path = row['path']
                x_min = row['bbox_x_min']
                y_min = row['bbox_y_min']
                x_max = row['bbox_x_max']
                y_max = row['bbox_y_max']
                eye_x = row['eye_x']
                eye_y = row['eye_y']
                gaze_x = row['gaze_x']
                gaze_y = row['gaze_y']
                cont_gaze.append([gaze_x, gaze_y])  # all ground truth gaze are stacked up
            for j in range(len(cont_gaze), 20):
                cont_gaze.append([-1, -1])  # pad dummy gaze to match size for batch processing
            cont_gaze = torch.FloatTensor(cont_gaze)
            gaze_inside = True # always consider test samples as inside
        else:
            path = self.X_train.iloc[index]
            x_min, y_min, x_max, y_max, eye_x, eye_y, gaze_x, gaze_y, inout = self.y_train.iloc[index]
            gaze_inside = bool(inout)        
        
        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)

        # read image
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')

        width, height = img.size
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])

        # read pose
        pose = Image.open(os.path.join(self.pose_dir, path[:-4]+'-pose.jpg'))
        
        # read depth
        depth = Image.open(os.path.join(self.depth_dir, path[:-3]+'png'))

        if self.imshow:
            img.save("origin_img.jpg")

        if self.test:
            imsize = torch.IntTensor([width, height])
            if privacy:
                img = Image.fromarray(np.uint8(np.zeros((height, width, 3))*255))
        else:
            ## data augmentation               
                        
            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([gaze_x * width, x_min, x_max])
                crop_y_min = np.min([gaze_y * height, y_min, y_max])
                crop_x_max = np.max([gaze_x * width, x_min, x_max])
                crop_y_max = np.max([gaze_y * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                pose = TF.crop(pose, crop_y_min, crop_x_min, crop_height, crop_width)
                depth = TF.crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)
                
                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                # if gaze_inside:
                gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)
                # else:
                #     gaze_x = -1; gaze_y = -1

                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                pose = pose.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                                
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        if cone_mode=='early':
            cone_resolution = input_resolution
        else:
            cone_resolution = input_resolution // 4
        
        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=cone_resolution, coordconv=False).unsqueeze(0)

        # Crop the face
        if privacy:
            face = pose.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        else:
            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # modality dropout
        height, width = np.int(height), np.int(width)
        num_modalities = 3
        dropped = np.zeros(num_modalities)
        if not self.test and self.modality=='attention':
            if modality_dropout:
                # keep one modality
                if privacy:
                    modality_idx = 1
                else:
                    modality_idx = 0
                m_keep = np.random.randint(modality_idx, num_modalities)

                if (np.random.rand() <= 0.2) and m_keep!=0:
                    img = Image.fromarray(np.uint8(np.random.rand(height, width, 3)*255))
                    dropped[0] = 1
                if (np.random.rand() <= 0.2) and m_keep!=1:
                    depth = Image.fromarray(np.uint8(np.random.rand(height, width)*255))
                    dropped[1] = 1
                if (np.random.rand() <= 0.2) and m_keep!=2:
                    pose = Image.fromarray(np.uint8(np.random.rand(height, width, 3)*255))
                    dropped[2] = 1

            if privacy:
                img = Image.fromarray(np.uint8(np.zeros((height, width, 3))))
                dropped[0] = 1
        
        # generate new gaze field (for human-centric branch)
        eye_point = np.array([eye_x, eye_y])
        gaze = np.array([gaze_x, gaze_y])
        gt_direction = np.array([-1.0, -1.0])
        if gaze_inside:
            gt_direction = gaze - eye_point            
            if gt_direction.mean()!=0:
                gt_direction = gt_direction / np.linalg.norm(gt_direction)
        
        gaze_field = gazeutils.generate_data_field(eye_point, width=cone_resolution, height=cone_resolution)
        # normalize
        norm = np.sqrt(np.sum(gaze_field ** 2, axis=0)).reshape([1, cone_resolution, cone_resolution])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        gaze_field /= norm
          
        if self.transform is not None:
            img = self.transform(img)
            face = self.transform(face)
            
            pose = self.transform_modality(pose)
            depth = self.transform_modality(depth)
            depth = depth / 65535    # depth maps are in 16 bit format

        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if self.test:  # aggregated heatmap
            num_valid = 0
            for gaze_x, gaze_y in cont_gaze:
                if gaze_x != -1:
                    num_valid += 1
                    gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                         3,
                                                         type='Gaussian')
            gaze_heatmap /= num_valid
        else:
            # if gaze_inside:
            gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3,
                                                 type='Gaussian')

        if self.imshow:
            fig = plt.figure(111)
            img = 255 - imutils.unnorm(img.numpy()) * 255
            img = np.clip(img, 0, 255)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.imshow(cv2.resize(gaze_heatmap, (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
            plt.imshow(cv2.resize(1 - head_channel.squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
            plt.savefig('viz_aug.png')

        if self.test:
            return img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, cont_gaze, imsize, path
        else:
            return img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, path, gaze_inside, dropped

    def __len__(self):
        return self.length


class VideoAttTarget_image(Dataset):
    def __init__(self, annotation_dir, transform, transform_modality, skip_frame, input_size=input_resolution, output_size=output_resolution,
                 test=False, modality='image', imshow=False):
        
        shows = glob.glob(os.path.join(annotation_dir, '*'))
        self.all_sequence_paths = []
        
        self.df = pd.DataFrame()
        for s in shows:
            sequence_annotations = glob.glob(os.path.join(s, '*', '*.txt'))
            for ann in sequence_annotations:
                parts = ann.split('/')
                show = parts[-3]
                clip = parts[-2]
                
                df_tmp = pd.read_csv(ann, header=None, index_col=False,
                         names=['path', 'xmin', 'ymin', 'xmax', 'ymax', 'gazex', 'gazey'])
                df_tmp['path'] = [os.path.join(show, clip, p) for p in df_tmp['path'].values]
                                
                # skip frames during training
                if not test and skip_frame>0:
                    indices = range(0, len(df_tmp), skip_frame)
                    df_tmp = df_tmp.iloc[indices]
                
                self.df = pd.concat([self.df, df_tmp])
                
        self.data_dir = videoattentiontarget_data
        self.pose_dir = videoattentiontarget_pose
        self.depth_dir = videoattentiontarget_depth
        self.transform = transform
        self.transform_modality = transform_modality
        self.input_size = input_size
        self.output_size = output_size
        self.test = test
        self.modality = modality
        self.imshow = imshow
        self.length = self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        path = row['path']
        x_min = row['xmin']
        y_min = row['ymin']
        x_max = row['xmax']
        y_max = row['ymax']
        gaze_x = row['gazex']
        gaze_y = row['gazey']

        if gaze_x == -1 and gaze_y == -1:
                gaze_inside = False
        else:
            if gaze_x < 0: # move gaze point that was sliglty outside the image back in
                gaze_x = 0
            if gaze_y < 0:
                gaze_y = 0
            gaze_inside = True

        # read image
        img = Image.open(os.path.join(self.data_dir, path))
        img = img.convert('RGB')
        width, height = img.size
        
        gaze_x = gaze_x / width
        gaze_y = gaze_y / height
        
        # read pose
        pose = Image.open(os.path.join(self.pose_dir, path[:-4]+'-pose.jpg'))
        
        # read depth
        depth = Image.open(os.path.join(self.depth_dir, path[:-3]+'png'))
                
        # generate gaze cone
        eye_x, eye_y = (x_min+x_max)/2.0, (0.65*y_min+0.35*y_max)    # approximate the location of the eyes
        eye_x, eye_y = eye_x/width, eye_y/height

        # expand face bbox a bit
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += k * abs(y_max - y_min)
        x_min, y_min, x_max, y_max = map(float, [x_min, y_min, x_max, y_max])
        
        if self.imshow:
            img.save("origin_img.jpg")
        
        if self.test:
            imsize = torch.IntTensor([width, height])
            if privacy:
                img = Image.fromarray(np.uint8(np.zeros((height, width, 3))*255))
        else:
            ## data augmentation

            # Jitter (expansion-only) bounding box size
            if np.random.random_sample() <= 0.5:
                k = np.random.random_sample() * 0.2
                x_min -= k * abs(x_max - x_min)
                y_min -= k * abs(y_max - y_min)
                x_max += k * abs(x_max - x_min)
                y_max += k * abs(y_max - y_min)

            x_min = np.clip(x_min, 0, width)
            x_max = np.clip(x_max, 0, width)
            y_min = np.clip(y_min, 0, height)
            y_max = np.clip(y_max, 0, height)    
            
            # Random Crop
            if np.random.random_sample() <= 0.5:
                # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
                crop_x_min = np.min([abs(gaze_x) * width, x_min, x_max])
                crop_y_min = np.min([abs(gaze_y) * height, y_min, y_max])
                crop_x_max = np.max([abs(gaze_x) * width, x_min, x_max])
                crop_y_max = np.max([abs(gaze_y) * height, y_min, y_max])

                # Randomly select a random top left corner
                if crop_x_min >= 0:
                    crop_x_min = np.random.uniform(0, crop_x_min)
                if crop_y_min >= 0:
                    crop_y_min = np.random.uniform(0, crop_y_min)

                # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
                crop_width_min = crop_x_max - crop_x_min
                crop_height_min = crop_y_max - crop_y_min
                crop_width_max = width - crop_x_min
                crop_height_max = height - crop_y_min
                # Randomly select a width and a height
                crop_width = np.random.uniform(crop_width_min, crop_width_max)
                crop_height = np.random.uniform(crop_height_min, crop_height_max)

                # Crop it
                img = TF.crop(img, crop_y_min, crop_x_min, crop_height, crop_width)
                pose = TF.crop(pose, crop_y_min, crop_x_min, crop_height, crop_width)
                depth = TF.crop(depth, crop_y_min, crop_x_min, crop_height, crop_width)
                
                # Record the crop's (x, y) offset
                offset_x, offset_y = crop_x_min, crop_y_min

                # convert coordinates into the cropped frame
                x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
                if gaze_inside:
                    gaze_x, gaze_y = (gaze_x * width - offset_x) / float(crop_width), \
                                 (gaze_y * height - offset_y) / float(crop_height)
                eye_x, eye_y = (eye_x * width - offset_x) / float(crop_width), \
                                 (eye_y * height - offset_y) / float(crop_height)
                # else:
                #     gaze_x = -1; gaze_y = -1

                width, height = crop_width, crop_height

            # Random flip
            if np.random.random_sample() <= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                pose = pose.transpose(Image.FLIP_LEFT_RIGHT)
                depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
                                
                x_max_2 = width - x_min
                x_min_2 = width - x_max
                x_max = x_max_2
                x_min = x_min_2
                gaze_x = 1 - gaze_x
                eye_x = 1 - eye_x

            # Random color change
            if np.random.random_sample() <= 0.5:
                img = TF.adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
                img = TF.adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        if cone_mode=='early':
            cone_resolution = input_resolution
        else:
            cone_resolution = input_resolution // 4
        
        head_channel = imutils.get_head_box_channel(x_min, y_min, x_max, y_max, width, height,
                                                    resolution=cone_resolution, coordconv=False).unsqueeze(0)

        # Crop the face
        if privacy:
            face = pose.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        else:
            face = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # modality dropout
        height, width = np.int(height), np.int(width)
        num_modalities = 3
        dropped = np.zeros(num_modalities)
        if not self.test and self.modality=='attention':
            if modality_dropout:
                # keep one modality
                if privacy:
                    modality_idx = 1
                else:
                    modality_idx = 0
                m_keep = np.random.randint(modality_idx, num_modalities)
                if (np.random.rand() <= 0.2) and m_keep!=0:
                    img = Image.fromarray(np.uint8(np.random.rand(height, width, 3)*255))
                    dropped[0] = 1
                if (np.random.rand() <= 0.2) and m_keep!=1:
                    depth = Image.fromarray(np.uint8(np.random.rand(height, width)*255))
                    dropped[1] = 1
                if (np.random.rand() <= 0.2) and m_keep!=2:
                    pose = Image.fromarray(np.uint8(np.random.rand(height, width, 3)*255))
                    dropped[2] = 1
                
            if privacy:
                img = Image.fromarray(np.uint8(np.zeros((height, width, 3))))
                dropped[0] = 1
        
        if self.imshow:
            img.save("img_aug.jpg")
            face.save('face_aug.jpg')

        if self.transform is not None:
            img = self.transform(img)
            pose = self.transform_modality(pose)
            depth = self.transform_modality(depth)
            depth = depth / 65535    # depth maps are in 16 bit format
            face = self.transform(face)
            
        # generate new gaze field (for human-centric branch)
        eye_point = np.array([eye_x, eye_y])
        gaze = np.array([gaze_x, gaze_y])
        gt_direction = np.array([-1.0, -1.0])
        if gaze_inside:
            gt_direction = gaze - eye_point
            if gt_direction.mean()!=0:
                gt_direction = gt_direction / np.linalg.norm(gt_direction)
        gaze_field = gazeutils.generate_data_field(eye_point, width=cone_resolution, height=cone_resolution)
        # normalize
        norm = np.sqrt(np.sum(gaze_field ** 2, axis=0)).reshape([1, cone_resolution, cone_resolution])
        # avoid zero norm
        norm = np.maximum(norm, 0.1)
        gaze_field /= norm
            
        # generate the heat map used for deconv prediction
        gaze_heatmap = torch.zeros(self.output_size, self.output_size)  # set the size of the output
        if gaze_x != -1:
                gaze_heatmap = imutils.draw_labelmap(gaze_heatmap, [gaze_x * self.output_size, gaze_y * self.output_size],
                                                 3, type='Gaussian')

        if gaze_inside:
            cont_gaze = [gaze_x, gaze_y]
        else:
            cont_gaze = [-1, -1]
        cont_gaze = torch.FloatTensor(cont_gaze)
        
        if self.imshow:
            fig = plt.figure(111)
            img = 255 - imutils.unnorm(img.numpy()) * 255
            img = np.clip(img, 0, 255)
            plt.imshow(np.transpose(img, (1, 2, 0)))
            plt.imshow(imresize(gaze_heatmap, (self.input_size, self.input_size)), cmap='jet', alpha=0.3)
            plt.imshow(imresize(1 - head_channel.squeeze(0), (self.input_size, self.input_size)), alpha=0.2)
            plt.savefig('viz_aug.png')

        if self.test:
            return img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, cont_gaze, gaze_inside, path
        else:
            return img, face, pose, depth, gaze_field, gt_direction, head_channel, gaze_heatmap, gaze_inside, dropped
        
    def __len__(self):
        return self.length    
