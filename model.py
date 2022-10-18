# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

from config import human_centric_weights


# returns gaze cone; resnet/efficientnet + prediction head
class HumanCentric(nn.Module):
    def __init__(self, backbone = 'resnet'):
        super(HumanCentric, self).__init__()
        
        self.backbone = backbone
        self.feature_dim = 512  # the dimension of the CNN feature to represent each frame
        # Build Network Base
        if backbone == 'resnet':
            self.base_head = models.resnet18(pretrained=True)
            self.base_head = nn.Sequential(*list(self.base_head.children())[:-1])
        elif backbone == 'efficientnet':
            self.base_head = models.efficientnet_b0(pretrained=True)
            self.base_head = nn.Sequential(*list(self.base_head.children())[:-1])
        else:
            assert False, 'Incorrect backbone, please choose from [resnet, efficientnet]'
        
        # Build Network Head
        num_outputs = 2
        self.num_outputs = num_outputs
        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.base_head(dummy_head)            
        self.head_new = nn.Sequential(
                        nn.Linear(dummy_head.size(1), self.feature_dim), 
                        nn.ReLU(inplace=True),
                        nn.Linear(self.feature_dim, num_outputs),
                        nn.Tanh()) 

    def forward(self, head, gaze_field):
        # Model output
        h = self.base_head(head).squeeze(dim=-1).squeeze(dim=-1) # Nx512   
        head_embedding = h.clone()
        
        direction = self.head_new(h) 
        # convert to unit vector
        normalized_direction = direction / direction.norm(dim=1).unsqueeze(1)
        
        # generate gaze field map
        batch_size, channel, height, width = gaze_field.size()
        gaze_field = gaze_field.permute([0, 2, 3, 1]).contiguous()
        gaze_field = gaze_field.view([batch_size, -1, self.num_outputs])
        gaze_field = torch.matmul(gaze_field, normalized_direction.view([batch_size, self.num_outputs, 1]))
        gaze_cone = gaze_field.view([batch_size, height, width, 1])
        gaze_cone = gaze_cone.permute([0, 3, 1, 2]).contiguous()

        gaze_cone = nn.ReLU()(gaze_cone)
    
        return gaze_cone, normalized_direction, head_embedding


# efficientnet followed by an FPN
class FeatureExtractor(nn.Module):
    
    def __init__(self, backbone_name):
        
        '''
        args:
        backbone_name: name of the backbone to be used; ex. 'efficientnet-b0'
        '''
        
        super(FeatureExtractor, self).__init__()
    
        self.backbone = EfficientNet.from_pretrained(backbone_name)
        if backbone_name=='efficientnet-b3':
            self.fpn = FeaturePyramidNetwork([32, 48, 136, 384], 64)
        elif backbone_name=='efficientnet-b2':
            self.fpn = FeaturePyramidNetwork([24, 48, 120, 352], 64)
        elif backbone_name=='efficientnet-b0' or backbone_name=='efficientnet-b1':
            self.fpn = FeaturePyramidNetwork([24, 40, 112, 320], 64)        
        
    def forward(self, x):
        
        features = self.backbone.extract_endpoints(x)
        
        # select features to use
        fpn_features = OrderedDict()
        fpn_features['reduction_2'] = features['reduction_2']
        fpn_features['reduction_3'] = features['reduction_3']
        fpn_features['reduction_4'] = features['reduction_4']
        fpn_features['reduction_5'] = features['reduction_5']
        
        # upsample features from efficientnet using an FPN to generate features at (H/4, W/4) resolution
        features = self.fpn(fpn_features)['reduction_2']
        
        return features


# simple prediction head that takes the features and gaze cone to regress the gaze target heatmap
class PredictionHead(nn.Module):
    
    def __init__(self, inchannels):
        super(PredictionHead, self).__init__()
        
        self.act = nn.ReLU()
        
        self.conv1 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn1 = nn.BatchNorm2d(inchannels)
        self.conv2 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn2 = nn.BatchNorm2d(inchannels)
        self.conv3 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn3 = nn.BatchNorm2d(inchannels)
        self.conv4 = nn.Conv2d(inchannels, inchannels, 3, padding=3, dilation=3)
        self.bn4 = nn.BatchNorm2d(inchannels)
        self.conv5 = nn.Conv2d(inchannels, inchannels//2, 3, padding=3, dilation=3)
        self.bn5 = nn.BatchNorm2d(inchannels//2)
        self.conv6 = nn.Conv2d(inchannels//2, inchannels//4, 3, padding=3, dilation=3)
        self.bn6 = nn.BatchNorm2d(inchannels//4)
        self.conv7 = nn.Conv2d(inchannels//4, 1, 1)

    def forward(self, x):
                
        # upsample the features to 64, 64
        x = nn.Upsample(size=(64,64), mode='bilinear', align_corners=False)(x)
        x = self.act(self.bn1(self.conv1(x)))
        
        # regress the heatmap
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.act(self.bn4(self.conv4(x)))
        x = self.act(self.bn5(self.conv5(x)))
        x = self.act(self.bn6(self.conv6(x)))
        x = self.conv7(x)
        
        return x
        
# compress modality spatially
class CompressModality(nn.Module):
    
    def __init__(self, in_channels):
        super(CompressModality, self).__init__()
        
        self.act = nn.GELU()
        
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(512)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act(x)
        x = nn.MaxPool2d(x.shape[2])(x)

        return x.squeeze(dim=-1).squeeze(dim=-1)
    

# predicts in vs out gaze; CompressModality + Linear
class InvsOut(nn.Module):
    
    def __init__(self, in_channels):
        
        '''
        args:
        in_channels: number of input channels
        '''
        
        super(InvsOut, self).__init__()
        self.compress_inout = CompressModality(in_channels)
        self.inout = nn.Sequential(nn.Linear(1024, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1),
                                   nn.Sigmoid())
    
    def forward(self, x, head_embedding):
        
        x = self.compress_inout(x)
        x = torch.cat([x, head_embedding], axis=1)
        x = self.inout(x)
        
        return x
    

# baseline model that takes a single modality and the gaze cone as input to predict a gaze target heatmap
class BaselineModel(nn.Module):
    
    def __init__(self, backbone_name, modality, cone_mode='early', pred_inout=False):
        
        '''
        args:
        backbone_name: name of the backbone to be used; ex. 'efficientnet-b0'
        cone_mode: early or late fusion of person information {'early', 'late'}
        pred_inout: predict an in vs out of frame gaze label
        '''
        
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor(backbone_name)    
        self.prediction_head = PredictionHead(64)
        self.human_centric = HumanCentric()
        # load weights
        state_dict = torch.load(human_centric_weights)['model_state_dict']
        self.human_centric.load_state_dict(state_dict, strict=False)

        # add additional channels
        self.cone_mode = cone_mode
        if cone_mode=='early':
            input_layer = self.feature_extractor.backbone._conv_stem.weight
            self.feature_extractor.backbone._conv_stem.weight = torch.nn.Parameter(torch.cat([input_layer.clone(), input_layer.clone()[:,0:2,:,:]], axis=1))
        elif cone_mode=='late':
            self.cat_conv = nn.Conv2d(66, 64, 3, padding=1)
            
        # drop additional channels
        if modality == 'depth':
            self.feature_extractor.backbone._conv_stem.weight = torch.nn.Parameter(self.feature_extractor.backbone._conv_stem.weight[:,0:-2,:,:])
        
        self.pred_inout = pred_inout
        if pred_inout:
            self.in_vs_out_head = InvsOut(64)
    
    def forward(self, img, face, gaze_field, head_mask):
        
        # dummy predictions
        batch_size = img.shape[0]
        in_vs_out = torch.zeros(batch_size).cuda()
        direction = torch.zeros(batch_size, 2).cuda()
        
        # get gaze cone
        gaze_cone, direction, head_embedding = self.human_centric(face, gaze_field)
                
        if self.cone_mode=='early':
            x = torch.cat([img, gaze_cone, head_mask], dim=1)
        else:
            x = img
        
        # extract the features
        x = self.feature_extractor(x)
        
        if self.cone_mode=='late':
            x = torch.cat([x, gaze_cone, head_mask], dim=1)
            x = self.cat_conv(x)
            
        # apply the prediction head to get the heatmap
        hm = self.prediction_head(x)
        
        # apply the in vs out head
        if self.pred_inout:
            in_vs_out = self.in_vs_out_head(x, head_embedding)
        
        return hm, direction, in_vs_out


# attention based model. multiple modalities processed separately. output feature maps are weighted and added using predicted attention weights to predict a gaze target heatmap
class AttentionModelCombined(nn.Module):
    
    def __init__(self, cone_mode='early', pred_inout=False):
        
        '''
        args:
        cone_mode: early or late fusion of person information {'early', 'late'}
        pred_inout: predict an in vs out of frame gaze label
        '''
        
        super(AttentionModelCombined, self).__init__()
        self.feature_extractor_image = FeatureExtractor('efficientnet-b1')
        self.feature_extractor_depth = FeatureExtractor('efficientnet-b0')
        self.feature_extractor_pose = FeatureExtractor('efficientnet-b0')
        
        num_modalities = 3
        
        self.bn_image = nn.BatchNorm2d(64)
        self.bn_depth = nn.BatchNorm2d(64)
        self.bn_pose = nn.BatchNorm2d(64)
        
        additional_channels = 0
        if cone_mode=='late':
            additional_channels = 2
        self.Wv_image = nn.Conv2d(64+additional_channels, 64, kernel_size=3, padding=1)
        self.Wv_depth = nn.Conv2d(64+additional_channels, 64, kernel_size=3, padding=1)
        self.Wv_pose = nn.Conv2d(64+additional_channels, 64, kernel_size=3, padding=1)
        
        self.compress_image = CompressModality(64)
        self.compress_depth = CompressModality(64)
        self.compress_pose = CompressModality(64)
        self.attention_layer = nn.Sequential(nn.Linear(512*num_modalities, num_modalities),
                                             nn.Softmax()
                                             )
        
        self.human_centric = HumanCentric()
        # load weights
        state_dict = torch.load(human_centric_weights)['model_state_dict']
        self.human_centric.load_state_dict(state_dict, strict=False)
            
        # add additional channels
        self.cone_mode = cone_mode
        if cone_mode=='early':
            input_layer = self.feature_extractor_image.backbone._conv_stem.weight
            self.feature_extractor_image.backbone._conv_stem.weight = torch.nn.Parameter(torch.cat([input_layer.clone(), input_layer.clone()[:,0:2,:,:]], axis=1))
            input_layer = self.feature_extractor_depth.backbone._conv_stem.weight
            self.feature_extractor_depth.backbone._conv_stem.weight = torch.nn.Parameter(torch.cat([input_layer.clone(), input_layer.clone()[:,0:2,:,:]], axis=1))
            input_layer = self.feature_extractor_pose.backbone._conv_stem.weight
            self.feature_extractor_pose.backbone._conv_stem.weight = torch.nn.Parameter(torch.cat([input_layer.clone(), input_layer.clone()[:,0:2,:,:]], axis=1))
        
        # drop additional channels
        self.feature_extractor_depth.backbone._conv_stem.weight = torch.nn.Parameter(self.feature_extractor_depth.backbone._conv_stem.weight[:,0:-2,:,:])
        
        self.prediction_head = PredictionHead(64)
        
        self.pred_inout = pred_inout
        if pred_inout:
            self.in_vs_out_head = InvsOut(64)
    
    def forward(self, x, face, gaze_field, head_mask):
        
        # dummy predictions
        batch_size = x[0].shape[0]
        in_vs_out = torch.zeros(batch_size).cuda()
        direction = torch.zeros(batch_size, 2).cuda()
                
        # get gaze cone
        gaze_cone, direction, head_embedding = self.human_centric(face, gaze_field)
        
        # extract the features
        if self.cone_mode=='early':
            x_image = torch.cat([x[0], gaze_cone, head_mask], dim=1)
            x_depth = torch.cat([x[1], gaze_cone, head_mask], dim=1)
            x_pose = torch.cat([x[2], gaze_cone, head_mask], dim=1)
        else:
            x_image = x[0]
            x_depth = x[1]
            x_pose = x[2]

        x_image = self.feature_extractor_image(x_image)
        x_image = self.bn_image(x_image)
        x_depth = self.feature_extractor_depth(x_depth)
        x_depth = self.bn_depth(x_depth)
        x_pose = self.feature_extractor_pose(x_pose)
        x_pose = self.bn_pose(x_pose)
        
        if self.cone_mode=='late':
            x_image = torch.cat([x_image, gaze_cone, head_mask], dim=1)
            x_depth = torch.cat([x_depth, gaze_cone, head_mask], dim=1)
            x_pose = torch.cat([x_pose, gaze_cone, head_mask], dim=1)
        
        # get the values
        v_image = self.Wv_image(x_image)
        v_depth = self.Wv_depth(x_depth)
        v_pose = self.Wv_pose(x_pose)
                
        # get attention weights
        att_image = self.compress_image(v_image)
        att_depth = self.compress_depth(v_depth)
        att_pose = self.compress_pose(v_pose)
        att = torch.cat([att_image, att_depth, att_pose], dim=1)
        att = self.attention_layer(att).unsqueeze(2).unsqueeze(3).unsqueeze(4)    # add extra dimensions for weighting in the next step

        # weight values
        v_image = v_image * att[:, 0]
        v_depth = v_depth * att[:, 1]
        v_pose = v_pose * att[:, 2]
        x = v_image + v_depth + v_pose
        
        # apply the prediction head
        hm = self.prediction_head(x)
        
        # apply the in vs out head
        if self.pred_inout:
            in_vs_out = self.in_vs_out_head(x, head_embedding)
        
        return hm, direction, in_vs_out, att