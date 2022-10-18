# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

# =============================================================================
# GazeFollow dataset dir config
# =============================================================================
gazefollow_data = "data/gazefollow"
gazefollow_pose = "data/gazefollow-pose"
gazefollow_depth = "data/gazefollow-depth"
gazefollow_train_label = "data/gazefollow/train_annotations_release.txt"
gazefollow_val_label = "data/gazefollow/test_annotations_release.txt"

# =============================================================================
# VideoAttTarget dataset dir config
# =============================================================================
videoattentiontarget_data = "data/videoatttarget/images"
videoattentiontarget_pose = "data/videoatttarget-pose/images"
videoattentiontarget_depth = "data/videoatttarget-depth/images"
videoattentiontarget_train_label = "data/videoatttarget/annotations/train"
videoattentiontarget_val_label = "data/videoatttarget/annotations/test"

# path to pretrained weights for human-centric branch
human_centric_weights = ''

# =============================================================================
# model config
# =============================================================================
input_resolution = 224
output_resolution = 64

cone_mode = 'early'    # {'late', 'early'} fusion of person information
modality_dropout = True    # only used for attention model
pred_inout = True    # {set True for VideoAttentionTarget}
privacy = False     # {set True to train/test privacy-sensitive model}

# pytorch amp to speed up training and reduce memory usage
use_amp = False