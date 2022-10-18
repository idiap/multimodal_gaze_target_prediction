# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

import numpy as np

WIDTH, HEIGHT = 960, 720
def generate_data_field(eye_point, width=WIDTH, height=HEIGHT):
    """eye_point is (x, y) and between 0 and 1"""
    x_grid = np.array(range(width)).reshape([1, width]).repeat(height, axis=0)
    y_grid = np.array(range(height)).reshape([height, 1]).repeat(width, axis=1)
    grid = np.stack((x_grid, y_grid)).astype(np.float32)

    x, y = eye_point
    x, y = x * width, y * height

    grid -= np.array([x, y]).reshape([2, 1, 1]).astype(np.float32)
    grid[0] = grid[0] / width
    grid[1] = grid[1] / height
#     norm = np.sqrt(np.sum(grid ** 2, axis=0)).reshape([1, height, width])
#     # avoid zero norm
#     norm = np.maximum(norm, 0.1)
#     grid /= norm
    return grid


def generate_gaze_cone(gaze_field, normalized_direction, width=WIDTH, height=HEIGHT):
        
    gaze_field = np.ascontiguousarray(gaze_field.transpose([1, 2, 0]))
    gaze_field = gaze_field.reshape([-1, 2])
    gaze_field = np.matmul(gaze_field, normalized_direction.reshape([2, 1]))
    gaze_field_map = gaze_field.reshape([height, width, 1])
    gaze_field_map = np.ascontiguousarray(gaze_field_map.transpose([2, 0, 1]))
    
    gaze_field_map = gaze_field_map * (gaze_field_map > 0).astype(np.int)
    
    return gaze_field_map.squeeze()