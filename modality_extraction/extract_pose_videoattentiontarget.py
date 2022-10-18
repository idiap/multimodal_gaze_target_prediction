# SPDX-FileCopyrightText: 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Samy Tafasca <samy.tafasca@idiap.ch>
# SPDX-License-Identifier: GPL-3.0

# IMPORTS
import os
import math
import json
import argparse 
from glob import glob

import torch

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmdet.apis import inference_detector, init_detector


limbs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (0, 17), (5, 6), (11, 12), (17, 12), (17, 11), (6, 8), (8, 10), (5, 7), (7, 9), 
         (12, 14), (14, 16), (11, 13), (13, 15)]

limb_colors = [(255, 0, 255), (255, 165, 165), (165, 165, 255), (0, 0, 255), (85, 85, 85), (0, 255, 165), (255, 255, 0), 
               (0, 255, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0), (0, 165, 255), (255, 0, 165), (165, 255, 0), 
               (165, 255, 165), (165, 165, 0), (165, 85, 85), (85, 0, 85)]

kp_colors = [(255, 0, 255), (165, 165, 255), (255, 165, 165), (0, 255, 165), (0, 0, 255), (255, 0, 165), (255, 255, 0), 
             (165, 255, 0), (255, 165, 0), (0, 255, 255), (0, 165, 255), (165, 85, 85), (165, 255, 165), (85, 0, 85), 
             (165, 165, 0), (0, 255, 0), (255, 0, 0), (255, 255, 255)]



def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for `person`)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results



def find_person_pose(image, head_bbox, poses):
    """
    Function using a heuristic to associate a head box to the closest matching pose given a list of poses.
    This is used to automatically detect the pose of a person given its head bounding box location.
    
    Args
        image: input image
        head_bbox: the coordinates of the head bounding box of the person of interest [x_min, y_min, x_max, y_max]
        poses: a list of poses where each pose is an nx3 array where n is the number of keypoints and 3 
        corresponds to (x, y) location and detection score.
        
    Output
        The id of the corresponding pose in the list `poses`, or -1 if the closest detected facial keypoints
        are not inside the head bounding box (e.g. when the person of interest was not detected by the pose model)
    """
    x_min, y_min, x_max, y_max = head_bbox
    head_cx = (x_min + x_max) / 2
    head_cy = (y_min + y_max) / 2
    head = np.array([head_cx, head_cy])
    
    faces = []
    for pose in poses:
        face_kp = pose[[0, 1, 2, 3, 4], :] # ids for the facial keypoints
        face_kp = face_kp[face_kp[:, 2] >= 0.3] # filter out non detected keypoints (to avoid biasing the average)
        if len(face_kp) == 0: # no facial keypoints detected
            face_cx, face_cy = 1e6, 1e6
        else:
            face_cx, face_cy = face_kp[:, :2].mean(axis=0)
        faces.append((face_cx, face_cy))
    faces = np.array(faces)
    dist = np.linalg.norm(faces - head, axis = 1)
    idx = np.argmin(dist)
    
    if (x_min <= faces[idx, 0] <= x_max) & (y_min <= faces[idx, 1] <= y_max):
        return idx

    return -1


def draw_pose(image, pose_results, blank=True, kp_thresh=0.3):
    """
    Utility function used to draw detected poses (keypoints and limbs) using a set of colors.
    Args
        image: input image in (HxWxC) format
        pose_results: a list of dicts of the form {'bbox': x, 'pose': y}. This should be the output of the 
        function `inference_top_down_pose_model`.
        blank: whether to draw on an empty canvas or overlay the pose on the original image.
        kp_thresh: the threshold for the keypoints (and their limbs) to keep in the visualization.
    Output
        (empty) image having the same dimension as the input with overlaid poses.
    """
    H, W, C = image.shape
    
    if blank:
        canvas = np.zeros((H, W, 3), dtype=float)
    else:
        canvas = np.copy(image)
        
    
    for result in pose_results:
        # Get Pose and Bbox
        bbox = result['bbox']
        pose = result['keypoints']
        
        # Compute radius and limb width
        bbox_w = np.abs(bbox[0] - bbox[2])
        radius = int(round(np.sqrt(bbox_w + 1) / 3))
        stickwidth = int(round(np.sqrt(bbox_w + 1) / 3))
                        
        # Compute shoulders midpoint
        keypoint17 = (pose[5] + pose[6]) / 2
        pose = np.vstack([pose, keypoint17])
        
        # Find Key Points to Skip (either because score is lower than thresh or because they're not connected to any detected limbs)
        kp_to_draw = np.array([False] * len(pose)) # 18
        for idx, limb in enumerate(limbs):
            p1, p2 = limb
            if (pose[p1, 2] >= kp_thresh) and (pose[p2, 2] >= kp_thresh):
                kp_to_draw[p1] = True
                kp_to_draw[p2] = True
        
        # Draw Key Points
        for p in range(len(pose)):
            if ~kp_to_draw[p]: 
                continue
            x = int(pose[p, 0])
            y = int(pose[p, 1])
            canvas = cv2.circle(canvas, (x, y), radius, kp_colors[p], thickness=-1)
        
        # Draw Limbs
        for idx, limb in enumerate(limbs):
            p1, p2 = limb
            if (pose[p1, 2] < kp_thresh) or (pose[p2, 2] < kp_thresh): 
                continue

            x1, y1 = int(pose[p1, 0]), int(pose[p1, 1])
            x2, y2 = int(pose[p2, 0]), int(pose[p2, 1])
            cur_canvas = canvas.copy()
            
            mean_x = (x1 + x2) / 2
            mean_y = (y1 + y2) / 2
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            angle = int(math.degrees(math.atan2(y1 - y2, x1 - x2)))
            
            polygon = cv2.ellipse2Poly((int(mean_x), int(mean_y)), (int(length / 2), stickwidth), angle, 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, limb_colors[idx])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)   
                        
    return canvas
        
    
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_path', help="Path to the folder containing the dataset.")
parser.add_argument('-o', '--output_path', help='Path to the output folder where to store the result.')
parser.add_argument('-dcf', '--det_config', help='Path to the detection model config file.')
parser.add_argument('-dck', '--det_checkpoint', help='Path to the detection model checkpoint file.')
parser.add_argument('-pcf', '--pose_config', help='Path to the pose model config file.')
parser.add_argument('-pck', '--pose_checkpoint', help='Path to the pose model checkpoint file.')
parser.add_argument('-d', '--device', default='cpu', help='Device type to use [cpu, cuda:0].')
parser.add_argument('-bth', '--bbox_thr', default=0.3, type=float, help='Person bounding box score threshold.')
parser.add_argument('-kth', '--kpt_thr', default=0.3, type=float, help='Pose keypoint score threshold.')
parser.add_argument("--eye_thr", default=0.6, help="Threshold level for a detected eye keypoint to be valid")
parser.add_argument("--min_kpt_pose", default=6, help="Minimum number of keypoints above confidence threshold for a pose to be valid")

args = parser.parse_args()
        


def main():
    """
    The function extracts 4 things
        1. pose.csv: a csv file storing the detections (ie. eyes and pose box) in tabular format 
        [image_path, person_id, left_eye_x, left_eye_y, right_eye_x, right_eye_y, pose_min_x, pose_min_y, pose_max_x, pose_max_y]
        2. {image-id}-pose.jpg: drawn detection poses on an empty canvas. The image-id is the same as the original dataset
        3. {image-id}-pose.json: file containing the detections returned by the pose model (this is the raw information to avoid having to re-run the model)
        4. {image-id}-{person-id}-pose.jpg: the pose of the annotated person (ie. gaze and head box) if it can be determined, otherwise, an empty image.
        
    The files are stored in a similar structure compared to the original VideoAttentionTarget (ie. images/show/clip/{files}).
    """
    
    print(f'Using device {args.device}')
    det_cat_id = 0 # category of person for bounding box detection model

    # Load Model
    det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)

    # Create Logging File
    logger = open(os.path.join(args.output_path, 'logs.txt'), 'a+')
    
    # Create Pose Bbox + Eyes location CSV
    f_out = open(os.path.join(args.output_path, 'pose.csv'), 'a+')

    # Annotation Files Column Names
    columns = ['path', 'bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max', 'gaze_x', 'gaze_y']

    folders = glob(os.path.join(args.input_path, 'images/*/*'))
    for folder in folders:

        # Decompose folder path
        show, clip = folder.split('/')[-2:] 

        # Create Output folder if doesn't exist
        if not os.path.isdir(os.path.join(args.output_path, show, clip)):
            msg = f'Creating folder {os.path.join(args.output_path, show, clip)} ...\n'
            print(msg)
            logger.write(msg)
            os.makedirs(os.path.join(args.output_path, show, clip), exist_ok=True) 

        # Read Annotations (either in train or test sub-folders)
        if os.path.isdir(os.path.join(args.input_path, 'annotations/train', show, clip)):
            person_paths = glob(os.path.join(args.input_path, 'annotations/train', show, clip, '*.txt'))
        else:
            person_paths = glob(os.path.join(args.input_path, 'annotations/test', show, clip, '*.txt'))

        person2df = {}
        for person_path in person_paths:
            identifier = os.path.basename(person_path).split('.')[0] # /path/to/annot/s01.txt --> s01
            df = pd.read_csv(person_path, names=columns)
            person2df[identifier] = df
        nb_persons = len(person2df)      

        # Iterate over images
        image_files = glob(os.path.join(folder, '*.jpg'))
        for image_file in image_files:
            image_name = image_file.split('/')[-1]
            image = Image.open(image_file)
            image = np.array(image.convert('RGB'))
            height, width, channels = image.shape

            # Output Names
            base, ext = image_name.split('.')
            img_pose_fname = base + f'-pose.' + ext
            json_fname = base + f'-pose.json'

            # Extract Pose
            try:
                # Find Bounding Boxes
                mmdet_results = inference_detector(det_model, image)
                # Filter out Bounding Boxes that are not Person Class
                person_results = process_mmdet_results(mmdet_results, det_cat_id)
                # Detect Pose
                results, _ = inference_top_down_pose_model(pose_model, image, person_results, bbox_thr=args.bbox_thr, format='xyxy', dataset='TopDownCocoDataset', return_heatmap=False, outputs=None)
            except Exception as e:
                msg = f"{type(e).__name__} was raised: {e}. Path: {image_file}"
                print(msg)
                logger.write(msg)

                # Save empty image pose
                img_canvas = np.zeros(image.shape, dtype=np.uint8)
                image_pose = Image.fromarray(img_canvas, mode='RGB')
                image_pose.save(os.path.join(args.output_path, show, clip, img_pose_fname))

                # Save empty subject poses
                subject_canvas = np.zeros((128, 64, 3), dtype=np.uint8)
                subject_pose = Image.fromarray(subject_canvas, mode='RGB')
                for identifier, df in person2df.items():
                    subject_pose_fname = base + f'-{identifier}-pose.' + ext
                    subject_pose.save(os.path.join(args.output_path, show, clip, subject_pose_fname))

                # Record Eye Position and Pose Bbox
                f_out.write(','.join(list(map(str, [image_file, -1, -1, -1, -1, -1, -1, -1, -1, -1]))) + '\n')

                # Save json of empty poses
                output_json = []
                with open(os.path.join(args.output_path, show, clip, json_fname), 'w') as fp:
                    json.dump(output_json, fp)
                continue
                
            # Discard Entries where Pose has less than n keypoints with confidence above kpt_thr
            if len(results) > 0:
                for j, res in enumerate(results):
                    pose = res['keypoints']
                    if (pose[:, 2] >= args.kpt_thr).sum() < args.min_kpt_pose:
                        del results[j]

            # If no pose is detected, skip this iteration
            if len(results) == 0:
                msg = f'No detected poses in the image (Path: {image_file}).\n'
                print(msg)
                logger.write(msg)

                # Save empty image pose
                img_canvas = np.zeros(image.shape, dtype=np.uint8)
                image_pose = Image.fromarray(img_canvas, mode='RGB')
                image_pose.save(os.path.join(args.output_path, show, clip, img_pose_fname))

                # Save empty subject poses
                subject_canvas = np.zeros((128, 64, 3), dtype=np.uint8)
                subject_pose = Image.fromarray(subject_canvas, mode='RGB')
                for identifier, df in person2df.items():
                    subject_pose_fname = base + f'-{identifier}-pose.' + ext
                    subject_pose.save(os.path.join(args.output_path, show, clip, subject_pose_fname))

                # Record Eye Position and Pose Bbox
                f_out.write(','.join(list(map(str, [image_file, -1, -1, -1, -1, -1, -1, -1, -1, -1]))) + '\n')

                # Save json of poses
                output_json = []
                with open(os.path.join(args.output_path, show, clip, json_fname), 'w') as fp:
                    json.dump(output_json, fp)

                continue

            # Draw & Save Image Pose
            image_pose = draw_pose(image, results, blank=True, kp_thresh=args.kpt_thr)
            image_pose = Image.fromarray(np.uint8(image_pose), mode='RGB')
            image_pose.save(os.path.join(args.output_path, show, clip, img_pose_fname))

            # Save JSON of Predicted Poses
            output_json = [{'bbox': res['bbox'].tolist(), 'keypoints': res['keypoints'].tolist()} for res in results]
            with open(os.path.join(args.output_path, show, clip, json_fname), 'w') as fp:
                json.dump(output_json, fp)


            # Iterate over people in image
            for identifier, annot_df in person2df.items():
                # Subject Pose File Name
                subject_pose_fname = base + f'-{identifier}-pose.' + ext

                # If Subject is not in this image (some people appear in some frames of a clip but not others)
                if not annot_df.path.str.contains(image_name).any():
                    continue

                # Retrieve head bounding box coordinates
                row = annot_df.loc[annot_df.path == image_name, ['bbox_x_min', 'bbox_y_min', 'bbox_x_max', 'bbox_y_max']].iloc[0]
                head_x_min = row['bbox_x_min']
                head_y_min = row['bbox_y_min']
                head_x_max = row['bbox_x_max']
                head_y_max = row['bbox_y_max']
                head_bbox = (head_x_min, head_y_min, head_x_max, head_y_max)

                # Find target person's pose
                poses = [res['keypoints'] for res in results]
                bboxes = [res['bbox'] for res in results]
                idx = find_person_pose(image, head_bbox, poses)

                if idx == -1:
                    msg = f'Detected head keypoints are not inside subject head bbox (ID: {identifier} | Path: {image_file}).\n'
                    print(msg)
                    logger.write(msg)

                    # Save empty subject poses
                    subject_canvas = np.zeros((128, 64, 3), dtype=np.uint8)
                    subject_pose = Image.fromarray(subject_canvas, mode='RGB')
                    subject_pose.save(os.path.join(args.output_path, show, clip, subject_pose_fname))

                    # Record Eye Position and Pose Bbox of Person
                    f_out.write(','.join(list(map(str, [image_file, identifier, -1, -1, -1, -1, -1, -1, -1, -1]))))
                    f_out.write('\n')
                    continue

                pose = poses[idx]
                bbox = bboxes[idx]

                # Compute Pose Bounding Box (sometimes keypoint locations are slightly outside the person bbox detected)        
                pose_min_x = max(min(bbox[0], pose[:, 0].min() - 5), 0)
                pose_min_y = max(min(bbox[1], pose[:, 1].min() - 5), 0)
                pose_max_x = min(max(bbox[2], pose[:, 0].max() + 5), width)
                pose_max_y = min(max(bbox[3], pose[:, 1].max() + 5), height)

                # Detect the Eyes
                if pose[1, 2] > args.eye_thr: # left eye
                    left_eye_x = pose[1, 0]
                    left_eye_y = pose[1, 1]
                else:
                    left_eye_x, left_eye_y = -1, -1

                if pose[2, 2] > args.eye_thr: # right eye
                    right_eye_x = pose[2, 0]
                    right_eye_y = pose[2, 1]
                else:
                    right_eye_x, right_eye_y = -1, -1

                # Record Eye Position and Pose Bbox
                f_out.write(','.join(list(map(str, [image_file, identifier, left_eye_x, left_eye_y, right_eye_x, 
                                                    right_eye_y, pose_min_x, pose_min_y, pose_max_x, pose_max_y]))) + '\n')

                # Draw subject pose on blank canvas
                subject_pose = draw_pose(image, [results[idx]], blank=True, kp_thresh=args.kpt_thr)

                # Crop and Save Subject Pose
                subject_pose = subject_pose[int(pose_min_y): int(pose_max_y), int(pose_min_x): int(pose_max_x), :]
                subject_pose = Image.fromarray(np.uint8(subject_pose), mode='RGB')
                subject_pose.save(os.path.join(args.output_path, show, clip, subject_pose_fname))
    
        break
        
    # Close Files    
    logger.close()
    f_out.close()
    


if __name__ == "__main__":
    main()