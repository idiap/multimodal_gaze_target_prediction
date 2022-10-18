### Modality Extraction
- Pose extraction is based on HRFormer (https://github.com/HRNet/HRFormer). You need to clone the repo and install the dependencies. We're using a top-down approach that first uses an independent pre-trained object detector to detect a person, and then a pose extraction model to detect the body keypoints from the person's crop.
  - We use a faster R-CNN as the object detector
    - Config file: HRFormer/pose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py
    - Checkpoint file: faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth (you need to download it from the github repo)
  - We use an HRNet for the pose extraction method
    - Config file: HRFormer/pose/configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py
    - Checkpoint file: hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth (you need to download it from the github repo)
  - But you can use other pre-trained models for both person detection and pose extraction without many changes to the code. Just download the corresponding checkpoints and use the corresponding config files.
- Depth extraction is based on MiDaS (https://github.com/isl-org/MiDaS). Again, you need to clone the repo and install any required dependencies.
  - We're using the large variant of MiDaS
    - Checkpoint file: pt_large-midas-2f21e586.pt (download from the repo)
- The structure of the input dataset is preserved when extracting modalities (e.g. /input_folder/train/show/image1.png will result in /output_folder/train/show/image1-modality.png)

Example code
- Pose
  - VideoAttentionTarget
  ```bash
  python extract_pose_videoattentiontarget.py --input_path '/home/database/VideoAttentionTarget/' --output_path '/home/projects/videoattentiontarget' --det_config '/home/projects/HRFormer/pose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py' --det_checkpoint '/home/projects/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' --pose_config '/home/projects/HRFormer/pose/configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py' --pose_checkpoint '/home/projects/weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
  ```
  - GazeFollow
  ```bash
  python extract_pose_gazefollow.py --input_path '/home/database/gazefollow_extended' --output_path '/home/projects/gazefollow' --det_config '/home/projects/HRFormer/pose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py' --det_checkpoint '/home/projects/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth' --pose_config '/home/projects/HRFormer/pose/configs/top_down/hrnet/coco/hrnet_w48_coco_256x192.py' --pose_checkpoint '/home/projects/weights/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
  ```
- Depth: needs to be ran from the MiDaS folder (or add the folder using `sys.path.append({path})`)
  - VideoAttentionTarget
  ```bash
  python extract_depth_videoattentiontarget.py -i '/home/database/VideoAttentionTarget/images' -o '/home/projects/videoattentiontarget' --no-optimize --model_weights '/home/projects/MiDaS/weights/dpt_large-midas-2f21e586.pt'
  ```
  - Gazefollow
  ```bash
  python extract_depth_gazefollow.py -i '/home/database/gazefollow_extended' -o '/home/projects/gazefollow' --no-optimize --model_weights '/home/projects/MiDaS/weights/dpt_large-midas-2f21e586.pt'
  ```