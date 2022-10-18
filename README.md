### Overview

This repo provides the training and testing code for our paper "A Modular Multimodal Architecture for Gaze Target Prediction: Application to Privacy-Sensitive Settings" published at the GAZE workshop at CVPR 2022.
[[paper]](https://openaccess.thecvf.com/content/CVPR2022W/GAZE/papers/Gupta_A_Modular_Multimodal_Architecture_for_Gaze_Target_Prediction_Application_to_CVPRW_2022_paper.pdf) [[video]](https://youtu.be/z-XSwLOpNzw)


### Setup

We use the GazeFollow and VideoAttentionTarget datasets for training and testing our models. Please download them at the following link provided by ejcgt/attention-target-detection: <br>
GazeFollow extended: [link](https://www.dropbox.com/s/3ejt9pm57ht2ed4/gazefollow_extended.zip?dl=0) <br>
VideoAttentionTarget: [link](https://www.dropbox.com/s/8ep3y1hd74wdjy5/videoattentiontarget.zip?dl=0)

Next, extract the pose and depth modalities for both datasets following the instructions in [modality_extraction.md](modality_extraction.md)

After, please update the paths to the datasets in the ```config.py``` file.

We use pytorch for our experiments. Use the provided environment file to create the conda environment for the experiments.
```
conda env create -f environment.yml
```


### Training

#### Training on GazeFollow
##### Step 1. Train the single modality models
```
python train_on_gazefollow.py --modality image --backbone_name efficientnet-b1 --log_dir <path>
python train_on_gazefollow.py --modality depth --backbone_name efficientnet-b0 --log_dir <path>
python train_on_gazefollow.py --modality pose --backbone_name efficientnet-b0 --log_dir <path>
```
The trained model weights will be saved in the specified ```log_dir```. 

##### Step 2. Initialize the weights for the attention model
```
python initialize_attention_model.py --image_weights <path> --depth_weights <path> --pose_weights <path> --attention_weights <path>
```
Provide the paths to the pretrained image, depth and pose models. The attention model with initialized weights will be saved in the path specified by the ```attention_weights``` argument.

##### Step 3. Train the attention model
```
python train_on_gazefollow.py --modality attention --init_weights <path> --log_dir <path>
```
Provide the path to the initialized attention model weights. The trained model weights will be saved in the specified ```log_dir```.


#### Training on VideoAttentionTarget

Set ```pred_inout=True``` in the ```config.py``` file.

##### Train the single modality models
```
python train_on_videoatttarget.py --modality image --init_weights <path> --backbone_name efficientnet-b1 --log_dir <path>
python train_on_videoatttarget.py --modality depth --init_weights <path> --backbone_name efficientnet-b0 --log_dir <path>
python train_on_videoatttarget.py --modality pose --init_weights <path> --backbone_name efficientnet-b0 --log_dir <path>
```
Provide the initial weights from training on GazeFollow. The trained model weights will be saved in the specified ```log_dir```.

##### Train the attention model
```
python train_on_videoatttarget.py --modality attention --init_weights <path> --log_dir <path>
```
Provide the initial weights from training on GazeFollow. The trained model weights will be saved in the specified ```log_dir```.


#### Training the privacy-sensitive models
Simply set ```privacy=True``` in the ```config.py``` file. Then follow the same steps as above to train the respective models.


### Testing

#### Testing on GazeFollow
```
python eval_on_gazefollow.py --model_weights <path> 
```
Provide the path to the model weights with the ```model_weights``` argument.

#### Testing on VideoAttentionTarget
```
python eval_on_videoatttarget.py --model_weights <path>
```
Provide the path to the model weights with the ```model_weights``` argument.


### Pre-trained models
Pre-trained human-centric module: [link](https://drive.switch.ch/index.php/s/5hDsBdP4OsLks5X) <br>
Pre-trained attention model on GazeFollow: [link](https://drive.switch.ch/index.php/s/fJVjWSJWQtoJeT3) <br>
Pre-trained attention model on VideoAttentionTarget: [link](https://drive.switch.ch/index.php/s/EjVQlvUDisvL1c4)


### Citation

If you use our code, please cite:
```bibtex
@inproceedings{gupta2022modular,
  title={A Modular Multimodal Architecture for Gaze Target Prediction: Application to Privacy-Sensitive Settings},
  author={Gupta, Anshul and Tafasca, Samy and Odobez, Jean-Marc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  pages={5041--5050},
  year={2022}
}
```

### References
Parts of the code have been adapted from ejcgt/attention-target-detection
