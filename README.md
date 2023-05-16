# Introduction
Adversarial Attacks on 3D Point Clouds

# Requirements
- A computer running on Linux
- NVIDIA GPU and NCCL
- Python 3.6 or higher version
- Pytorch 1.1 or higher version

# Usage
## Data preparing
Download alignment ModelNet [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`

## Model tranining
```markdown
e.g., pointnet without normal features
python train_classification.py --model pointnet_cls --log_dir pointnet_cls
```
## Shapley Value 
- Save the center point of the regions of the point cloud
```markdown
python save_fps.py 
```
- Calculate the Shapley Value of each region of the point cloud
```markdown
python shapley_value.py 
```

## Attack
```markdown
python AL-Adv.py 
```
