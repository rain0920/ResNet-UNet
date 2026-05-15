# 项目描述 Project Description

这是一个基于 ResNet 骨干网络并结合 UNet++ 架构的深度学习图像分割项目，支持多种网络架构，包括 UNet、UNet++、DeepLabV3、SegNet 和 FCN。
A deep learning image segmentation project based on ResNet backbone combined with UNet++ architecture, supporting multiple network architectures including UNet, UNet++, DeepLabV3, SegNet, and FCN.

## 特征 Features

- **Multiple Network Architectures**: UNet, UNet++, DeepLabV3, SegNet, FCN
- **ResNet Backbone**: ResNet50 as encoder with support for pruning optimization
- **Various Upsampling Methods**: DUpsampling, Bilinear interpolation
- **Training & Evaluation**: Complete training pipeline with miou evaluation
- **Data Augmentation**: Support for image augmentation

## 项目结构 Project Structure

```
ResNet-UNet++/
├── nets/          # Network architectures
├── models/        # Trained models
├── utils/         # Utility functions
├── DataSet/       # Dataset
├── train.py       # Training script
├── dataloader.py  # Data loading
├── miou.py        # mIoU evaluation
└── predict.py     # Prediction script
```

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Pandas
- tqdm

## Usage

```bash
# Training
python train.py

# Evaluation
python miou.py

# Prediction
python predict.py
```

## 项目声明 Project Statement

本项目的作者及单位:
The author and affiliation of this project:

```bash
项目名称(Project Name):SegMaster
目者 (Author) : Yu Wang, Jilian Zhang
作者单位 (Affiliation):暨南大学网络空间安全学院(college of cyber Security,Jinan University)
```


