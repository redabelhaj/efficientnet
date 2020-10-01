## Introduction

This project was an academic project at Ecole Polytechnique made under the supervision of Maks Ovsjanikov, by Reda Belhaj-Soullami and Mohamed Amine Rabhi.
For further information, the project report is available in the report folder


## Project outline

In this project, we investigate a compound scaling method for deep convolutional
neural networks called EfficientNet. To demonstrate the relevance of compound
scaling, we apply this scaling method to a ResNet baseline network on the
CIFAR-10 dataset. In a second part we use the EfficientNet method to produce
a backbone network for a R-CNN network, for the task of object detection on a
Kaggle dataset.

## Requirements

PyTorch, NumPy, matplotlib, tqdm, PIL, cv2, torchsummary, os, torchvision

## Usage

 Please refer to the Jupyter Notebooks for examples : 

 - scale_gs.ipynb uses the grid search method to find suitable compound scaling parameters as in the EfficientNet article.

 - backbone_africa.ipynb loads the Kaggle "African Wildlife" dataset and trains a classifier based on a ResNet architecture.

 - rcnn_africa.ipynb uses this classifier to build a R-CNN based detector.

 - The scaling_results folder contains a script for plotting the scaling results we obtained, that are stored in an Excel sheet.

 - The report folder contains the project report.

