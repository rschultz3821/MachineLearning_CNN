# Fruit Classification using CNN
## Overview
This repository contains a supervised learning project that explores image classification for fruit recognition:

*A custom Convolutional Neural Network (CNN) trained from scratch on fruit images

### CNN Fruit Image Classification
#### Dataset: Fruit Classification Dataset (subset of 5 fruit types)

#### Goal: Classify fruit images into one of 5 categories using a CNN 

#### Key Steps:
#### Preprocessing:

Resize all images to 64x64

Apply random rotation (±10°) during training for augmentation

Normalize and convert images to tensors

#### Model: Custom FruitCNN built using PyTorch:

3 convolutional blocks with BatchNorm, ReLU, and downsampling

Adaptive average pooling and fully connected layers with dropout

#### Training:

Optimizer: Adam

Loss: CrossEntropyLoss

100 epochs

Batch size: 16

#### Evaluation:

Test accuracy printed after each epoch

Final accuracy reported after training

Confusion matrix generated and saved

#### Results:
Test Accuracy: ~0.94

![image](https://github.com/user-attachments/assets/19e79ae7-b310-448c-8789-a2588982cc5b)

The CNN achieved strong performance with ~94% accuracy on the fruit classification task. Augmentation and regularization (dropout) helped improve generalization. This shows CNNs are highly effective for image-based classification tasks, outperforming traditional models like SVC.











