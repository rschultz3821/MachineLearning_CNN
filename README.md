# Fruit Classification using CNN & SVC
## Overview
This repository contains two supervised learning projects that explore image classification for fruit recognition:

*A custom Convolutional Neural Network (CNN) trained from scratch on fruit images

*A Support Vector Classifier (SVC) trained on features extracted using a pretrained ResNet-18 (ResNet-18 is a convolutional neural network that is 18 layers deep)

Both models are evaluated on the same dataset and compared using accuracy and confusion matrices.

### 1. CNN Fruit Image Classification
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

#### Output:
Test Accuracy: ~0.94

Confusion Matrix: FruitCNN_confusion_matrix.png

### 2. SVC with ResNet-18 Feature Extraction
#### Dataset: Same as above

#### Goal: Classify fruits using SVC trained on features from a pretrained ResNet-18

#### Key Steps:
#### Preprocessing:

Resize all images to 64x64

Convert to tensors (no augmentation)

#### Feature Extraction:

Load pretrained ResNet-18 from torchvision.models

Replace final FC layer with identity to extract features

Extract features for both train and test datasets

#### Model: SVC (RBF kernel)

#### Hyperparameter Tuning:

GridSearchCV

#### Evaluation:

Best C and gamma printed

Accuracy score on test set

Confusion matrix generated and saved

#### Output:
Best Parameters: C=25.75, gamma=0.001

Test Accuracy: ~0.87

Confusion Matrix: FruitSVC_confusion_matrix.png

#### Review
Both models performed well on the fruit classification task, with the CNN achieving a higher test accuracy of approximately 94%, thanks to its ability to learn features directly from images and benefit from data augmentation. The SVC model with ResNet-18 feature extraction achieved around 87% accuracy, offering a faster and more lightweight alternative ideal for limited compute environments. While the CNN is more accurate, the SVC approach is easier to train and leverages powerful pretrained features effectively.








