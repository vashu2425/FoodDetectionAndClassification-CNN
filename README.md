# Food Image Classification with EfficientNetB0

This project involves training a deep learning model using the EfficientNetB0 architecture for classifying food images into 11 categories. The model is fine-tuned using the Food11 image dataset, and includes various techniques like data augmentation, mixed-precision training, and class weighting to optimize performance.

## Overview

The objective is to build an image classification model to categorize food images based on their features. The dataset used for training, validation, and testing is the Food11 dataset, consisting of 11 different food categories. We use **EfficientNetB0** as the base model and fine-tune it with additional layers for classification.

## Key Features

- **Data Augmentation**: Applied to the training set for better generalization.
- **EfficientNetB0**: Pre-trained model from ImageNet used as a base for transfer learning.
- **Class Weighting**: Applied to balance the class distribution and improve model performance.
- **Mixed-Precision Training**: Improved computational efficiency and faster training using mixed-precision floating point format.

## Steps in the Project:

1. **Data Loading and Preprocessing**: The Food11 dataset is loaded from specified directories for training, validation, and testing.
2. **Model Architecture**: Fine-tuned EfficientNetB0 model with custom dense layers for classification.
3. **Callbacks**: Includes EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
4. **Class Weight Calculation**: Addresses class imbalance by computing weights for each class.
5. **Model Training and Evaluation**: Model is trained with data augmentation and evaluated on validation data.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- NumPy
- Matplotlib
- PIL (Pillow)

## How to Run

1. Clone this repository:
   ```bash
   https://github.com/vashu2425/FoodDetectionAndClassification-CNN.git
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

