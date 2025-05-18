# Custom CNN for MNIST and CIFAR-10 Classification

This project implements a custom Convolutional Neural Network (CNN) with two convolutional layers to classify images from the MNIST and CIFAR-10 datasets.

## Model Architecture

The CNN architecture consists of:
- 2 Convolutional layers with batch normalization and max pooling
- Fully connected layers with dropout for regularization
- ReLU activation functions

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

To train the model on both MNIST and CIFAR-10 datasets, simply run:
```bash
python train.py
```

The script will:
1. Train the model on MNIST dataset
2. Train the model on CIFAR-10 dataset
3. Generate training history plots for both datasets

## Implementation Details

- The model automatically adapts to handle both MNIST (grayscale) and CIFAR-10 (RGB) images
- Training progress is displayed with a progress bar
- Training and test accuracies are recorded for each epoch
- Training history plots are saved as PNG files

## Expected Results

The model should achieve:
- MNIST: ~98-99% test accuracy
- CIFAR-10: ~70-75% test accuracy

Note: Actual results may vary depending on hardware and random initialization. 