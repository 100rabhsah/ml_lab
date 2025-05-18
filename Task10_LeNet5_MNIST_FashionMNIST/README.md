# LeNet-5 Implementation for MNIST and Fashion MNIST

This project implements the LeNet-5 convolutional neural network architecture to classify images from both the MNIST and Fashion MNIST datasets.

## Architecture

The LeNet-5 architecture consists of:
- 2 Convolutional layers with ReLU activation and MaxPooling
- 3 Fully Connected layers
- Output layer with 10 classes (digits 0-9 for MNIST, 10 clothing categories for Fashion MNIST)

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python lenet5.py
```

The script will:
1. Download and prepare both MNIST and Fashion MNIST datasets
2. Train separate models for each dataset
3. Save the trained models as:
   - `mnist_lenet5.pth`
   - `fashion_mnist_lenet5.pth`
4. Generate training history plots:
   - `MNIST_training_history.png`
   - `FashionMNIST_training_history.png`

## Training Details

- Batch size: 64
- Number of epochs: 10
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Cross Entropy Loss

The training progress will be displayed for each epoch, showing:
- Training loss and accuracy
- Testing loss and accuracy

## Results

The training history plots will show:
- Loss curves for both training and testing
- Accuracy curves for both training and testing

These plots help visualize the model's learning progress and potential overfitting issues. 