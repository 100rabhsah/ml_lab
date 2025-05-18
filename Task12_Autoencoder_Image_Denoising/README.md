# Image Denoising Autoencoder

This project implements an autoencoder for image denoising using the MNIST dataset. The autoencoder is trained to reconstruct clean images from noisy versions of the same images.

## Project Structure

- `autoencoder.py`: Contains the autoencoder model architecture and noise addition function
- `train.py`: Main training script with data loading, training loop, and visualization
- `requirements.txt`: Required Python packages
- `results/`: Directory containing saved model and visualizations

## Model Architecture

The autoencoder consists of:
- Encoder: 4 convolutional layers with increasing channels (1→16→32→64→128)
- Decoder: 4 transposed convolutional layers with decreasing channels (128→64→32→16→1)
- ReLU activation functions and Sigmoid output

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model, simply run:
```bash
python train.py
```

The script will:
1. Download the MNIST dataset
2. Train the autoencoder for 50 epochs
3. Save visualizations of reconstructions every 5 epochs
4. Save the trained model and loss plots

## Results

The training process will generate:
- Reconstruction visualizations in `results/reconstruction_epoch_*.png`
- Loss plot in `results/loss_plot.png`
- Trained model in `results/autoencoder_model.pth`

Each reconstruction visualization shows:
- Original images
- Noisy images (input to the autoencoder)
- Reconstructed images (output from the autoencoder)

## Hyperparameters

- Batch size: 128
- Learning rate: 0.001
- Number of epochs: 50
- Noise factor: 0.3
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE) 