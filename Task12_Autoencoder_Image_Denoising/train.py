import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from autoencoder import Autoencoder, add_noise

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
NOISE_FACTOR = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory for saving results
os.makedirs('results', exist_ok=True)

def load_data():
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')):
            data = data.to(DEVICE)
            
            # Add noise to the input images
            noisy_data = add_noise(data, NOISE_FACTOR)
            
            # Forward pass
            output = model(noisy_data)
            loss = criterion(output, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(DEVICE)
                noisy_data = add_noise(data, NOISE_FACTOR)
                output = model(noisy_data)
                test_loss += criterion(output, data).item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Save sample reconstructions every 5 epochs
        if (epoch + 1) % 5 == 0:
            visualize_reconstruction(model, test_loader, epoch + 1)
    
    return train_losses, test_losses

def visualize_reconstruction(model, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        data, _ = next(iter(test_loader))
        data = data[:8].to(DEVICE)  # Take first 8 images
        
        # Add noise
        noisy_data = add_noise(data, NOISE_FACTOR)
        
        # Get reconstructions
        output = model(noisy_data)
        
        # Plot original, noisy, and reconstructed images
        fig, axes = plt.subplots(3, 8, figsize=(20, 8))
        
        for i in range(8):
            # Original images
            axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original')
            
            # Noisy images
            axes[1, i].imshow(noisy_data[i].cpu().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Noisy')
            
            # Reconstructed images
            axes[2, i].imshow(output[i].cpu().squeeze(), cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(f'results/reconstruction_epoch_{epoch}.png')
        plt.close()

def plot_losses(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()
    plt.savefig('results/loss_plot.png')
    plt.close()

def main():
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize model
    model = Autoencoder().to(DEVICE)
    
    # Train model
    train_losses, test_losses = train_model(model, train_loader, test_loader)
    
    # Plot losses
    plot_losses(train_losses, test_losses)
    
    # Save the model
    torch.save(model.state_dict(), 'results/autoencoder_model.pth')

if __name__ == '__main__':
    main() 