import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from model import CustomCNN
from utils import get_data_loaders, plot_training_history

def train_model(dataset_name, num_epochs=10, batch_size=64, learning_rate=0.001):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(dataset_name, batch_size)
    
    # Initialize model
    in_channels = 3 if dataset_name.lower() == 'cifar10' else 1
    model = CustomCNN(num_classes=10, in_channels=in_channels).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    train_losses = []
    train_accs = []
    test_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        print('-' * 50)
    
    # Plot training history
    plot_training_history(train_losses, train_accs, test_accs, dataset_name)
    
    return model, train_losses, train_accs, test_accs

if __name__ == '__main__':
    # Train on MNIST
    print("Training on MNIST dataset...")
    mnist_model, mnist_losses, mnist_accs, mnist_test_accs = train_model(
        dataset_name='MNIST',
        num_epochs=10,
        batch_size=64,
        learning_rate=0.001
    )
    
    # Train on CIFAR-10
    print("\nTraining on CIFAR-10 dataset...")
    cifar_model, cifar_losses, cifar_accs, cifar_test_accs = train_model(
        dataset_name='CIFAR10',
        num_epochs=10,
        batch_size=64,
        learning_rate=0.001
    ) 