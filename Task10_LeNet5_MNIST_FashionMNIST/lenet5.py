import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # First Convolutional Layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second Convolutional Layer
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return running_loss / len(test_loader), accuracy

def plot_training_history(train_losses, train_accs, test_losses, test_accs, dataset_name):
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss History - {dataset_name}')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(test_accs, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy History - {dataset_name}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_training_history.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST dataset
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    # Load Fashion MNIST dataset
    fashion_mnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_mnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    fashion_mnist_train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size, shuffle=True)
    fashion_mnist_test_loader = DataLoader(fashion_mnist_test, batch_size=batch_size, shuffle=False)
    
    # Train on MNIST
    print("\nTraining on MNIST dataset...")
    mnist_model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mnist_model.parameters(), lr=learning_rate)
    
    mnist_train_losses = []
    mnist_train_accs = []
    mnist_test_losses = []
    mnist_test_accs = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(mnist_model, mnist_train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(mnist_model, mnist_test_loader, criterion, device)
        
        mnist_train_losses.append(train_loss)
        mnist_train_accs.append(train_acc)
        mnist_test_losses.append(test_loss)
        mnist_test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    plot_training_history(mnist_train_losses, mnist_train_accs, 
                         mnist_test_losses, mnist_test_accs, 'MNIST')
    
    # Train on Fashion MNIST
    print("\nTraining on Fashion MNIST dataset...")
    fashion_mnist_model = LeNet5().to(device)
    optimizer = optim.Adam(fashion_mnist_model.parameters(), lr=learning_rate)
    
    fashion_mnist_train_losses = []
    fashion_mnist_train_accs = []
    fashion_mnist_test_losses = []
    fashion_mnist_test_accs = []
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(fashion_mnist_model, fashion_mnist_train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_model(fashion_mnist_model, fashion_mnist_test_loader, criterion, device)
        
        fashion_mnist_train_losses.append(train_loss)
        fashion_mnist_train_accs.append(train_acc)
        fashion_mnist_test_losses.append(test_loss)
        fashion_mnist_test_accs.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    plot_training_history(fashion_mnist_train_losses, fashion_mnist_train_accs,
                         fashion_mnist_test_losses, fashion_mnist_test_accs, 'FashionMNIST')
    
    # Save models
    torch.save(mnist_model.state_dict(), 'mnist_lenet5.pth')
    torch.save(fashion_mnist_model.state_dict(), 'fashion_mnist_lenet5.pth')

if __name__ == '__main__':
    main() 