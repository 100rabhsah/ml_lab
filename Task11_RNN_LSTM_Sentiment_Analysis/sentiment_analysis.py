import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import movie_reviews
from utils import plot_training_history, plot_confusion_matrix, print_classification_metrics

# Download required NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and convert to indices
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
            
        return torch.tensor(indices), torch.tensor(label)

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = output[:, -1, :]  # Get the last output
        output = self.dropout(output)
        output = torch.relu(self.fc1(output))
        output = self.sigmoid(self.fc2(output))
        return output

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = output[:, -1, :]  # Get the last output
        output = self.dropout(output)
        output = torch.relu(self.fc1(output))
        output = self.sigmoid(self.fc2(output))
        return output

def build_vocab(texts, max_words=10000):
    """Build vocabulary from texts."""
    word_freq = {}
    for text in texts:
        for word in word_tokenize(text.lower()):
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort words by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in sorted_words[:max_words-2]:  # -2 for <pad> and <unk>
        vocab[word] = len(vocab)
    
    return vocab

def load_data():
    """Load and preprocess the movie reviews dataset."""
    print("Loading movie reviews dataset...")
    
    # Load data
    texts = []
    labels = []
    
    for category in ['pos', 'neg']:
        for fileid in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)
    
    # Split into train and test sets (80-20 split)
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    # Build vocabulary
    vocab = build_vocab(train_texts)
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)
    
    # Create dataloaders with smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, len(vocab)

def train_and_evaluate_model(model, train_loader, test_loader, model_name, device):
    """Train and evaluate a model."""
    print(f"\nTraining {model_name}...")
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=0.001)  # Lower learning rate
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    # Training loop
    for epoch in range(20):  # Increased epochs
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(data).squeeze()
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (output > 0.5).float()
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.float().to(device)
                output = model(data).squeeze()
                val_loss += criterion(output, target).item()
                predicted = (output > 0.5).float()
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).squeeze()
            predicted = (output > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Plot results
    plot_training_history(history, model_name)
    plot_confusion_matrix(all_targets, all_preds, model_name)
    print_classification_metrics(all_targets, all_preds, model_name)
    
    return model, history

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader, vocab_size = load_data()
    
    # Create and train RNN model
    rnn_model = RNNModel(vocab_size)
    rnn_model, rnn_history = train_and_evaluate_model(
        rnn_model, train_loader, test_loader, "RNN", device
    )
    
    # Create and train LSTM model
    lstm_model = LSTMModel(vocab_size)
    lstm_model, lstm_history = train_and_evaluate_model(
        lstm_model, train_loader, test_loader, "LSTM", device
    )

if __name__ == "__main__":
    main() 