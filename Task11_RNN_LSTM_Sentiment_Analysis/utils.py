import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def preprocess_data(texts, max_words=10000, max_len=200):
    """
    Preprocess text data by tokenizing and padding sequences.
    
    Args:
        texts (list): List of text reviews
        max_words (int): Maximum number of words to keep
        max_len (int): Maximum sequence length
        
    Returns:
        tuple: (tokenizer, padded_sequences)
    """
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.word_index
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    
    return tokenizer, padded_sequences

def plot_training_history(history, model_name):
    """
    Plot training and validation metrics.
    
    Args:
        history (dict): Dictionary containing training history
        model_name (str): Name of the model (RNN or LSTM)
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def print_classification_metrics(y_true, y_pred, model_name):
    """
    Print classification metrics for model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name (str): Name of the model
    """
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_true, y_pred)) 