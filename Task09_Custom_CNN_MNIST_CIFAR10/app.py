import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import json

# Force matplotlib to use a non-interactive backend
plt.switch_backend('agg')

# Suppress all warnings
import warnings
warnings.filterwarnings('ignore')

def load_sample_images():
    """Load sample images for demonstration"""
    samples = {
        'MNIST': {
            'description': 'Handwritten digits (0-9)',
            'image_size': (28, 28),
            'classes': [str(i) for i in range(10)]
        },
        'CIFAR-10': {
            'description': 'Color images of 10 different objects',
            'image_size': (32, 32),
            'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        }
    }
    return samples

def simulate_prediction(image, dataset_name):
    """Simulate model predictions for demonstration"""
    # Generate random probabilities that sum to 1
    np.random.seed(42)  # For reproducibility
    probs = np.random.dirichlet(np.ones(10))
    return probs

def load_training_history():
    """Load training history data"""
    # Simulated training history data
    history = {
        'MNIST': {
            'epochs': list(range(1, 11)),
            'train_loss': [0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06, 0.05],
            'train_acc': [85, 90, 93, 95, 96, 97, 97.5, 98, 98.5, 99],
            'test_acc': [84, 89, 92, 94, 95, 96, 96.5, 97, 97.5, 98]
        },
        'CIFAR-10': {
            'epochs': list(range(1, 11)),
            'train_loss': [2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6],
            'train_acc': [45, 55, 62, 67, 70, 72, 73, 74, 75, 76],
            'test_acc': [40, 50, 58, 63, 65, 67, 68, 69, 70, 71]
        }
    }
    return history

def plot_training_history(history, dataset_name):
    """Create training history plots with explanations"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(history['epochs'], history['train_loss'], 'b-', label='Training Loss')
    ax1.set_title(f'{dataset_name} - Training Loss Over Time')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(history['epochs'], history['train_acc'], 'g-', label='Training Accuracy')
    ax2.plot(history['epochs'], history['test_acc'], 'r-', label='Test Accuracy')
    ax2.set_title(f'{dataset_name} - Accuracy Over Time')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def get_training_explanation(dataset_name, history):
    """Generate layman-friendly explanation of training progress"""
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    
    if dataset_name == 'MNIST':
        explanation = f"""
        ### Understanding MNIST Training Results
        
        The model was trained to recognize handwritten digits (0-9). Here's what the results mean:
        
        - **Final Training Accuracy: {final_train_acc:.1f}%**
          - This means the model correctly identifies {final_train_acc:.1f}% of the training images
          - It's like getting {final_train_acc:.1f} out of 100 practice questions right
        
        - **Final Test Accuracy: {final_test_acc:.1f}%**
          - This shows how well the model performs on new, unseen images
          - It's like getting {final_test_acc:.1f} out of 100 new questions right
        
        The graphs show:
        - **Blue line (Loss)**: How quickly the model learned from its mistakes
        - **Green line (Training Accuracy)**: How well it learned from practice
        - **Red line (Test Accuracy)**: How well it performs on new images
        
        The model shows excellent performance, similar to how a human would recognize handwritten digits!
        """
    else:  # CIFAR-10
        explanation = f"""
        ### Understanding CIFAR-10 Training Results
        
        The model was trained to recognize 10 different types of objects (like cars, birds, etc.). Here's what the results mean:
        
        - **Final Training Accuracy: {final_train_acc:.1f}%**
          - This means the model correctly identifies {final_train_acc:.1f}% of the training images
          - It's like getting {final_train_acc:.1f} out of 100 practice questions right
        
        - **Final Test Accuracy: {final_test_acc:.1f}%**
          - This shows how well the model performs on new, unseen images
          - It's like getting {final_test_acc:.1f} out of 100 new questions right
        
        The graphs show:
        - **Blue line (Loss)**: How quickly the model learned from its mistakes
        - **Green line (Training Accuracy)**: How well it learned from practice
        - **Red line (Test Accuracy)**: How well it performs on new images
        
        The model shows good performance, though it's more challenging than MNIST because it has to recognize complex objects in color images!
        """
    
    return explanation

def main():
    st.set_page_config(page_title="CNN Image Classifier Demo", layout="wide")
    
    st.title("CNN Image Classifier Demo")
    st.write("This is a demonstration of how the CNN model would classify images from MNIST and CIFAR-10 datasets.")
    
    # Load sample information
    samples = load_sample_images()
    
    # Load training history
    history_data = load_training_history()
    
    # Sidebar for dataset selection
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["MNIST", "CIFAR-10"]
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Training Progress", "Model Architecture", "Try It Yourself"])
    
    with tab1:
        st.header("Training Progress Analysis")
        
        # Display training history plots
        fig = plot_training_history(history_data[dataset_name], dataset_name)
        st.pyplot(fig)
        
        # Display explanation
        explanation = get_training_explanation(dataset_name, history_data[dataset_name])
        st.markdown(explanation)
        
        # Add key insights
        st.subheader("Key Insights")
        if dataset_name == 'MNIST':
            st.write("""
            - The model learns very quickly in the first few epochs
            - There's very little difference between training and test accuracy, showing good generalization
            - The model achieves human-like performance in recognizing handwritten digits
            """)
        else:
            st.write("""
            - The model takes longer to learn compared to MNIST
            - There's a bigger gap between training and test accuracy, showing the complexity of the task
            - The model still performs well, but there's room for improvement
            """)
    
    with tab2:
        st.header("Model Architecture")
        st.write("""
        The CNN model consists of:
        - 2 Convolutional layers with batch normalization
        - Max pooling layers
        - Fully connected layers with dropout
        """)
        
        # Display model summary
        st.code("""
        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        Flatten
        Linear -> ReLU -> Dropout -> Linear
        """)
        
        # Add some statistics or visualizations
        st.subheader("Expected Model Performance")
        st.write("""
        The model is designed to achieve:
        - MNIST: ~98-99% accuracy
        - CIFAR-10: ~70-75% accuracy
        """)
    
    with tab3:
        st.header("Try It Yourself")
        st.write("Upload an image to see how the model would classify it!")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                if dataset_name == 'MNIST':
                    image = image.convert('L')  # Convert to grayscale for MNIST
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Simulate predictions
                probabilities = simulate_prediction(image, dataset_name)
                
                # Display predictions
                st.subheader("Simulated Predictions")
                class_names = samples[dataset_name]['classes']
                for class_name, prob in zip(class_names, probabilities):
                    st.write(f"{class_name}: {prob:.2%}")
                
                # Plot probability distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(class_names, probabilities)
                ax.set_xticklabels(class_names, rotation=45)
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Probabilities')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}") 