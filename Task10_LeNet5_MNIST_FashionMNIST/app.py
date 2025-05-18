import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from lenet5 import LeNet5

# Set page config
st.set_page_config(
    page_title="LeNet-5 MNIST & Fashion MNIST Visualizer",
    page_icon="🎯",
    layout="wide"
)

# Title and description
st.title("LeNet-5 Neural Network Visualizer")
st.markdown("""
This interactive app helps you understand how the LeNet-5 neural network performs on MNIST and Fashion MNIST datasets.
You can:
- View training history plots
- Test the model with your own handwritten digits
- Learn about the model architecture
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training History", "Model Architecture", "Try it Yourself"])

if page == "Training History":
    st.header("Training History")
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("MNIST Dataset")
        st.image("MNIST_training_history.png", use_column_width=True)
        st.markdown("""
        **MNIST Dataset Results:**
        - The model was trained on handwritten digits (0-9)
        - You can see how the model's accuracy improved over time
        - The gap between training and testing lines shows if the model is overfitting
        """)
    
    with col2:
        st.subheader("Fashion MNIST Dataset")
        st.image("FashionMNIST_training_history.png", use_column_width=True)
        st.markdown("""
        **Fashion MNIST Dataset Results:**
        - The model was trained on clothing items (10 categories)
        - The training process shows how well the model learned to classify different types of clothing
        - The plots help us understand if the model is learning effectively
        """)

elif page == "Model Architecture":
    st.header("LeNet-5 Architecture")
    
    st.markdown("""
    ### Understanding LeNet-5
    
    LeNet-5 is a classic Convolutional Neural Network (CNN) architecture that consists of:
    
    1. **Input Layer**
       - Takes 28x28 pixel images (grayscale)
       - Each image is a single channel (black and white)
    
    2. **First Convolutional Layer**
       - 6 filters of size 5x5
       - ReLU activation
       - MaxPooling to reduce size
    
    3. **Second Convolutional Layer**
       - 16 filters of size 5x5
       - ReLU activation
       - MaxPooling to reduce size
    
    4. **Fully Connected Layers**
       - Three dense layers
       - Final output: 10 classes (digits 0-9 or clothing categories)
    
    ### How it Works
    
    1. The image goes through convolutional layers that detect features
    2. Pooling layers reduce the size while keeping important information
    3. Fully connected layers combine these features to make the final prediction
    """)
    
    # Add a simple diagram
    st.image("https://miro.medium.com/max/1400/1*1TI1aGBZ4dybR6__DI9dzA.png", 
             caption="LeNet-5 Architecture Diagram", use_column_width=True)

else:  # Try it Yourself
    st.header("Try it Yourself!")
    
    st.markdown("""
    ### Test the Model
    
    You can test the model by drawing a digit or uploading an image.
    The model will predict whether it's a digit (0-9) or a clothing item.
    """)
    
    # Model selection
    model_type = st.radio("Select Model", ["MNIST (Digits)", "Fashion MNIST (Clothing)"])
    
    # Load the appropriate model
    @st.cache_resource
    def load_model(model_type):
        model = LeNet5()
        if model_type == "MNIST (Digits)":
            model.load_state_dict(torch.load('mnist_lenet5.pth', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load('fashion_mnist_lenet5.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    
    model = load_model(model_type)
    
    # Image upload
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Process the image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Display the image
        st.image(image, caption="Uploaded Image", width=200)
        
        # Make prediction
        if st.button("Predict"):
            # Transform image
            img_tensor = transform(image).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted = torch.max(output, 1)
                
            # Display result
            if model_type == "MNIST (Digits)":
                st.success(f"Predicted Digit: {predicted.item()}")
            else:
                fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
                st.success(f"Predicted Item: {fashion_classes[predicted.item()]}")
            
            # Show confidence scores
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            st.write("Confidence Scores:")
            for i, prob in enumerate(probabilities):
                if model_type == "MNIST (Digits)":
                    st.write(f"Digit {i}: {prob.item():.2%}")
                else:
                    st.write(f"{fashion_classes[i]}: {prob.item():.2%}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit</p>
    <p>LeNet-5 Implementation for MNIST and Fashion MNIST</p>
</div>
""", unsafe_allow_html=True) 