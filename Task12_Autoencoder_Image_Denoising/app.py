import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
from autoencoder import Autoencoder
import torchvision.transforms as transforms

# Set page config
st.set_page_config(
    page_title="Image Denoising Autoencoder Results",
    page_icon="🎨",
    layout="wide"
)

# Title and introduction
st.title("🎨 Image Denoising with Autoencoder")
st.markdown("""
### What is this project about?
This project demonstrates how we can use a special type of neural network called an **Autoencoder** to clean up noisy images. 
Think of it like having a smart assistant that can look at a blurry or noisy picture and restore it to its original clear version!

### How does it work?
1. We take clear images (in this case, handwritten digits from MNIST dataset)
2. We add some random noise to make them blurry
3. The Autoencoder learns to:
   - Look at the noisy image
   - Understand what the original image should look like
   - Remove the noise and restore the image
""")

# Load the loss plot
st.header("📊 Training Progress")
st.image('results/loss_plot.png', caption='Training and Testing Losses Over Time')
st.markdown("""
### Understanding the Loss Plot
- The blue line shows how well the model is learning during training
- The orange line shows how well it performs on new, unseen images
- Lower values mean better performance
- The lines getting closer to zero means the model is getting better at cleaning up noisy images!
""")

# Load and display reconstruction results
st.header("🖼️ Image Reconstructions")
st.markdown("""
### What you're seeing below:
- **Original**: The clean, original images
- **Noisy**: The same images with added noise
- **Reconstructed**: The cleaned-up versions produced by our Autoencoder
""")

# Get all reconstruction images
reconstruction_files = sorted(glob.glob('results/reconstruction_epoch_*.png'))

# Create tabs for different epochs
tabs = st.tabs([f"Epoch {i*5}" for i in range(1, len(reconstruction_files) + 1)])

for tab, file in zip(tabs, reconstruction_files):
    with tab:
        st.image(file, caption=f'Reconstruction Results at {os.path.basename(file)}')
        st.markdown("""
        ### How to interpret these results:
        1. Look at how the noisy images (middle row) have random speckles and blur
        2. Notice how the reconstructed images (bottom row) are much clearer
        3. The model gets better at cleaning up the images as training progresses
        """)

# Interactive demo section
st.header("🎮 Try it yourself!")
st.markdown("""
### Want to see how the model works with your own images?
1. Upload an image of a handwritten digit (0-9)
2. We'll add some noise to it
3. Watch as the model tries to clean it up!
""")

# Add file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    
    try:
        model.load_state_dict(torch.load('results/autoencoder_model.pth', map_location=device))
        model.eval()
        
        # Process the uploaded image
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        
        # Prepare the image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Add noise
        noisy_img = img_tensor + 0.3 * torch.randn_like(img_tensor)
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed = model(noisy_img)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(image, caption="Original Image")
        
        with col2:
            st.image(noisy_img[0].cpu().numpy(), caption="Noisy Image")
        
        with col3:
            st.image(reconstructed[0].cpu().numpy(), caption="Reconstructed Image")
    except Exception as e:
        st.error(f"Error processing the image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
### Technical Details (for the curious!)
- The model uses a Convolutional Neural Network architecture
- It was trained on the MNIST dataset of handwritten digits
- The noise factor used during training was 0.3
- The model was trained for 50 epochs using the Adam optimizer
""") 