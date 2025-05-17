import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn_classifier import KNNClassifier

# Set page config
st.set_page_config(
    page_title="KNN Classifier",
    page_icon="📊",
    layout="wide"
)

# Title and introduction
st.title("K-Nearest Neighbors (KNN) Classifier")
st.markdown("""
This app demonstrates a simple K-Nearest Neighbors classifier. You can either use our sample data or upload your own dataset.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Data source selection
data_source = st.sidebar.radio(
    "Choose Data Source",
    ["Use Sample Data", "Upload Your Data"],
    help="Select whether to use sample data or upload your own dataset"
)

# Number of neighbors
k = st.sidebar.slider(
    "Number of Neighbors (k)",
    min_value=1,
    max_value=10,
    value=3,
    help="This determines how many nearby points the algorithm will consider when making a prediction"
)

# Data loading section
if data_source == "Use Sample Data":
    # Generate a more realistic dataset
    np.random.seed(42)
    n_samples = 50  # 50 samples per class
    
    # Generate two overlapping clusters with noise
    cluster1 = np.random.randn(n_samples, 2) + np.array([1, 1])
    cluster2 = np.random.randn(n_samples, 2) + np.array([-1, -1])
    
    # Add some overlap by moving some points
    overlap_indices1 = np.random.choice(n_samples, size=10, replace=False)
    overlap_indices2 = np.random.choice(n_samples, size=10, replace=False)
    
    cluster1[overlap_indices1] += np.random.randn(10, 2) * 0.5
    cluster2[overlap_indices2] += np.random.randn(10, 2) * 0.5
    
    X = np.vstack([cluster1, cluster2])
    y = np.array([0] * n_samples + [1] * n_samples)
    
    st.sidebar.markdown("""
    ### Sample Data Information
    - 100 data points (50 per class)
    - 2 features
    - 2 classes with some overlap
    - Added noise for realism
    """)
else:
    st.sidebar.markdown("""
    ### Data Upload Instructions
    1. Upload a CSV file with the following format:
       - First two columns should be your features (X and Y coordinates)
       - Last column should be your class labels (0 or 1)
    2. Example format:
       ```
       feature1,feature2,class
       1.2,3.4,0
       2.3,4.5,1
       ...
       ```
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if len(data.columns) != 3:
                st.error("Please upload a CSV file with exactly 3 columns (2 features + 1 class)")
                st.stop()
            
            X = data.iloc[:, :2].values
            y = data.iloc[:, 2].values
            
            if not all(np.unique(y) == np.array([0, 1])):
                st.error("Class labels should be 0 and 1")
                st.stop()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.stop()
    else:
        st.info("Please upload a CSV file to begin")
        st.stop()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train KNN classifier
knn = KNNClassifier(k=k)
knn.fit(X_train, y_train)

# Calculate accuracies
train_accuracy = knn.score(X_train, y_train)
test_accuracy = knn.score(X_test, y_test)

# Create two columns for visualizations
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Your Data")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.8, edgecolors='k')
    ax.set_title("Data Points")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, label='Class')
    st.pyplot(fig)

with col2:
    st.markdown("### Classification Regions")
    # Create mesh grid with larger step size for faster computation
    h = 0.1  # Increased step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh grid
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.8, edgecolors='k')
    ax.set_title(f"Classification Regions (k={k})")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, label='Class')
    st.pyplot(fig)

# Display accuracy metrics
st.markdown("### Model Performance")
col3, col4 = st.columns(2)
with col3:
    st.metric("Training Accuracy", f"{train_accuracy:.2%}")
with col4:
    st.metric("Test Accuracy", f"{test_accuracy:.2%}")

# Explanation section
st.markdown("""
### How to Use This App

1. **Choose Your Data**:
   - Use our sample data (two overlapping clusters), or
   - Upload your own CSV file following the format in the sidebar

2. **Adjust Parameters**:
   - Use the slider to change the number of neighbors (k)
   - Watch how the classification regions change
   - Notice how accuracy changes with different k values

3. **Understand the Results**:
   - Left plot shows your data points
   - Right plot shows how the algorithm classifies different regions
   - Performance metrics show how accurate the model is

### Tips for Best Results
- Try different values of k to find the best performance
- The model works best when classes are well-separated
- For uploaded data, make sure your classes are clearly distinct
- Real-world data rarely achieves 100% accuracy
""")

# Add footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit") 