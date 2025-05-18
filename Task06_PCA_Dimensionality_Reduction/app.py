import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(
    page_title="PCA on Iris Dataset",
    page_icon="🌸",
    layout="wide"
)

# Add title and description
st.title("Principal Component Analysis (PCA) on Iris Dataset")
st.markdown("""
This interactive app demonstrates how Principal Component Analysis (PCA) works on the famous Iris dataset.
PCA helps us reduce the dimensionality of our data while preserving the most important information.
""")

# Load and prepare data
@st.cache_data
def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names

X, y, feature_names, target_names = load_data()

# Sidebar for user controls
st.sidebar.header("Controls")

# Add explanation about PCA
st.sidebar.markdown("""
### What is PCA?
PCA (Principal Component Analysis) is a technique that:
- Reduces the number of features in your data
- Preserves the most important information
- Helps visualize high-dimensional data
""")

# Add explanation about Iris dataset
st.sidebar.markdown("""
### About Iris Dataset
The Iris dataset contains:
- 150 samples of iris flowers
- 4 features (sepal length, sepal width, petal length, petal width)
- 3 classes (Setosa, Versicolor, Virginica)
""")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Create two columns for the plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("PCA Visualization")
    # Create scatter plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.set_title('PCA of Iris Dataset')
    legend1 = ax1.legend(*scatter.legend_elements(),
                        title="Iris Classes",
                        labels=target_names)
    st.pyplot(fig1)
    
    # Add explanation
    st.markdown("""
    **What you're seeing:**
    - Each point represents an iris flower
    - Colors show different iris species
    - The plot shows how well PCA separates the different species
    """)

with col2:
    st.subheader("Explained Variance")
    # Create variance plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(range(1, len(explained_variance_ratio) + 1), 
            explained_variance_ratio, 
            alpha=0.5, 
            label='Individual')
    ax2.step(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 
             where='mid', 
             label='Cumulative')
    ax2.set_xlabel('Principal Component')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Principal Components')
    ax2.legend()
    st.pyplot(fig2)
    
    # Add explanation
    st.markdown("""
    **What this means:**
    - Blue bars show how much information each component captures
    - Orange line shows how much information we keep by using more components
    - We can see that just 2 components capture most of the information!
    """)

# Add detailed information section
st.markdown("---")
st.subheader("Detailed Information")

# Create a DataFrame with explained variance information
variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Individual Variance': explained_variance_ratio,
    'Cumulative Variance': cumulative_variance_ratio
})

# Display the DataFrame
st.dataframe(variance_df.style.format({
    'Individual Variance': '{:.3f}',
    'Cumulative Variance': '{:.3f}'
}))

# Add conclusion
st.markdown("""
### Key Takeaways
1. The first two principal components capture most of the information in the dataset
2. We can reduce our 4 features to just 2 while keeping most of the important information
3. The visualization shows that the different iris species are well-separated in this reduced space
""") 