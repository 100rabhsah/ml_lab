import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import io

# Set page config with a nice theme
st.set_page_config(
    page_title="SVM Kernel Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    th, td {
        padding: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
    }
    th {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Add a nice header with emoji
st.title("🤖 SVM Kernel Comparison Analysis")
st.markdown("---")

# Add a clear explanation section
st.markdown("""
### 📚 What is SVM and Kernel Functions?

Support Vector Machine (SVM) is a powerful machine learning algorithm that can be used for both classification and regression tasks. 
The performance of SVM depends heavily on the choice of kernel function, which helps in transforming the data into a higher dimension 
where it becomes easier to separate different classes.

#### 🔍 The Three Kernel Functions We're Testing:

1. **Linear Kernel** 📈
   - Simplest kernel function
   - Works best when data is linearly separable
   - Fastest to compute

2. **Polynomial Kernel** 📊
   - Can capture non-linear relationships
   - Good for normalized data
   - More complex than linear kernel

3. **RBF (Radial Basis Function) Kernel** 🌟
   - Most commonly used kernel
   - Works well in most cases
   - Can handle non-linear relationships

""")

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Create a nice sidebar with dataset information
with st.sidebar:
    st.header("📊 Dataset Information")
    st.markdown("---")
    
    # Add dataset info with icons
    st.markdown("""
    ### 📑 Dataset Details
    - **Dataset**: Breast Cancer Wisconsin
    - **Type**: Binary Classification
    - **Purpose**: Predict if a tumor is malignant or benign
    """)
    
    st.markdown("---")
    st.markdown(f"**📊 Samples**: {X.shape[0]}")
    st.markdown(f"**🔢 Features**: {X.shape[1]}")
    st.markdown(f"**🎯 Classes**: {len(np.unique(y))}")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    test_size = st.slider(
        "Test Size",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.05,
        help="Percentage of data to use for testing"
    )

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}
kernels = ['linear', 'poly', 'rbf']

# Create columns for results
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔄 Model Training Progress")
    progress_bar = st.progress(0)
    
    # Train and evaluate SVM with different kernels
    for i, kernel in enumerate(kernels):
        st.markdown(f"### Training {kernel.upper()} Kernel...")
        
        # Create and train the SVM classifier
        if kernel == 'poly':
            svm = SVC(kernel=kernel, degree=3)
        else:
            svm = SVC(kernel=kernel)
        
        svm.fit(X_train_scaled, y_train)
        y_pred = svm.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
        
        # Update progress
        progress_bar.progress((i + 1) / len(kernels))
        
        # Print detailed classification report in table format
        st.markdown("#### 📊 Classification Report")
        
        # Convert classification report to DataFrame
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Format the DataFrame
        df_report = df_report.round(3)
        df_report = df_report.drop('support', axis=1)  # Remove support column for cleaner display
        
        # Display the table with custom styling
        st.dataframe(
            df_report,
            use_container_width=True,
            hide_index=False
        )
        
        st.markdown("---")

with col2:
    st.subheader("📈 Accuracy Comparison")
    
    # Create a bar plot with a modern style
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(results.keys(), results.values(), color=['#2ecc71', '#3498db', '#9b59b6'])

    # Customize the plot
    ax.set_title('SVM Kernel Comparison', fontsize=14, pad=20)
    ax.set_xlabel('Kernel Type', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)

    # Set y-axis limits with some padding
    min_acc = min(results.values())
    max_acc = max(results.values())
    ax.set_ylim(min_acc - 0.05, max_acc + 0.05)

    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12)

    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Display the plot
    st.pyplot(fig)
    
    # Add interpretation
    st.markdown("""
    ### 📝 Interpretation
    The bar chart above shows the accuracy of each kernel function. Higher bars indicate better performance.
    - The best performing kernel is highlighted in the results below
    - All accuracies are calculated on the test set
    - The model is trained on the remaining data
    """)

# Display results summary in a nice format
st.markdown("---")
st.subheader("🎯 Results Summary")
for kernel, accuracy in results.items():
    st.markdown(f"**{kernel.upper()} Kernel**: {accuracy:.3f}")

# Add a nice footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Created with ❤️ using Streamlit</p>
    <p>Machine Learning Lab - SVM Kernel Comparison</p>
</div>
""", unsafe_allow_html=True) 