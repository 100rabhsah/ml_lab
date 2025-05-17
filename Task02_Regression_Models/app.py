import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="Regression Models Explorer",
    page_icon="📈",
    layout="wide"
)

# Set style
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Title and introduction
st.title("📈 Regression Models Explorer")
st.markdown("""
This interactive app helps you understand different types of regression models and how they work.
You can adjust the parameters and see how they affect the model's performance in real-time!
""")

# Sidebar controls
st.sidebar.header("Model Parameters")

# Data generation parameters
st.sidebar.subheader("Data Settings")
n_samples = st.sidebar.slider("Number of Data Points", 50, 200, 100)
noise_level = st.sidebar.slider("Noise Level", 0.1, 2.0, 0.5, 0.1)

# Model parameters
st.sidebar.subheader("Model Settings")
ridge_alpha = st.sidebar.slider("Ridge Regression Alpha", 0.1, 10.0, 1.0, 0.1)
polynomial_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)

def generate_sample_data(n_samples=100, noise=0.1):
    """Generate sample data with a polynomial relationship."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = 2 * X**2 + 3 * X + 1 + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y

def plot_regression_models(X, y, X_test, y_test, models, model_names):
    """Plot the regression models and their predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.scatter(X, y, color='blue', label='Training Data', alpha=0.5)
    ax.scatter(X_test, y_test, color='red', label='Test Data', alpha=0.5)
    
    # Plot predictions for each model
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    colors = ['green', 'orange', 'purple']
    
    for model, name, color in zip(models, model_names, colors):
        if 'Polynomial' in name:
            X_plot_poly = PolynomialFeatures(degree=polynomial_degree).fit_transform(X_plot)
            y_pred = model.predict(X_plot_poly)
        else:
            y_pred = model.predict(X_plot)
        
        ax.plot(X_plot, y_pred, color=color, label=f'{name} Prediction', linewidth=2)
    
    ax.set_title('Comparison of Regression Models', fontsize=15, pad=20)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def evaluate_models(X_test, y_test, models, model_names):
    """Evaluate and return metrics for each model."""
    results = []
    for model, name in zip(models, model_names):
        if 'Polynomial' in name:
            X_test_poly = PolynomialFeatures(degree=polynomial_degree).fit_transform(X_test)
            y_pred = model.predict(X_test_poly)
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            'Model': name,
            'MSE': mse,
            'R² Score': r2
        })
    return pd.DataFrame(results)

# Main app logic
def main():
    # Generate data
    X, y = generate_sample_data(n_samples=n_samples, noise=noise_level)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    poly_features = PolynomialFeatures(degree=polynomial_degree)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    ridge_model = Ridge(alpha=ridge_alpha)
    ridge_model.fit(X_train, y_train)
    
    models = [linear_model, poly_model, ridge_model]
    model_names = ['Linear Regression', 'Polynomial Regression', 'Ridge Regression']
    
    # Display the plot
    st.subheader("Model Predictions")
    fig = plot_regression_models(X_train, y_train, X_test, y_test, models, model_names)
    st.pyplot(fig)
    
    # Display model metrics
    st.subheader("Model Performance Metrics")
    metrics_df = evaluate_models(X_test, y_test, models, model_names)
    st.dataframe(metrics_df.style.format({
        'MSE': '{:.4f}',
        'R² Score': '{:.4f}'
    }))
    
    # Model explanations
    st.subheader("Understanding the Models")
    st.markdown("""
    ### 📊 Model Explanations:
    
    1. **Linear Regression**: 
       - The simplest model that tries to fit a straight line to the data
       - Best for data that shows a linear relationship
    
    2. **Polynomial Regression**:
       - Fits a curved line to the data
       - Can capture more complex relationships
       - Higher degrees can fit more complex patterns but might overfit
    
    3. **Ridge Regression**:
       - Similar to Linear Regression but with regularization
       - Helps prevent overfitting
       - Alpha parameter controls the strength of regularization
    """)
    
    # Model coefficients
    st.subheader("Model Coefficients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Linear Regression**")
        st.write(f"Coefficient: {linear_model.coef_[0]:.4f}")
        st.write(f"Intercept: {linear_model.intercept_:.4f}")
    
    with col2:
        st.write("**Polynomial Regression**")
        st.write("Coefficients:")
        st.write(poly_model.coef_)
    
    with col3:
        st.write("**Ridge Regression**")
        st.write(f"Coefficient: {ridge_model.coef_[0]:.4f}")
        st.write(f"Intercept: {ridge_model.intercept_:.4f}")

if __name__ == "__main__":
    main() 