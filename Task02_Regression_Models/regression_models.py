import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style for better visualizations
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

def generate_sample_data(n_samples=100, noise=0.1):
    """Generate sample data with a polynomial relationship."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n_samples)
    y = 2 * X**2 + 3 * X + 1 + noise * np.random.randn(n_samples)
    return X.reshape(-1, 1), y

def plot_regression_models(X, y, X_test, y_test, models, model_names):
    """Plot the regression models and their predictions."""
    plt.figure(figsize=(15, 10))
    
    # Plot training data
    plt.scatter(X, y, color='blue', label='Training Data', alpha=0.5)
    plt.scatter(X_test, y_test, color='red', label='Test Data', alpha=0.5)
    
    # Plot predictions for each model
    X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    colors = ['green', 'orange', 'purple']
    
    for model, name, color in zip(models, model_names, colors):
        # Transform X_plot for polynomial features if needed
        if 'Polynomial' in name:
            X_plot_poly = PolynomialFeatures(degree=2).fit_transform(X_plot)
            y_pred = model.predict(X_plot_poly)
        else:
            y_pred = model.predict(X_plot)
        
        plt.plot(X_plot, y_pred, color=color, label=f'{name} Prediction', linewidth=2)
    
    plt.title('Comparison of Regression Models', fontsize=15, pad=20)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_models(X_test, y_test, models, model_names):
    """Evaluate and print metrics for each model."""
    print("\nModel Evaluation Metrics:")
    print("-" * 50)
    print(f"{'Model':<20} {'MSE':<15} {'R² Score':<15}")
    print("-" * 50)
    
    for model, name in zip(models, model_names):
        # Transform X_test for polynomial features if needed
        if 'Polynomial' in name:
            X_test_poly = PolynomialFeatures(degree=2).fit_transform(X_test)
            y_pred = model.predict(X_test_poly)
        else:
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} {mse:<15.4f} {r2:<15.4f}")

def main():
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=0.5)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 1. Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    
    # 2. Polynomial Regression
    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    
    # 3. Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    
    # Store models and their names
    models = [linear_model, poly_model, ridge_model]
    model_names = ['Linear Regression', 'Polynomial Regression', 'Ridge Regression']
    
    # Plot the results
    plot_regression_models(X_train, y_train, X_test, y_test, models, model_names)
    
    # Evaluate models
    evaluate_models(X_test, y_test, models, model_names)
    
    # Print model coefficients
    print("\nModel Coefficients:")
    print("-" * 50)
    print("Linear Regression:", linear_model.coef_[0], linear_model.intercept_)
    print("Polynomial Regression:", poly_model.coef_)
    print("Ridge Regression:", ridge_model.coef_[0], ridge_model.intercept_)

if __name__ == "__main__":
    main() 