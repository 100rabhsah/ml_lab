import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class KNNClassifier:
    def __init__(self, k=3):
        """Initialize KNN Classifier with k neighbors"""
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store the training data"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        for x in X:
            # Calculate distances and get k nearest neighbors
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            # Get most common label
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return np.array(predictions)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

def visualize_data_and_boundary(X, y, model, title="KNN Classification"):
    """Visualize data points and decision boundary"""
    # Create mesh grid
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', alpha=0.8, edgecolors='k')
    plt.title(f"{title} (k={model.k})")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    plt.show()

if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 50
    
    # Create two overlapping clusters
    cluster1 = np.random.randn(n_samples, 2) + np.array([1, 1])
    cluster2 = np.random.randn(n_samples, 2) + np.array([-1, -1])
    
    # Add some overlap
    overlap_indices1 = np.random.choice(n_samples, size=10, replace=False)
    overlap_indices2 = np.random.choice(n_samples, size=10, replace=False)
    cluster1[overlap_indices1] += np.random.randn(10, 2) * 0.5
    cluster2[overlap_indices2] += np.random.randn(10, 2) * 0.5
    
    X = np.vstack([cluster1, cluster2])
    y = np.array([0] * n_samples + [1] * n_samples)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train classifier
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    
    # Print results
    print(f"Training accuracy: {knn.score(X_train, y_train):.2%}")
    print(f"Test accuracy: {knn.score(X_test, y_test):.2%}")
    
    # Visualize results
    visualize_data_and_boundary(X, y, knn) 