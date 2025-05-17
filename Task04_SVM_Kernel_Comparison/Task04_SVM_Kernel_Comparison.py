import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dictionary to store results
results = {}

# Define different kernel functions
kernels = ['linear', 'poly', 'rbf']

# Train and evaluate SVM with different kernels
for kernel in kernels:
    print(f"\nTraining SVM with {kernel} kernel...")
    
    # Create and train the SVM classifier
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3)  # Using degree 3 for polynomial kernel
    else:
        svm = SVC(kernel=kernel)
    
    svm.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[kernel] = accuracy
    
    # Print detailed classification report
    print(f"\nClassification Report for {kernel} kernel:")
    print(classification_report(y_test, y_pred))

# Create a bar plot to compare accuracies
plt.figure(figsize=(10, 6))
bars = plt.bar(results.keys(), results.values())

# Customize the plot
plt.title('SVM Kernel Comparison', fontsize=14, pad=20)
plt.xlabel('Kernel Type', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)

# Set y-axis limits with some padding
min_acc = min(results.values())
max_acc = max(results.values())
plt.ylim(min_acc - 0.05, max_acc + 0.05)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}',
             ha='center', va='bottom', fontsize=12)

# Add grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()  # Display the plot directly instead of saving

print("\nResults Summary:")
for kernel, accuracy in results.items():
    print(f"{kernel} kernel accuracy: {accuracy:.3f}") 