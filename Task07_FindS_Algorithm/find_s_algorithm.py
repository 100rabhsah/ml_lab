"""
Find-S Algorithm Implementation

The Find-S algorithm is a basic concept learning algorithm that finds the most specific hypothesis
consistent with the positive training examples. It starts with the most specific hypothesis and
generalizes it as needed to cover positive examples.

Author: Sourabh Sah
Date: 2025-05-18
"""

import numpy as np
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate


class FindSAlgorithm:
    def __init__(self, attributes: List[str], possible_values: List[List[str]]):
        """
        Initialize the Find-S Algorithm.
        
        Args:
            attributes (List[str]): List of attribute names
            possible_values (List[List[str]]): List of possible values for each attribute
        """
        self.attributes = attributes
        self.possible_values = possible_values
        self.hypothesis = ['0'] * len(attributes)  # Initialize with most specific hypothesis
        self.hypothesis_history = [self.hypothesis.copy()]  # Track hypothesis changes
        
    def train(self, training_data: List[Tuple[List[str], str]]) -> None:
        """
        Train the model using the Find-S algorithm.
        
        Args:
            training_data (List[Tuple[List[str], str]]): List of training examples,
                where each example is a tuple of (features, label)
        """
        for example, label in training_data:
            if label == 'Yes':  # Only consider positive examples
                for i, value in enumerate(example):
                    if self.hypothesis[i] == '0':  # First positive example
                        self.hypothesis[i] = value
                    elif self.hypothesis[i] != value:  # Generalize if different
                        self.hypothesis[i] = '?'
                self.hypothesis_history.append(self.hypothesis.copy())
    
    def predict(self, example: List[str]) -> str:
        """
        Predict the class of a new example.
        
        Args:
            example (List[str]): The example to classify
            
        Returns:
            str: Predicted class ('Yes' or 'No')
        """
        for i, value in enumerate(example):
            if self.hypothesis[i] != '?' and self.hypothesis[i] != value:
                return 'No'
        return 'Yes'
    
    def get_hypothesis(self) -> List[str]:
        """
        Get the current hypothesis.
        
        Returns:
            List[str]: The current hypothesis
        """
        return self.hypothesis
    
    def visualize_hypothesis_evolution(self) -> None:
        """
        Visualize how the hypothesis changes during training.
        """
        plt.figure(figsize=(12, 6))
        
        # Create a color map for different values
        unique_values = set()
        for hyp in self.hypothesis_history:
            unique_values.update(hyp)
        color_map = {val: i for i, val in enumerate(unique_values)}
        
        # Create the heatmap data
        data = np.zeros((len(self.hypothesis_history), len(self.attributes)))
        for i, hyp in enumerate(self.hypothesis_history):
            for j, val in enumerate(hyp):
                data[i, j] = color_map[val]
        
        # Plot the heatmap
        sns.heatmap(data, 
                   xticklabels=self.attributes,
                   yticklabels=[f'Step {i}' for i in range(len(self.hypothesis_history))],
                   cmap='viridis',
                   cbar=False)
        
        plt.title('Hypothesis Evolution During Training')
        plt.xlabel('Attributes')
        plt.ylabel('Training Steps')
        
        # Add value annotations
        for i, hyp in enumerate(self.hypothesis_history):
            for j, val in enumerate(hyp):
                plt.text(j + 0.5, i + 0.5, val,
                        ha='center', va='center',
                        color='white' if val == '0' else 'black')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_final_hypothesis(self) -> None:
        """
        Visualize the final hypothesis in a table format.
        """
        # Create a table with attribute names and their values
        table_data = [[attr, val] for attr, val in zip(self.attributes, self.hypothesis)]
        
        # Print the table
        print("\nFinal Hypothesis:")
        print(tabulate(table_data, 
                      headers=['Attribute', 'Value'],
                      tablefmt='grid'))
        
        # Create a bar plot for the final hypothesis
        plt.figure(figsize=(10, 6))
        colors = ['green' if val == '?' else 'blue' for val in self.hypothesis]
        plt.bar(self.attributes, [1] * len(self.attributes), color=colors)
        plt.title('Final Hypothesis Visualization')
        plt.xticks(rotation=45, ha='right')
        plt.yticks([])
        
        # Add value annotations
        for i, val in enumerate(self.hypothesis):
            plt.text(i, 0.5, val,
                    ha='center', va='center',
                    color='white')
        
        plt.tight_layout()
        plt.show()


def main():
    # Example: Enjoy Sport problem
    attributes = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']
    possible_values = [
        ['Sunny', 'Rainy'],
        ['Warm', 'Cold'],
        ['Normal', 'High'],
        ['Strong', 'Weak'],
        ['Warm', 'Cool'],
        ['Same', 'Change']
    ]
    
    # Training data
    training_data = [
        (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
        (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
        (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes')
    ]
    
    # Create and train the model
    find_s = FindSAlgorithm(attributes, possible_values)
    find_s.train(training_data)
    
    # Visualize the results
    find_s.visualize_hypothesis_evolution()
    find_s.visualize_final_hypothesis()
    
    # Test the model
    test_example = ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same']
    prediction = find_s.predict(test_example)
    print(f"\nTest Example: {test_example}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main() 