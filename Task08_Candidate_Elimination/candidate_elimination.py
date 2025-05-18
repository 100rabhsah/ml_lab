import numpy as np
from typing import List, Set, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

class CandidateElimination:
    def __init__(self, num_attributes: int):
        """
        Initialize the Candidate Elimination Algorithm
        
        Args:
            num_attributes (int): Number of attributes in the training data
        """
        self.num_attributes = num_attributes
        # Initialize S to contain the most specific hypothesis
        self.S = [['?' for _ in range(num_attributes)]]
        # Initialize G to contain the most general hypothesis
        self.G = [['?' for _ in range(num_attributes)]]
        
    def is_consistent(self, hypothesis: List[str], example: List[str], label: str) -> bool:
        """
        Check if a hypothesis is consistent with a given example
        
        Args:
            hypothesis (List[str]): The hypothesis to check
            example (List[str]): The training example
            label (str): The class label ('positive' or 'negative')
            
        Returns:
            bool: True if consistent, False otherwise
        """
        if label == 'positive':
            # For positive examples, check if hypothesis matches example
            for h, e in zip(hypothesis, example):
                if h != '?' and h != e:
                    return False
            return True
        else:
            # For negative examples, check if hypothesis doesn't match example
            for h, e in zip(hypothesis, example):
                if h != '?' and h != e:
                    return True
            return False
    
    def more_general(self, h1: List[str], h2: List[str]) -> bool:
        """
        Check if hypothesis h1 is more general than h2
        
        Args:
            h1 (List[str]): First hypothesis
            h2 (List[str]): Second hypothesis
            
        Returns:
            bool: True if h1 is more general than h2
        """
        for a1, a2 in zip(h1, h2):
            if a1 != '?' and a1 != a2:
                return False
        return True
    
    def update_S(self, example: List[str], label: str):
        """
        Update the S set based on the training example
        
        Args:
            example (List[str]): The training example
            label (str): The class label ('positive' or 'negative')
        """
        if label == 'positive':
            # Remove from S any hypothesis inconsistent with the positive example
            self.S = [h for h in self.S if self.is_consistent(h, example, label)]
            
            # Add to S all minimal generalizations of h in S
            new_S = []
            for h in self.S:
                for i in range(self.num_attributes):
                    if h[i] == '?':
                        continue
                    new_h = h.copy()
                    new_h[i] = '?'
                    if self.is_consistent(new_h, example, label):
                        new_S.append(new_h)
            self.S.extend(new_S)
            
            # Remove from S any hypothesis more general than another in S
            self.S = [h for h in self.S if not any(
                self.more_general(h2, h) for h2 in self.S if h2 != h
            )]
    
    def update_G(self, example: List[str], label: str):
        """
        Update the G set based on the training example
        
        Args:
            example (List[str]): The training example
            label (str): The class label ('positive' or 'negative')
        """
        if label == 'negative':
            # Remove from G any hypothesis inconsistent with the negative example
            self.G = [h for h in self.G if self.is_consistent(h, example, label)]
            
            # Add to G all minimal specializations of h in G
            new_G = []
            for h in self.G:
                for i in range(self.num_attributes):
                    if h[i] != '?':
                        continue
                    new_h = h.copy()
                    new_h[i] = example[i]
                    if self.is_consistent(new_h, example, label):
                        new_G.append(new_h)
            self.G.extend(new_G)
            
            # Remove from G any hypothesis more specific than another in G
            self.G = [h for h in self.G if not any(
                self.more_general(h, h2) for h2 in self.G if h2 != h
            )]
    
    def train(self, examples: List[List[str]], labels: List[str]):
        """
        Train the algorithm on the given examples
        
        Args:
            examples (List[List[str]]): List of training examples
            labels (List[str]): List of corresponding labels
        """
        for example, label in zip(examples, labels):
            self.update_S(example, label)
            self.update_G(example, label)
    
    def get_version_space(self) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Get the current version space
        
        Returns:
            Tuple[List[List[str]], List[List[str]]]: The S and G sets
        """
        return self.S, self.G

    def visualize_version_space(self, attribute_names: List[str] = None):
        """
        Visualize the version space using a grid representation
        
        Args:
            attribute_names (List[str], optional): Names of the attributes for better visualization
        """
        if attribute_names is None:
            attribute_names = [f'Attr_{i+1}' for i in range(self.num_attributes)]
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot S hypotheses
        for i, h in enumerate(self.S):
            y_pos = len(self.S) - i - 1
            for j, val in enumerate(h):
                color = 'lightgreen' if val == '?' else 'green'
                ax.add_patch(Rectangle((j, y_pos), 1, 1, facecolor=color, edgecolor='black'))
                ax.text(j + 0.5, y_pos + 0.5, str(val), ha='center', va='center')
        
        # Plot G hypotheses
        for i, h in enumerate(self.G):
            y_pos = -i - 1
            for j, val in enumerate(h):
                color = 'lightblue' if val == '?' else 'blue'
                ax.add_patch(Rectangle((j, y_pos), 1, 1, facecolor=color, edgecolor='black'))
                ax.text(j + 0.5, y_pos + 0.5, str(val), ha='center', va='center')
        
        # Set axis labels and title
        ax.set_xticks(np.arange(self.num_attributes) + 0.5)
        ax.set_xticklabels(attribute_names)
        ax.set_yticks([])
        ax.set_title('Version Space Visualization')
        
        # Add legend
        s_patch = mpatches.Patch(color='green', label='S (Specific)')
        g_patch = mpatches.Patch(color='blue', label='G (General)')
        ax.legend(handles=[s_patch, g_patch], loc='upper right')
        
        # Set axis limits
        ax.set_xlim(0, self.num_attributes)
        ax.set_ylim(-len(self.G) - 1, len(self.S))
        
        plt.tight_layout()
        plt.show()

def main():
    # Example usage
    # Let's say we have 4 attributes and some training examples
    num_attributes = 4
    ce = CandidateElimination(num_attributes)
    
    # Example training data
    examples = [
        ['Sunny', 'Warm', 'Normal', 'Strong'],
        ['Sunny', 'Warm', 'High', 'Strong'],
        ['Rainy', 'Cold', 'High', 'Strong'],
        ['Sunny', 'Warm', 'High', 'Strong']
    ]
    
    labels = ['positive', 'positive', 'negative', 'positive']
    
    # Train the algorithm
    ce.train(examples, labels)
    
    # Get the version space
    S, G = ce.get_version_space()
    
    print("Version Space:")
    print("\nS (Most specific hypotheses):")
    for h in S:
        print(h)
    print("\nG (Most general hypotheses):")
    for h in G:
        print(h)
    
    # Visualize the version space
    attribute_names = ['Sky', 'Temperature', 'Humidity', 'Wind']
    ce.visualize_version_space(attribute_names)

if __name__ == "__main__":
    main() 