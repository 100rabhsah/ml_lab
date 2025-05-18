import streamlit as st
import pandas as pd
import numpy as np
from candidate_elimination import CandidateElimination
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Candidate Elimination Algorithm", layout="wide")

# Title and Introduction
st.title("🎯 Candidate Elimination Algorithm Visualizer")
st.markdown("""
This app helps you understand and visualize how the Candidate Elimination Algorithm works in machine learning.
The algorithm learns from examples to create a version space of possible hypotheses.

### 📝 How to use this app:

1. **Choose your input method:**
   - Use the example data (recommended for first-time users)
   - Upload your own CSV file
   - Enter data manually

2. **Data Format:**
   - Each row represents one example
   - The last column should be the label ('positive' or 'negative')
   - Other columns are attributes
   - Example format:
     ```
     Sky,Temperature,Humidity,Wind,Label
     Sunny,Warm,Normal,Strong,positive
     Rainy,Cold,High,Strong,negative
     ```

3. **Understanding the Output:**
   - S (Specific) hypotheses: Most specific rules that match positive examples
   - G (General) hypotheses: Most general rules that don't match negative examples
   - '?' means any value is acceptable for that attribute
""")

# Sidebar for input method selection
st.sidebar.title("Input Method")
input_method = st.sidebar.radio(
    "Choose how to input your data:",
    ["Use Example Data", "Upload CSV", "Manual Input"]
)

# Enhanced example data with more diverse cases
example_data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny', 'Rainy', 'Sunny', 'Cloudy'],
    'Temperature': ['Warm', 'Warm', 'Cold', 'Warm', 'Cold', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High', 'Normal', 'High', 'Normal'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong'],
    'Label': ['positive', 'positive', 'negative', 'positive', 'negative', 'negative', 'positive']
}

def process_data(data):
    # Convert data to the format expected by the algorithm
    examples = []
    labels = []
    
    # Convert DataFrame to list of lists
    for _, row in data.iterrows():
        example = row[:-1].tolist()  # All columns except the last one
        label = row[-1]  # Last column is the label
        examples.append(example)
        labels.append(label)
    
    return examples, labels

def display_hypothesis(hypothesis, attribute_names):
    """Helper function to display a hypothesis in a readable format"""
    return " ∧ ".join([f"{attr}={val}" for attr, val in zip(attribute_names, hypothesis)])

# Main content area
if input_method == "Use Example Data":
    st.subheader("Example Data")
    df = pd.DataFrame(example_data)
    st.dataframe(df)
    
    # Add explanation of the example data
    st.markdown("""
    ### 📊 Example Data Explanation
    
    This example dataset represents weather conditions for playing tennis:
    - **Positive examples** (when tennis can be played):
        - Sunny, Warm, Normal, Strong
        - Sunny, Warm, High, Strong
        - Sunny, Warm, High, Weak
        - Cloudy, Warm, Normal, Strong
    
    - **Negative examples** (when tennis cannot be played):
        - Rainy, Cold, High, Strong
        - Rainy, Cold, Normal, Strong
        - Sunny, Cold, High, Weak
    
    This diverse dataset will help demonstrate how the algorithm:
    1. Creates specific hypotheses from positive examples
    2. Forms general hypotheses that exclude negative examples
    3. Maintains a version space between these extremes
    """)
    
    if st.button("Run Algorithm on Example Data"):
        examples, labels = process_data(df)
        ce = CandidateElimination(len(df.columns) - 1)  # -1 for label column
        ce.train(examples, labels)
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### S (Specific) Hypotheses")
            S, _ = ce.get_version_space()
            if S:
                for i, h in enumerate(S):
                    st.markdown(f"**Hypothesis {i+1}:**")
                    st.markdown(f"```\n{display_hypothesis(h, df.columns[:-1])}\n```")
            else:
                st.markdown("No specific hypotheses found.")
            st.markdown("""
            **Interpretation:**
            - These are the most specific rules that match all positive examples
            - They represent the minimum conditions required for a positive outcome
            """)
        
        with col2:
            st.markdown("### G (General) Hypotheses")
            _, G = ce.get_version_space()
            if G:
                for i, h in enumerate(G):
                    st.markdown(f"**Hypothesis {i+1}:**")
                    st.markdown(f"```\n{display_hypothesis(h, df.columns[:-1])}\n```")
            else:
                st.markdown("No general hypotheses found.")
            st.markdown("""
            **Interpretation:**
            - These are the most general rules that don't match any negative examples
            - They represent the maximum conditions that could still lead to a positive outcome
            """)
        
        # Visualization
        st.subheader("Visualization")
        st.markdown("""
        ### 📈 Version Space Visualization
        
        The visualization below shows:
        - **Green cells**: Specific hypotheses (S)
        - **Blue cells**: General hypotheses (G)
        - **'?'**: Represents a wildcard (any value is acceptable)
        - **Actual values**: Show the specific conditions required
        """)
        
        # Create a new figure for visualization
        plt.figure(figsize=(12, 6))
        ce.visualize_version_space(df.columns[:-1].tolist())
        st.pyplot(plt.gcf())
        plt.close()
        
        # Add interpretation of the visualization
        st.markdown("""
        ### 🔍 Understanding the Visualization
        
        1. **Specific Hypotheses (S)**:
           - Show the minimum conditions that must be met
           - More filled cells indicate more specific requirements
           - Helps identify critical attributes for positive outcomes
        
        2. **General Hypotheses (G)**:
           - Show the maximum flexibility in conditions
           - More '?' symbols indicate more flexible attributes
           - Helps understand which attributes are less critical
        
        3. **Version Space**:
           - The space between S and G represents all possible valid hypotheses
           - Any hypothesis in this space is consistent with the training data
        """)

elif input_method == "Upload CSV":
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
            
            if st.button("Run Algorithm on Uploaded Data"):
                examples, labels = process_data(df)
                ce = CandidateElimination(len(df.columns) - 1)
                ce.train(examples, labels)
                
                # Display results
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### S (Specific) Hypotheses")
                    S, _ = ce.get_version_space()
                    for i, h in enumerate(S):
                        st.write(f"Hypothesis {i+1}:", h)
                
                with col2:
                    st.markdown("### G (General) Hypotheses")
                    _, G = ce.get_version_space()
                    for i, h in enumerate(G):
                        st.write(f"Hypothesis {i+1}:", h)
                
                # Visualization
                st.subheader("Visualization")
                fig, ax = plt.subplots(figsize=(12, 6))
                ce.visualize_version_space(df.columns[:-1].tolist())
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your CSV file follows the required format.")

else:  # Manual Input
    st.subheader("Manual Data Input")
    
    # Number of attributes
    num_attributes = st.number_input("Number of attributes (excluding label)", min_value=1, value=4)
    
    # Create input fields for attribute names
    attribute_names = []
    for i in range(num_attributes):
        name = st.text_input(f"Name of attribute {i+1}", value=f"Attribute_{i+1}")
        attribute_names.append(name)
    
    # Number of examples
    num_examples = st.number_input("Number of examples", min_value=1, value=4)
    
    # Create input fields for examples
    examples = []
    labels = []
    
    for i in range(num_examples):
        st.markdown(f"### Example {i+1}")
        example = []
        for j in range(num_attributes):
            value = st.text_input(f"Value for {attribute_names[j]}", key=f"ex_{i}_attr_{j}")
            example.append(value)
        examples.append(example)
        
        label = st.radio(f"Label for example {i+1}", ["positive", "negative"], key=f"label_{i}")
        labels.append(label)
    
    if st.button("Run Algorithm on Manual Data"):
        ce = CandidateElimination(num_attributes)
        ce.train(examples, labels)
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### S (Specific) Hypotheses")
            S, _ = ce.get_version_space()
            for i, h in enumerate(S):
                st.write(f"Hypothesis {i+1}:", h)
        
        with col2:
            st.markdown("### G (General) Hypotheses")
            _, G = ce.get_version_space()
            for i, h in enumerate(G):
                st.write(f"Hypothesis {i+1}:", h)
        
        # Visualization
        st.subheader("Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ce.visualize_version_space(attribute_names)
        st.pyplot(fig)

# Footer with explanation
st.markdown("""
---
### 📚 Understanding the Algorithm

The Candidate Elimination Algorithm works by maintaining two sets of hypotheses:
1. **S (Specific)**: Contains the most specific hypotheses that match all positive examples
2. **G (General)**: Contains the most general hypotheses that don't match any negative examples

The algorithm updates these sets as it processes each example:
- For positive examples: Generalizes S and removes inconsistent hypotheses from G
- For negative examples: Specializes G and removes inconsistent hypotheses from S

The final version space contains all possible hypotheses that are consistent with the training data.
""") 