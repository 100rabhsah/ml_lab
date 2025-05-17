import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Create a sample dataset (you can replace this with your own dataset)
# This is a sample of grocery store transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'diapers'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['milk', 'bread', 'diapers'],
    ['bread', 'diapers', 'beer'],
    ['milk', 'bread', 'diapers', 'beer'],
    ['bread', 'milk'],
    ['milk', 'diapers', 'beer', 'cola'],
    ['bread', 'milk', 'diapers', 'beer'],
    ['bread', 'milk', 'diapers']
]

# Data Preprocessing
# Convert transactions to a format suitable for association rule mining
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Display the preprocessed data
print("\nPreprocessed Data:")
print(df.head())

# Perform Association Rule Mining
# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:")
print(rules)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Support vs Confidence')
plt.show()

# Create a heatmap of the rules
plt.figure(figsize=(10, 6))
sns.heatmap(rules[['support', 'confidence', 'lift']], annot=True, cmap='YlOrRd')
plt.title('Association Rules Heatmap')
plt.show()

# Print the top 5 rules by lift
print("\nTop 5 Rules by Lift:")
print(rules.sort_values('lift', ascending=False).head()) 