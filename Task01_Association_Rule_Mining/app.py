import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set page configuration
st.set_page_config(
    page_title="Association Rule Mining Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper functions
def convert_frozenset_to_string(fset):
    """Convert frozenset to sorted string representation."""
    if isinstance(fset, frozenset):
        return ', '.join(sorted(list(fset)))
    return str(fset)

def prepare_rules_for_display(rules_df):
    """Prepare rules DataFrame for display by converting frozensets to strings."""
    display_df = rules_df.copy()
    
    # Convert frozenset columns to strings
    for col in ['antecedents', 'consequents']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(convert_frozenset_to_string)
    
    # Keep only essential columns and round them
    display_df = display_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    display_df[['support', 'confidence', 'lift']] = display_df[['support', 'confidence', 'lift']].round(3)
    
    return display_df

def create_plot_data(rules_df):
    """Create a new DataFrame suitable for plotting."""
    return pd.DataFrame({
        'support': rules_df['support'],
        'confidence': rules_df['confidence'],
        'lift': rules_df['lift'],
        'antecedents': rules_df['antecedents'].apply(convert_frozenset_to_string),
        'consequents': rules_df['consequents'].apply(convert_frozenset_to_string)
    })

# Title and description
st.title("🔍 Association Rule Mining Dashboard")
st.markdown("""
This dashboard helps you discover interesting relationships between items in your dataset using association rule mining.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Add data format instructions
st.sidebar.markdown("""
### 📋 Data Format Instructions

Your CSV file should follow these rules:

1. **File Format**: CSV (Comma-Separated Values)

2. **Data Structure**:
   - Each row represents one transaction
   - Each column represents one item
   - Use 1 for items present in transaction
   - Use 0 for items not present

3. **Example Format**:
```
milk,bread,butter,eggs
1,1,0,1
0,1,1,0
1,0,1,1
```

4. **Alternative Format**:
   - Each row as a list of items
   - Items separated by commas
   - No need for 0s and 1s
```
milk,bread,butter
bread,eggs
milk,eggs,butter
```

5. **Best Practices**:
   - Use clear, consistent item names
   - Avoid special characters
   - Keep item names short
   - Use lowercase letters
   - No spaces in item names

6. **Sample Data**:
   - If no file is uploaded, sample data will be used
   - Sample shows grocery store transactions
""")

# Sample dataset
@st.cache_data
def load_sample_data():
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
    return transactions

# File uploader with better description
uploaded_file = st.sidebar.file_uploader(
    "Upload your dataset (CSV)", 
    type=['csv'],
    help="Upload a CSV file with your transaction data. See format instructions above."
)

# Parameters
min_support = st.sidebar.slider("Minimum Support", 0.1, 0.5, 0.2, 0.05)
min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 0.9, 0.5, 0.05)

# Load data
if uploaded_file is not None:
    # Read uploaded file
    df = pd.read_csv(uploaded_file)
    transactions = df.values.tolist()
else:
    # Use sample data
    transactions = load_sample_data()

# Data preprocessing
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Display raw data
st.subheader("📊 Raw Data")
st.dataframe(df_encoded)

# Generate frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Number of Rules", len(rules))
with col2:
    st.metric("Number of Items", len(te.columns_))
with col3:
    st.metric("Number of Transactions", len(transactions))

# Visualizations
st.subheader("📈 Visualizations")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Top Product Combinations", "Product Connections", "Popular Products"])

with tab1:
    if len(rules) > 0:
        # Create bar chart of top rules by lift
        top_rules = rules.sort_values('lift', ascending=False).head(10)
        
        # Create a more readable format for the rules
        top_rules['rule'] = top_rules.apply(
            lambda x: f"{convert_frozenset_to_string(x['antecedents'])} → {convert_frozenset_to_string(x['consequents'])}", 
            axis=1
        )
        
        # Create the bar chart
        fig = px.bar(
            top_rules,
            x='lift',
            y='rule',
            orientation='h',
            title='Top 10 Product Combinations',
            labels={
                'lift': 'Strength of Relationship',
                'rule': 'Product Combinations'
            },
            color='lift',
            color_continuous_scale='Viridis'
        )
        
        # Update layout for better readability
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            showlegend=False,
            height=500,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        ### Understanding the Bar Chart
        
        This chart shows our strongest product combinations:
        - **Taller bars** = Stronger relationships between products
        - **Top combinations** = Most reliable recommendations
        - **Use this to**: 
          - Plan product placement
          - Create product bundles
          - Design promotions
        """)
    else:
        st.info("No rules found with the current parameters. Try adjusting the support and confidence thresholds.")

with tab2:
    if len(rules) > 0:
        # Create a simplified network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for _, rule in rules.iterrows():
            ant = convert_frozenset_to_string(rule['antecedents'])
            cons = convert_frozenset_to_string(rule['consequents'])
            lift = rule['lift']
            
            # Add edge with lift as weight
            G.add_edge(ant, cons, weight=lift)
        
        # Create the plot with better styling
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes with better styling
        nx.draw_networkx_nodes(G, pos,
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7,
                             edgecolors='white',
                             linewidths=2)
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos,
                             width=[w * 2 for w in edge_weights],
                             alpha=0.6,
                             edge_color='gray')
        
        # Draw labels with better formatting
        nx.draw_networkx_labels(G, pos,
                              font_size=10,
                              font_weight='bold',
                              bbox=dict(facecolor='white', 
                                      edgecolor='none', 
                                      alpha=0.7))
        
        # Add title and explanation
        plt.title("How Products Are Connected\n(Thicker lines = Stronger relationships)", 
                 pad=20, size=14)
        
        # Add simple legend
        plt.figtext(0.02, 0.02, 
                   "How to read this graph:\n"
                   "• Each circle = A product\n"
                   "• Each line = Products bought together\n"
                   "• Thicker line = Stronger relationship\n"
                   "• More connections = More popular product",
                   fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(plt)
        
        # Add explanation
        st.markdown("""
        ### Understanding the Network Graph
        
        This graph shows how products are related to each other:
        - 🟦 **Circles**: Each circle represents a product
        - ➖ **Lines**: Connect products that are often bought together
        - 📏 **Line Thickness**: Shows how strong the relationship is
        
        **How to use it:**
        1. Look for thick lines - these show strong relationships
        2. Products with many connections are popular
        3. Groups of connected products often form natural bundles
        
        **Business insights:**
        - Place connected products close to each other
        - Create bundles for products with strong connections
        - Use popular products to promote less popular ones
        """)
    else:
        st.info("No rules found with the current parameters. Try adjusting the support and confidence thresholds.")

with tab3:
    if len(rules) > 0:
        # Create word cloud of popular items
        all_items = []
        for _, rule in rules.iterrows():
            all_items.extend(list(rule['antecedents']))
            all_items.extend(list(rule['consequents']))
        
        # Count item frequencies
        item_freq = pd.Series(all_items).value_counts()
        
        # Create word cloud with better styling
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=50,
            min_font_size=10,
            max_font_size=100,
            random_state=42
        ).generate_from_frequencies(item_freq)
        
        # Display word cloud
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Popular Products', pad=20, size=14)
        st.pyplot(plt)
        
        # Add explanation
        st.markdown("""
        ### Understanding the Word Cloud
        
        This visualization shows our most popular products:
        - **Bigger words** = More popular products
        - **Different colors** = Different product categories
        - **Use this to**: 
          - Identify best-selling items
          - Plan inventory
          - Focus marketing efforts
        
        **Business insights:**
        - Stock more of the larger products
        - Use popular products to attract customers
        - Consider promotions for smaller products
        """)
    else:
        st.info("No rules found with the current parameters. Try adjusting the support and confidence thresholds.")

# Display rules
st.subheader("📋 Association Rules")
if len(rules) > 0:
    # Format the rules for display
    rules_display = prepare_rules_for_display(rules)
    st.dataframe(rules_display)
else:
    st.info("No rules found with the current parameters. Try adjusting the support and confidence thresholds.")

# Add explanation section
st.subheader("📚 Understanding the Results")
st.markdown("""
- **Support**: How often items appear together (0-1)
- **Confidence**: How reliable the rule is (0-1)
- **Lift**: How much more likely items are to be bought together
- **Rule Format**: If [antecedents] then [consequents]
""") 