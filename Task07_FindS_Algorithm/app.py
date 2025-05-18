import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from find_s_algorithm import FindSAlgorithm

# Set page configuration
st.set_page_config(
    page_title="Find-S Algorithm: Learning Made Simple",
    page_icon="🎯",
    layout="wide"
)

# Add a friendly header
st.title("🎯 Find-S Algorithm: Learning Made Simple")
st.markdown("""
### What is Find-S Algorithm?
The Find-S algorithm is like a smart student who learns from examples. It starts with no knowledge and gradually learns patterns from positive examples.

Think of it like learning to identify a good day for playing sports:
- It looks at different features like weather, temperature, etc.
- It learns from examples of good days for sports
- It creates rules to identify similar good days in the future
""")

# Add a sidebar with explanation
with st.sidebar:
    st.header("📚 How It Works")
    st.markdown("""
    1. **Start with No Knowledge**
       - Initially, the algorithm knows nothing
       - It's like a blank slate
    
    2. **Learn from Examples**
       - It only learns from positive examples (good days for sports)
       - It ignores negative examples (bad days)
    
    3. **Create Rules**
       - It creates specific rules based on what it sees
       - If it sees different values for the same feature, it becomes flexible (marked as '?')
    
    4. **Make Predictions**
       - Uses the learned rules to predict if a new day is good for sports
    """)

# Example data
st.header("🎮 Let's See It in Action!")

# Training data with friendly labels
attributes = ['Sky', 'Air Temperature', 'Humidity', 'Wind', 'Water Temperature', 'Forecast']
possible_values = [
    ['Sunny', 'Rainy'],
    ['Warm', 'Cold'],
    ['Normal', 'High'],
    ['Strong', 'Weak'],
    ['Warm', 'Cool'],
    ['Same', 'Change']
]

# Training data with friendly labels
training_data = [
    (['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], 'Yes'),
    (['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], 'No'),
    (['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], 'Yes')
]

# Create and train the model
find_s = FindSAlgorithm(attributes, possible_values)
find_s.train(training_data)

# Display training data in a user-friendly way
st.subheader("📊 Our Training Examples")
st.markdown("""
These are the examples we'll use to teach our algorithm. Each row represents a day, and we know whether it was good for sports or not.
""")

training_df = pd.DataFrame([
    {**dict(zip(attributes, example)), 'Good for Sports?': label}
    for example, label in training_data
])
st.dataframe(training_df, use_container_width=True)

# Display hypothesis evolution in a simpler way
st.subheader("🔄 How the Algorithm Learns")
st.markdown("""
Watch how the algorithm's understanding changes as it sees more examples. The '?' means the algorithm learned that this feature can be flexible.
""")

evolution_data = []
for i, hyp in enumerate(find_s.hypothesis_history):
    for j, (attr, val) in enumerate(zip(attributes, hyp)):
        evolution_data.append({
            'Learning Step': i,
            'Feature': attr,
            'Learned Value': val
        })
evolution_df = pd.DataFrame(evolution_data)

# Create a simpler heatmap
fig = px.density_heatmap(
    evolution_df,
    x='Feature',
    y='Learning Step',
    z='Learned Value',
    color_continuous_scale='Viridis',
    title='Learning Progress'
)
st.plotly_chart(fig, use_container_width=True)

# Display final hypothesis in a friendly way
st.subheader("🎯 What the Algorithm Learned")
st.markdown("""
This is what the algorithm learned about good days for sports. The '?' means that feature can be flexible.
""")

final_hypothesis = pd.DataFrame({
    'Feature': attributes,
    'Learned Rule': find_s.hypothesis
})
st.dataframe(final_hypothesis, use_container_width=True)

# Create a simple bar plot
fig = go.Figure(data=[
    go.Bar(
        x=attributes,
        y=[1] * len(attributes),
        text=find_s.hypothesis,
        textposition='auto',
        marker_color=['green' if val == '?' else 'blue' for val in find_s.hypothesis]
    )
])
fig.update_layout(
    title='Final Rules Visualization',
    showlegend=False,
    yaxis_showticklabels=False
)
st.plotly_chart(fig, use_container_width=True)

# Add interactive test section
st.subheader("🔮 Test Your Own Data!")
st.markdown("""
### How to Test Your Own Data
1. For each feature, select a value from the dropdown menu
2. Click 'Predict' to see if it would be a good day for sports
3. The algorithm will use its learned rules to make a prediction

**Note:** Make sure to select valid values for each feature:
- Sky: Sunny or Rainy
- Air Temperature: Warm or Cold
- Humidity: Normal or High
- Wind: Strong or Weak
- Water Temperature: Warm or Cool
- Forecast: Same or Change
""")

# Create columns for the input form
col1, col2 = st.columns(2)

# Create input fields for each feature
user_input = {}
with col1:
    user_input['Sky'] = st.selectbox('Sky', ['Sunny', 'Rainy'])
    user_input['Air Temperature'] = st.selectbox('Air Temperature', ['Warm', 'Cold'])
    user_input['Humidity'] = st.selectbox('Humidity', ['Normal', 'High'])

with col2:
    user_input['Wind'] = st.selectbox('Wind', ['Strong', 'Weak'])
    user_input['Water Temperature'] = st.selectbox('Water Temperature', ['Warm', 'Cool'])
    user_input['Forecast'] = st.selectbox('Forecast', ['Same', 'Change'])

# Create a button to trigger prediction
if st.button('Predict', type='primary'):
    # Convert user input to list in the correct order
    test_example = [user_input[attr] for attr in attributes]
    
    # Get prediction
    prediction = find_s.predict(test_example)
    
    # Display the test example
    st.subheader("Your Test Data")
    test_df = pd.DataFrame({
        'Feature': attributes,
        'Selected Value': test_example
    })
    st.dataframe(test_df, use_container_width=True)
    
    # Display prediction with explanation
    st.subheader("Prediction Result")
    if prediction == 'Yes':
        st.success("""
        🎉 This would be a good day for sports!
        
        The algorithm predicts this because your selected values match the learned rules for good sports days.
        """)
    else:
        st.error("""
        ❌ This would not be a good day for sports.
        
        The algorithm predicts this because your selected values don't match the learned rules for good sports days.
        """)
    
    # Show which rules were matched or not matched
    st.subheader("Rule Matching Analysis")
    rule_matches = []
    for attr, val in zip(attributes, test_example):
        if find_s.hypothesis[attributes.index(attr)] == '?':
            rule_matches.append(f"✅ {attr}: Flexible rule (?)")
        elif find_s.hypothesis[attributes.index(attr)] == val:
            rule_matches.append(f"✅ {attr}: Matches rule")
        else:
            rule_matches.append(f"❌ {attr}: Doesn't match rule")
    
    for match in rule_matches:
        st.write(match)

# Add a conclusion
st.subheader("🎓 Summary")
st.markdown("""
The Find-S algorithm is a simple but powerful way to learn from examples. It:
- Starts with no knowledge
- Learns from positive examples
- Creates rules to make predictions
- Can be used to solve many real-world problems

Try different combinations of weather conditions to see how the algorithm's predictions change!
""") 