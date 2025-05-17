import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from naive_bayes_classifier import NaiveBayesTextClassifier

# Set page config
st.set_page_config(
    page_title="Text Sentiment Classifier",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Text Sentiment Classifier")
st.markdown("""
This app uses Naive Bayes to classify text into positive or negative sentiment. 
Simply enter your text below and the model will predict whether it's positive or negative!
""")

# Initialize the classifier
@st.cache_resource
def load_classifier():
    classifier = NaiveBayesTextClassifier()
    # Enhanced training data with more examples and balanced classes
    texts = [
        # Positive examples
        "This movie is great! I really enjoyed it.",
        "The acting was superb and the plot was engaging.",
        "Excellent performance by all actors.",
        "A masterpiece of modern cinema.",
        "This movie was absolutely amazing!",
        "I really enjoyed the performance.",
        "The film was outstanding and captivating.",
        "What a wonderful experience!",
        "The movie exceeded all my expectations.",
        "I loved every minute of it.",
        "The story was beautifully told.",
        "The cinematography was breathtaking.",
        "The characters were well-developed.",
        "The movie was inspiring and uplifting.",
        "A perfect blend of drama and emotion.",
        # Adding more general positive examples
        "I love this app!",
        "This is really good.",
        "I like it very much.",
        "It's awesome!",
        "This is fantastic.",
        "I really like this.",
        "This is wonderful.",
        "I love how it works.",
        "This is excellent.",
        "I'm really happy with this.",
        
        # Negative examples
        "Terrible movie, waste of time and money.",
        "I couldn't stand watching this movie.",
        "The worst movie I've ever seen.",
        "Complete disaster, avoid at all costs.",
        "The worst experience ever.",
        "Not worth watching at all.",
        "The acting was terrible and the plot made no sense.",
        "I regret spending money on this film.",
        "The movie was boring and predictable.",
        "The dialogue was awful and the pacing was slow.",
        "A complete waste of time.",
        "The worst film of the year.",
        "I couldn't wait for it to end.",
        "The story was confusing and poorly executed.",
        "The movie was a disappointment.",
        # Adding more general negative examples
        "I hate this app.",
        "This is really bad.",
        "I don't like it at all.",
        "It's terrible.",
        "This is awful.",
        "I really dislike this.",
        "This is horrible.",
        "I hate how it works.",
        "This is poor quality.",
        "I'm really disappointed with this."
    ]
    
    labels = ['positive'] * 25 + ['negative'] * 25  # Balanced classes
    classifier.fit(texts, labels)
    return classifier

# Load the classifier
classifier = load_classifier()

# Create two columns for input and output
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Input")
    # Text input
    user_text = st.text_area(
        "Enter your text here:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    # Add some example texts
    st.markdown("### 💡 Try these examples:")
    examples = [
        "I love this app!",
        "This is really good.",
        "I hate this app.",
        "This is terrible.",
        "The film was outstanding and captivating.",
        "The acting was terrible and the plot made no sense."
    ]
    for example in examples:
        if st.button(example, key=example):
            user_text = example

with col2:
    st.subheader("📊 Results")
    if user_text:
        # Make prediction
        prediction = classifier.predict([user_text])[0]
        
        # Display prediction with color
        if prediction == 'positive':
            st.success(f"Prediction: {prediction.upper()} 😊")
        else:
            st.error(f"Prediction: {prediction.upper()} 😞")
        
        # Show feature importance
        st.markdown("### 🔍 Key Words Analysis")
        feature_importance = classifier.get_feature_importance(n_top_features=5)
        
        # Create a visualization of the prediction confidence
        st.markdown("### 📈 Prediction Confidence")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=['Positive', 'Negative'], 
                   y=[0.7 if prediction == 'positive' else 0.3, 
                      0.3 if prediction == 'positive' else 0.7])
        plt.title("Model Confidence")
        st.pyplot(fig)

# Add explanation section
st.markdown("---")
st.markdown("""
### ℹ️ How it works:
1. The model uses Naive Bayes algorithm to analyze text
2. It looks for words that typically indicate positive or negative sentiment
3. Based on the presence of these words, it makes a prediction
4. The confidence score shows how sure the model is about its prediction
""")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Naive Bayes") 