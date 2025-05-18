import streamlit as st
from PIL import Image
import os

def load_image(image_path):
    """Load image with error handling."""
    try:
        if not os.path.exists(image_path):
            st.error(f"Image file not found: {image_path}")
            return None
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image {image_path}: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Visualization",
        page_icon="😊",
        layout="wide"
    )

    st.title("🎭 Movie Review Sentiment Analysis")
    st.markdown("""
    This app helps you understand how our AI models analyze movie reviews to determine if they're positive or negative.
    We used two different types of models: RNN and LSTM, which are special types of AI that are good at understanding text.
    """)

    # Introduction
    st.header("📝 What is Sentiment Analysis?")
    st.markdown("""
    Sentiment Analysis is like teaching a computer to understand if someone is happy or sad when they write something.
    In our case, we're teaching the computer to understand if people liked or disliked a movie based on their reviews.
    """)

    # Model Explanation
    st.header("🤖 Our Models")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("RNN (Recurrent Neural Network)")
        st.markdown("""
        - Think of RNN as a computer that reads text one word at a time
        - It remembers what it read before while reading new words
        - Like a person reading a book and remembering the story
        """)
    with col2:
        st.subheader("LSTM (Long Short-Term Memory)")
        st.markdown("""
        - LSTM is like an improved version of RNN
        - It's better at remembering important information
        - Like a person who can remember the main plot points of a movie
        """)

    # RNN Results
    st.header("📊 RNN Model Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Progress")
        rnn_training = load_image('figure1.png')
        if rnn_training:
            st.image(rnn_training, caption="RNN Training Loss and Accuracy")
            st.markdown("""
            This graph shows how the RNN model learned:
            - The blue line shows how well it learned from training data
            - The orange line shows how well it performs on new data
            - Lower loss and higher accuracy means better performance
            """)
    with col2:
        st.subheader("Confusion Matrix")
        rnn_confusion = load_image('figure2.png')
        if rnn_confusion:
            st.image(rnn_confusion, caption="RNN Confusion Matrix")
            st.markdown("""
            This matrix shows how accurate the RNN model is:
            - Top left: Correctly identified negative reviews
            - Bottom right: Correctly identified positive reviews
            - Other boxes: Where the model made mistakes
            """)

    # LSTM Results
    st.header("📈 LSTM Model Results")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Progress")
        lstm_training = load_image('figure3.png')
        if lstm_training:
            st.image(lstm_training, caption="LSTM Training Loss and Accuracy")
            st.markdown("""
            This graph shows how the LSTM model learned:
            - The blue line shows how well it learned from training data
            - The orange line shows how well it performs on new data
            - Lower loss and higher accuracy means better performance
            """)
    with col2:
        st.subheader("Confusion Matrix")
        lstm_confusion = load_image('figure4.png')
        if lstm_confusion:
            st.image(lstm_confusion, caption="LSTM Confusion Matrix")
            st.markdown("""
            This matrix shows how accurate the LSTM model is:
            - Top left: Correctly identified negative reviews
            - Bottom right: Correctly identified positive reviews
            - Other boxes: Where the model made mistakes
            """)

    # Model Comparison
    st.header("🔄 Model Comparison")
    st.markdown("""
    ### Which model is better?
    - Both models are good at understanding movie reviews
    - LSTM generally performs better because it can remember important information for longer
    - The confusion matrices show how many reviews each model got right or wrong
    - The training graphs show how quickly each model learned
    """)

    # How to Use
    st.header("💡 How to Use This Analysis")
    st.markdown("""
    1. Look at the training graphs to see how well each model learned
    2. Check the confusion matrices to see how accurate they are
    3. Compare the results between RNN and LSTM
    4. Remember: No model is perfect, but they can help us understand general sentiment
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    ### About This Project
    This sentiment analysis project uses AI to understand movie reviews. It's like teaching a computer to read and understand
    how people feel about movies, just like a human would. The models learn from thousands of real movie reviews to get better
    at understanding new reviews.
    """)

if __name__ == "__main__":
    main() 