import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data at module level
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

# Download NLTK data when module is imported
download_nltk_data()

class NaiveBayesTextClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.classifier = MultinomialNB()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Preprocess the input text by tokenizing and removing stopwords."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Split into words
        tokens = text.split()
        
        # Remove stopwords and non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        
        return ' '.join(tokens)
    
    def fit(self, texts, labels):
        """Train the Naive Bayes classifier."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Convert texts to feature vectors
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train the classifier
        self.classifier.fit(X, labels)
    
    def predict(self, texts):
        """Predict the class labels for new texts."""
        # Preprocess the texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Convert texts to feature vectors
        X = self.vectorizer.transform(processed_texts)
        
        # Make predictions
        return self.classifier.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the classifier's performance."""
        # Make predictions on test data
        y_pred = self.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Create and plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def get_feature_importance(self, n_top_features=10):
        """Get the most important features for each class."""
        feature_names = self.vectorizer.get_feature_names_out()
        classes = self.classifier.classes_
        
        for i, class_label in enumerate(classes):
            # Get the log probabilities for this class
            log_probs = self.classifier.feature_log_prob_[i]
            
            # Get the indices of the top features
            top_indices = np.argsort(log_probs)[-n_top_features:]
            
            print(f"\nTop {n_top_features} features for class '{class_label}':")
            for idx in reversed(top_indices):
                print(f"{feature_names[idx]}: {np.exp(log_probs[idx]):.4f}")

def main():
    # Example usage
    # Sample data (you should replace this with your actual dataset)
    texts = [
        "This movie is great! I really enjoyed it.",
        "Terrible movie, waste of time and money.",
        "The acting was superb and the plot was engaging.",
        "I couldn't stand watching this movie.",
        "Excellent performance by all actors.",
        "The worst movie I've ever seen.",
        "A masterpiece of modern cinema.",
        "Complete disaster, avoid at all costs."
    ]
    
    labels = ['positive', 'negative', 'positive', 'negative', 
              'positive', 'negative', 'positive', 'negative']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create and train the classifier
    classifier = NaiveBayesTextClassifier()
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier
    classifier.evaluate(X_test, y_test)
    
    # Show feature importance
    classifier.get_feature_importance(n_top_features=5)
    
    # Example prediction
    new_texts = [
        "This film was absolutely amazing!",
        "I regret watching this movie."
    ]
    predictions = classifier.predict(new_texts)
    print("\nPredictions for new texts:")
    for text, pred in zip(new_texts, predictions):
        print(f"Text: {text}")
        print(f"Predicted class: {pred}\n")

if __name__ == "__main__":
    main() 