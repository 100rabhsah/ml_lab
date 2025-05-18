# RNN and LSTM for Sentiment Analysis

This project implements sentiment analysis on movie reviews using both Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. The implementation uses the IMDB dataset, which contains movie reviews labeled as positive or negative.

## Project Structure
- `sentiment_analysis.py`: Main implementation file containing RNN and LSTM models
- `utils.py`: Utility functions for data preprocessing and visualization
- `requirements.txt`: Required Python packages

## Setup Instructions
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the implementation:
```bash
python sentiment_analysis.py
```

## Features
- Data preprocessing and text tokenization
- Implementation of both RNN and LSTM models
- Model training and evaluation
- Performance comparison between RNN and LSTM
- Visualization of training metrics

## Model Architecture
- RNN Model: Simple recurrent neural network with embedding layer
- LSTM Model: LSTM network with embedding layer and dropout for regularization

## Dataset
The implementation uses the IMDB dataset, which contains:
- 50,000 movie reviews
- Binary classification (positive/negative)
- Pre-split into training and testing sets 