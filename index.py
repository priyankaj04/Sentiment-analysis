import nltk
import os
import ssl
from nltk.sentiment import SentimentIntensityAnalyzer

# Set SSL certificate verification to False
ssl._create_default_https_context = ssl._create_unverified_context

# Download the necessary NLTK data
nltk.download('vader_lexicon')

# Get the path to the vader_lexicon folder
nltk_data_path = os.path.join(nltk.data.path[0], 'vader_lexicon')
# print(nltk_data_path)

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Define the sentence
sentence = open('read.txt').read()

# Perform sentiment analysis
scores = sia.polarity_scores(sentence)

# Print the sentiment scores
print(scores)