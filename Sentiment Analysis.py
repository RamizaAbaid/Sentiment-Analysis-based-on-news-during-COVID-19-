# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 20:20:31 2024

@author: user
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
## loading data
# Load the dataset
df = pd.read_csv('C:/Users/user/Documents/reports/COVID19_Articles_2019_2020_2021.csv')  # Replace with your file path

# Display the first few rows of the DataFrame
print(df.head())
## Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    return sentiment_scores
## Applying sentiment analyzer to data
# Assuming the text of the articles is in a column named 'article_text'
df['sentiment'] = df['Summary'].apply(analyze_sentiment)

# Convert the sentiment scores into separate columns
df_sentiment = df['sentiment'].apply(pd.Series)

# Combine the original DataFrame with the sentiment scores
df = pd.concat([df, df_sentiment], axis=1)

# Display the updated DataFrame
print(df.head())
##Analyzing the results
# Calculate average sentiment scores
average_sentiment = df[['neg', 'neu', 'pos', 'compound']].mean()
print("Average Sentiment Scores:")
print(average_sentiment)

# Optional: Visualize the sentiment distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(df['sentiment'].apply(lambda x: x['compound']), bins=30, color='blue', alpha=0.7)
plt.title('Sentiment Score Distribution')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')
plt.grid()
plt.show()