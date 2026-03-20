from transformers import pipeline
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime

# Load the sentiment analysis model
analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

# Load the merged news data
with open('data/processed/apple_news_data.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

# Helper function to split long text into chunks of 512 tokens
def split_into_chunks(text, max_tokens=512):
    return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]

# Data structure to store daily sentiment metrics
sentiment_summary = {}

# Analyze each news article and aggregate results by day
for article in tqdm(news_data):
    date = article['date'].split("T")[0]  # Extract date from timestamp
    title = article['title']
    content = article['content']
    text = title + " " + content
    
    # Split long texts into chunks
    text_chunks = split_into_chunks(text)
    
    # Initialize metrics for this article
    total_score = 0
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    
    # Perform sentiment analysis on each chunk and aggregate the results
    for chunk in text_chunks:
        sentiment_result = analyzer(chunk)[0]  # Get the first result from the list
        
        # Extract sentiment label and score
        label = sentiment_result['label']
        score = sentiment_result['score']
        total_score += score
        
        # Count sentiment labels
        if label == 'positive':
            positive_count += 1
        elif label == 'neutral':
            neutral_count += 1
        elif label == 'negative':
            negative_count += 1
    
    # Get the average sentiment score for the article
    avg_score = total_score / len(text_chunks)
    
    # Aggregate sentiment data per day
    if date not in sentiment_summary:
        sentiment_summary[date] = {
            'total_articles': 0,
            'positive_articles': 0,
            'neutral_articles': 0,
            'negative_articles': 0,
            'total_score': 0,
            'total_chunks': 0  # Number of chunks analyzed
        }
    
    # Update daily metrics
    sentiment_summary[date]['total_articles'] += 1
    sentiment_summary[date]['positive_articles'] += positive_count
    sentiment_summary[date]['neutral_articles'] += neutral_count
    sentiment_summary[date]['negative_articles'] += negative_count
    sentiment_summary[date]['total_score'] += total_score
    sentiment_summary[date]['total_chunks'] += len(text_chunks)

# Prepare results for CSV/JSON
daily_summary = []
for date, metrics in sentiment_summary.items():
    daily_summary.append({
        'date': date,
        'total_articles': metrics['total_articles'],
        'positive_articles': metrics['positive_articles'],
        'neutral_articles': metrics['neutral_articles'],
        'negative_articles': metrics['negative_articles'],
        'average_sentiment_score': metrics['total_score'] / metrics['total_chunks'] if metrics['total_chunks'] > 0 else 0
    })

# Convert to DataFrame
df_summary = pd.DataFrame(daily_summary)

# Save the daily summary to a CSV file
df_summary.to_csv('sentiment_daily_summary.csv', index=False)

# Optionally save as JSON
df_summary.to_json('sentiment_daily_summary.json', orient='records', indent=4)

print("Sentiment analysis complete. Results saved to 'sentiment_daily_summary.csv' and 'sentiment_daily_summary.json'.")
