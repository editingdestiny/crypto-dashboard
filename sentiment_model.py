from transformers import pipeline
import pandas as pd
from sentiment_api import news_data
from web_scrapping_gn import all_news
import os
import time
from datetime import datetime, timedelta
from transformers import pipeline

# Load sentiment analysis model
nlp = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone', tokenizer='yiyanghkust/finbert-tone')
FILE_PATH = "sentiment_results.csv"
TIME_FILE = "last_run.txt"

def should_run_analysis():
    if not os.path.exists(TIME_FILE):
        return True  # First-time run

    with open(TIME_FILE, "r") as f:
        last_run = datetime.fromisoformat(f.read().strip())

    return datetime.now() >= last_run + timedelta(hours=2)

if should_run_analysis():
    print("Running sentiment analysis...")

    # Load sentiment model
    nlp = pipeline('sentiment-analysis', model='yiyanghkust/finbert-tone', tokenizer='yiyanghkust/finbert-tone')

    # Process news data
    data = []
    for article in news_data + all_news:
        headline = article.get('headline') or article.get('Title', 'Unknown Title')
        country = article.get('Country', 'Global')  # Default to 'Global' if no country info

        sentiment = nlp(headline)[0]
        data.append({
            'Country': country,
            'Headline': headline,
            'Sentiment': sentiment['label'],
            'Confidence': round(sentiment['score'], 2),
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Track time of analysis
        })

    # Convert to DataFrame & Save as CSV
    df = pd.DataFrame(data)
    df.to_csv(FILE_PATH, index=False)
    print(f"Sentiment analysis complete! Data saved to {FILE_PATH}")

    # Update last run timestamp
    with open(TIME_FILE, "w") as f:
        f.write(datetime.now().isoformat())
else:
    print(f"Skipping analysis. Sentiment data was updated less than 2 hours ago.")