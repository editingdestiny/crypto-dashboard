#%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
#plt.use('TkAgg') 
import seaborn as sns
import os

plt.ion()  # Interactive mode ON
# Load the latest sentiment results
FILE_PATH = "sentiment_results.csv"

if not os.path.exists(FILE_PATH):
    print("Error: Sentiment results not found. Run sentiment_analysis.py first.")
    exit()

df = pd.read_csv(FILE_PATH)

# Set stylepip
plt.style.use("ggplot")

#1️⃣ Fixed Sentiment Distribution (Bar Chart)
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette="coolwarm", legend=False)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show(block = True)
#print(df.head())

# 2️⃣ Fixed Sentiment Confidence (Box Plot)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Sentiment', y='Confidence', hue='Sentiment', palette="coolwarm", legend=False)
plt.title("Confidence Score Distribution per Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Confidence Score")
plt.show(block = True)
plt.savefig("sentiment_distribution.png")
plt.savefig("confidence_distribution.png")

# 2️⃣ **Stacked Bar Chart (Alternative)**
sentiment_counts = df.groupby(["Country", "Sentiment"]).size().unstack()

sentiment_counts.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
plt.title("Stacked Sentiment Distribution by Country")
plt.xlabel("Country")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(title="Sentiment")
plt.show()