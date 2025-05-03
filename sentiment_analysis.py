import pandas as pd
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

#load the dataset
df = pd.read_csv("training.1600000.processed.noemoticon.csv", 
                 encoding='latin-1', 
                 names=["target", "id", "date", "flag", "user", "text"])

df = df[["target", "text"]]

#convert target to labels
def convert_target(t):
    if t == 0:
        return "Negative"
    elif t == 2:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["target"].apply(convert_target)

#clean the tweet text
import re

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)

#Re-analyze Sentiment Using TextBlob
df["polarity"] = df["clean_text"].apply(lambda x: TextBlob(x).sentiment.polarity)



#visualize the sentiment distribution
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Distribution in Sentiment140 Dataset")
plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

#save processed dataset
df.to_csv("cleaned_sentiment140.csv", index=False)
