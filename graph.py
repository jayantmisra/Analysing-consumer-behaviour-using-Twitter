import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('tweets_data.csv')

pos=df['Date and Time'][df['compound']>0.0]
nopn=df['Date and Time'][df['compound']==0.0]
neg=df['Date and Time'][df['compound']<0.0]
plt.hist([pos, nopn, neg],
     stacked=False,
     label=["positive", "no opinion", "negative"])

plt.legend()
plt.title("Sentiment Analysis: Uber")
plt.xlabel("Dates")
plt.ylabel("No. of tweets")
plt.show()