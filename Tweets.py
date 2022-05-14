# importing required libraries

import tweepy
import pandas as pd
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import config
# function for authentication


def authenticate(api_key, api_secret, access_token, access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    print(api.verify_credentials().screen_name)
    return api

# function to recieve data


def tweet_data(api, keyword, max_r):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(max_r)
    # Changes can be made here to get more tweet fields
    tweets_list = [[tweet.user.id, tweet.user.name, tweet.id,
                    tweet.user.location, tweet.created_at, tweet.text] for tweet in tweets]
    # Converting the data to Pandas DataFrame
    tweets_df = pd.DataFrame(tweets_list, columns=[
                             'UserID', 'Name', 'TweetID', 'User Location', 'Date and Time', 'Text'])
    return tweets_df

# function to perform Sentiment Analysis


def sentiments(tweets):
    
    #Translates all of the tweets into english for the sentiment analysis
    translator = Translator()
    for column in range(0,tweets['Text'].size):
        tweets.loc[column,'Text'] = translator.translate(tweets['Text'].iloc[column]).text
    
    analyzer = SentimentIntensityAnalyzer()    
    tweets['scores'] = tweets['Text'].apply(
        lambda Text: analyzer.polarity_scores(Text))
    tweets['compound'] = tweets['scores'].apply(
        lambda score_dict: score_dict['compound'])
    # If the compound score is less than 0 it's negative, else positive
    tweets['comp_score'] = tweets['compound'].apply(
        lambda c: 'positive' if c >= 0 else 'negative')
    return tweets

# main method


def main():
    
    # To take name of the brand and max number of tweets from the user
    keyword = input("Enter the keyword :")
    max_r = input("Enter the max number of tweets required :")

    api = authenticate(config.api_key, config.api_secret, config.access_token, config.access_token_secret)
    tweets = tweet_data(api, keyword, int(max_r))
    # print(tweets['User Location'])
    analysed_tweets = sentiments(tweets)
    print(analysed_tweets)
    # 'analysed_tweets' is the final data yet


# calling the main function
if __name__ == "__main__":
    main()
