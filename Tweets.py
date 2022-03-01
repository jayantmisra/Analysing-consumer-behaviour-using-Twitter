# importing required libraries

import tweepy
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# function for authentication
def authenticate(api_key,api_secret,access_token,access_token_secret):
    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)  
    print(api.verify_credentials().screen_name)
    return api
    
# function to recieve data
def tweet_data(api, keyword, max_r):
    tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(max_r)
    tweets_list = [[tweet.user.id, tweet.user.name, tweet.id, tweet.user.location, tweet.created_at, tweet.text] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets_list, columns = ['UserID','Name','TweetID','User Location','Date and Time','Text'])
    return tweets_df
    
# function to perform Sentiment Analysis
def sentiments(tweets):
    analyzer = SentimentIntensityAnalyzer()
    tweets['scores'] = tweets['Text'].apply(lambda Text: analyzer.polarity_scores(Text))
    tweets['compound'] = tweets['scores'].apply(lambda score_dict: score_dict['compound'])
    tweets['comp_score'] = tweets['compound'].apply(lambda c: 'positive' if c>=0 else 'negative')
    return tweets

# main method
def main():
    api_key = "YJKQrvmFj4IOSv27nonp8aBGx"
    api_secret = "3wDOUZcAAeTGvH4cNBjgAGYJ2gqYqOEc80rUI3oanGl9igjqbG"
    access_token = "1364148883329220609-WEj8Ijit8g79xor6qCRqJ7pMHAqdIe"
    access_token_secret = "LNZWMyDykX0x2oHe5i8z3dLF1g4lMWQsSszaZDNNJsECE"
    
    keyword = input("Enter the keyword :")
    max_r = input("Enter the max number of tweets required :")
    
    api = authenticate(api_key,api_secret,access_token,access_token_secret)
    tweets = tweet_data(api, keyword, int(max_r))
    
    analysed_tweets = sentiments(tweets)
    print(analysed_tweets)
    
# calling the main function
if __name__ ==  "__main__":
    main()
