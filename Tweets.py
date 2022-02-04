# This code is just written to take an idea on how to fetch tweets
# Unlike this code, the final implementation would not use Tweepy library and tweets will be stored in SQL database


import tweepy
import pandas as pd
import csv
#authentication

# I removed the keys and tokens 
api_key = ""
api_secret = ""
access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

#printing the Username of the developer after authentication
print(api.verify_credentials().screen_name)

product_keyword = 'Uber'
no_of_tweets = 100

tweets = tweepy.Cursor(api.search_tweets, q=product_keyword).items(no_of_tweets)

tweets_list = [[tweet.user.id, tweet.user.name, tweet.id, tweet.user.location, tweet.created_at, tweet.text] for tweet in tweets]

tweets_df = pd.DataFrame(tweets_list, columns = ['UserID','Name','TweetID','User Location','Date and Time','Text'])

#converting the dataframe to csv file
tweets_df.to_csv('test.csv')