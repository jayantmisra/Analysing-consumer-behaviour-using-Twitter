import tweepy
import config

# authenticating using twitter api version 2
def auth():
    client = tweepy.Client(config.bearer_token)
    return client