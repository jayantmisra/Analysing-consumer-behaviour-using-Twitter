# importing required libraries

import requests
import os
import json

# for preprocessing twitter data
import pandas as pd


#MyBearerToken: AAAAAAAAAAAAAAAAAAAAAGjjVgEAAAAAPswsQesNfrnWbjovEBsb2nHkB8g%3DbUaNGMAsVRxWVWdnHMIgeqYEiYRYq2OOELMChMymnmslznvcMO


# function to recieve bearer token from the user
def enter_bearer_token():
    bt = input("Enter the Bearer Token :");
    return bt


# function to create url 
def create_tweet_url(brand, no_of_tweets):
    max_tweets = "max_results={}".format(no_of_tweets) 
    url = "https://api.twitter.com/1.1/search/tweets.json?q=%40{}&{}".format(brand,max_tweets)
    return url

# for authentication and getting json response for the URL requests
def twitter_auth_response(bt, url):
    header = {"Authorization": "Bearer {}".format(bt)}
    response = requests.request("GET", url, headers=header) 
    return response.json()


# main method
def main():
    brand_name = input("Enter the brand name:")
    no_of_tweets = input("Enter the number of tweets:")
    url = create_tweet_url(brand_name, no_of_tweets)
    
    bearer_token = enter_bearer_token()
    respjson = twitter_auth_response(bearer_token, url)
    print(json.dumps(respjson))

    
# calling the main function
if __name__ == "__main__":
    main()
