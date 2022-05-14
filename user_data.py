import tweepy
import pandas as pd
import spacy
import spacy_ke
from authenticate import auth
from collections import Counter 
from string import punctuation
from preprocessing import remove_emojies, url_free_text
from classify_user_gender import predict_user_gender

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake") 

def user_tweets(user_id):
    response = auth().get_users_tweets(user_id,max_results=100)
    tweets = response.data
    tweets_list = [[tweet.id,tweet.text] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets_list, columns = ['TweetID','Text'])
    return tweets_df


def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result

def get_keywords(tweets_df):
    
    tweets_df['no_emoji'] = tweets_df['Text'].apply(remove_emojies)
    tweets_df['no_url'] = tweets_df['no_emoji'].apply(url_free_text)
    tweets_df['keywords'] = tweets_df['no_url'].apply(get_hotwords)
    return tweets_df


def main():
    
    user_id = input("Enter the user id:")
    tweets_df = user_tweets(user_id)
    tweets_k = get_keywords(tweets_df)
    interests = tweets_k['keywords'].tolist() 
    token = list(set([interest for sublist in interests for interest in sublist]))
    print(token)
    gender = predict_user_gender(token)
    print(gender)
    
    
# calling the main function
if __name__ ==  "__main__":
    main()

