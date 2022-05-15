import tweepy
import pandas as pd
import spacy
import spacy_ke
from googletrans import Translator
from authenticate import auth
from collections import Counter 
from string import punctuation
from preprocessing import remove_emojies, url_free_text
import re, os
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet 
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("yake") 

#Loading pre processed training data
tweets = pd.read_csv('preprocessed_training_data.csv')

def predict_user_gender(token):
    text_transformer = TfidfVectorizer()
    print("Training the classifier to predict user's gender...")
    X = text_transformer.fit_transform(tweets['token'])
    y = tweets['gender']

    # Splitting data into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=39)

    # Train using Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print("Classifier trained!")
   
    token_data = " ".join(str(x) for x in token)
    test_data = {'screenname':['person'],'token':[token_data]}

    # test_tweets_df = pd.read_json(TEST_TWEET_FILE)
    test_tweets_df = pd.DataFrame.from_dict(test_data)
    #print(test_tweets_df)

    # Preprocess tweets
    test_tweets = test_tweets_df[test_tweets_df['token'].apply(len)>20]
    test_tweets = test_tweets.reset_index(drop=True)
    # TF-IDF features
    bow = text_transformer.transform(test_tweets['token'])
    df_bow_test = pd.DataFrame(bow.todense(), columns=text_transformer.get_feature_names())
    # Predict probability
    pred_prob = pd.DataFrame(logreg.predict_proba(df_bow_test))
    # Predict classification
    pred = pd.DataFrame(data=logreg.predict(df_bow_test), columns=['pred'])
    # Merge into the same DataFrame
    result = pd.concat([test_tweets, pred, pred_prob], axis=1, sort=False)
    #print(result['pred'])
    return result['pred']

def user_tweets(user_id):
    response = auth().get_users_tweets(user_id,max_results=100)
    tweets = response.data
    tweets_list = [[tweet.id,tweet.text] for tweet in tweets]
    tweets_df = pd.DataFrame(tweets_list, columns = ['TweetID','Text'])
    #Translates tweets into english
    translator = Translator()
    for column in range(0,tweets_df['Text'].size):
        tweets_df.loc[column,'Text'] = translator.translate(tweets_df['Text'].iloc[column]).text
    
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


def main(user_id):
    

    tweets_df = user_tweets(user_id)
    tweets_k = get_keywords(tweets_df)
    interests = tweets_k['keywords'].tolist() 
    token = list(set([interest for sublist in interests for interest in sublist]))
    gender = predict_user_gender(token)
    gender = gender.to_string().split()
    return gender[1], interests
    
# calling the main function
if __name__ ==  "__main__":
    user_id = "1364148883329220609" # male
    user_id = "1040587167230124032" # female
    
    #user_id = input("Enter the user id:")
    main(user_id)
