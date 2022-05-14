# Import libraries
import pandas as pd
import re, os
from collections import Counter
import nltk
from nltk.corpus import stopwords, wordnet 
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, confusion_matrix
# File paths and names
DATA_PATH = r'./data/'
TWITTER_ACCOUNTS_FILE = DATA_PATH + 'celebrity_twitter_accounts.csv'
ORIG_TWEET_FILE = DATA_PATH + 'all_tweets.json'
additional_stop_words=['twitter','com','via']
# Load training celebrity twitter accounts
celebrity_twitter_accounts = pd.read_csv(TWITTER_ACCOUNTS_FILE)
# Load tweets for training
#all_tweets_df = pd.read_json(ORIG_TWEET_FILE)
tweets = pd.read_csv('preprocessed_training_data.csv')

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_cleanup(text):  
    '''
    Text pre-processing
        return tokenized list of cleaned words
    '''
    # Convert to lowercase
    text_clean = text.lower()
    # Remove non-alphabet
    text_clean = re.sub(r'[^a-zA-Z]|(\w+:\/\/\S+)',' ', text_clean).split()    
    # Remove short words (length < 3)
    text_clean = [w for w in text_clean if len(w)>2]
    # Lemmatize text with the appropriate POS tag
    lemmatizer = WordNetLemmatizer()
    text_clean = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in text_clean]
    # Filter out stop words in English 
    stops = set(stopwords.words('english')).union(additional_stop_words)
    text_clean = ' '.join([w for w in text_clean if w not in stops])
    
    return text_clean

# def preprocess_tweets(all_tweets_df):

#     all_tweets_df['token'] = [text_cleanup(x) for x in all_tweets_df['text']] 
#     # Only take processed tweets that have more than 20 characters (too short tweet has little meaning)
#     tweets = all_tweets_df[all_tweets_df['token'].apply(len)>20]
#     tweets = tweets.reset_index(drop=True)
#     return tweets

# Term Frequency-Inverse Document Frequency (TF-IDF) features
# def train_gender_classifier(tweets):
#     X = text_transformer.fit_transform(tweets['token'])
#     y = tweets['gender']

#     # Splitting data into training and validation set
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=39)

#     # Train using Logistic Regression
#     logreg = LogisticRegression()
#     logreg.fit(X_train, y_train)
#     return logreg

# print("Preprocessing training data for gender classification...")
# tweets = preprocess_tweets(all_tweets_df)
# tweets.to_csv('preprocessed_training_data.csv')


# print("Training the classifier to predict user's gender...")
# logreg = train_gender_classifier(tweets)
# print("Classifier trained!")

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
    # Load test tweets
    #['loki', 'fascism', 'quantum', 'interdisciplinary', 'moon', 'man', 'tmc', 'treasure', 'goons', 'rajiv', 'session', 'asteroids', 'messaging', 'normal', 'classical', 'real', 'events', 'nahi', 'birth', 'invitation', '@cucompscistem', 'blue', 'redirec', '@rickandmorty', 'fascinating', '@xkcdcomic', 'impact', 'janata', 'journalists', 'nice', 'cardiff', 'sessions', 'fr', '@amitshah', 's', 'qna', 'wrong', 'disney', 'able', 'person', 'computing', 'people', 'good', 'dharna', '@spacex', '3rd', 'dharne', 'willis', 'govt', 'damn', 'dinosaurs', 'lectures', 'twitter', 'phone', 'kashmiri', 'year', 'way', 'influencers', 'hunt', 'aukaat', 'code', 'sure', 'gandhi', 'bengal', 'life', '@cucybersoc', 'vps', 'attacks', 'b*tch', 'bruce', 'rt', 'superstar', 'mars', 'jupiter', 'back!', 'government', 'karlenge', '@awokeaware', 'powerless', 'months', 'autocorrect', 'saturn', 'bro', 'ab', 'universe', 'spinthariscope', 'awkward', 'entertaining', 'ki', 'spring', 'romance', 'yesterday', '@qmunitytech', 'best', '@nobelprize', '@elonmusk', 'tonight', '@ixfoduap', 'exodus', 'undergraduate', 'responsible', 'bjp', 'atari', 'group', 'nasa', 'kickoff', 'live', 'control', '@amiramorphism', 'long', 'time', 'type', 'blackberry', 'thing', '@theanushcasm', 'machine', 'movie', '@xkcd', 'saver', '@qiskit', 'fun', '@mahuamoitra', 'sunset', 'field', 'chrome', '@amoghgeorge', 'url', 'bolne', 'dal', 'telephoto', 'bengalviolence2021', 'technological', 'half', 'hai', 'pandits', 'dragon', 'action', 'men', 'aap', 'congrats', 'singh', 'ke', 'computers', 'falcon', '@lexfridman', 'nevermind', 'pm', 'tweet', 'amazing', 'structured', 'hahaha', 'dec', '@bjp4india', 'vp', 'impaired', 'thanks', 'qgss21', 'great', 'audacity', 'qr', 'fault', '2nd', '@devchhatbar', '@pmoindia', 'panel', 'mana', 'potential', 'true', 'right', 'jan', 'instagram', 'kya', 'tf', 'smart', 'download', 'tenure', 'windows11', 'devilish', 'centre', 'quote', 'favourite', 'functioning', 'netflix', 'winners', 'jklf', 'spectacular', 'ark', '@coldplay', 'event', 'hostile', 'song', 'btw']
    #['meet', 'people', 'child', 'hind', '@tvduishere', 'funny', 'better', '@surbhijtweets', '@lexfridman', 'wishes', 'jai', 'medical', 'women', 'oxygen', 'vegan', 'anganwadis', 'wala', 'woman', 'menstrual', 'congo', 'social', 'gandhi', 'day', 'development', 'lie', 'life', 'hospitals', 'thevampirediaries', 'masks', 'late', 'service', 'trio', 'shortage', 'dog', 'naagin3', '@slopez30', 'sign', 'night', 'rt', "b'day", 'mina', '@tvdobrxv', 'attention', 'jharkhand', 'petition', 'happy', 'maneka', '@upasanakonidela', 'u', 'guys', 'justice', 'url', 'leg', 'covid', 'years', 'ways', 'pet', '@pearlvpuri', 'denounce', 'cases', 'ministry', 'hygiene', '@anitahasnandani', '@amarsurendran99', 'pat', '@umarabb11317243', 'mandatory', 'easier', 'adult']
    
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

print(predict_user_gender(['loki', 'fascism', 'quantum', 'interdisciplinary', 'moon', 'man', 'tmc', 'treasure', 'goons', 'rajiv', 'session', 'asteroids', 'messaging', 'normal', 'classical', 'real', 'events', 'nahi', 'birth', 'invitation', '@cucompscistem', 'blue', 'redirec', '@rickandmorty', 'fascinating', '@xkcdcomic', 'impact', 'janata', 'journalists', 'nice', 'cardiff', 'sessions', 'fr', '@amitshah', 's', 'qna', 'wrong', 'disney', 'able', 'person', 'computing', 'people', 'good', 'dharna', '@spacex', '3rd', 'dharne', 'willis', 'govt', 'damn', 'dinosaurs', 'lectures', 'twitter', 'phone', 'kashmiri', 'year', 'way', 'influencers', 'hunt', 'aukaat', 'code', 'sure', 'gandhi', 'bengal', 'life', '@cucybersoc', 'vps', 'attacks', 'b*tch', 'bruce', 'rt', 'superstar', 'mars', 'jupiter', 'back!', 'government', 'karlenge', '@awokeaware', 'powerless', 'months', 'autocorrect', 'saturn', 'bro', 'ab', 'universe', 'spinthariscope', 'awkward', 'entertaining', 'ki', 'spring', 'romance', 'yesterday', '@qmunitytech', 'best', '@nobelprize', '@elonmusk', 'tonight', '@ixfoduap', 'exodus', 'undergraduate', 'responsible', 'bjp', 'atari', 'group', 'nasa', 'kickoff', 'live', 'control', '@amiramorphism', 'long', 'time', 'type', 'blackberry', 'thing', '@theanushcasm', 'machine', 'movie', '@xkcd', 'saver', '@qiskit', 'fun', '@mahuamoitra', 'sunset', 'field', 'chrome', '@amoghgeorge', 'url', 'bolne', 'dal', 'telephoto', 'bengalviolence2021', 'technological', 'half', 'hai', 'pandits', 'dragon', 'action', 'men', 'aap', 'congrats', 'singh', 'ke', 'computers', 'falcon', '@lexfridman', 'nevermind', 'pm', 'tweet', 'amazing', 'structured', 'hahaha', 'dec', '@bjp4india', 'vp', 'impaired', 'thanks', 'qgss21', 'great', 'audacity', 'qr', 'fault', '2nd', '@devchhatbar', '@pmoindia', 'panel', 'mana', 'potential', 'true', 'right', 'jan', 'instagram', 'kya', 'tf', 'smart', 'download', 'tenure', 'windows11', 'devilish', 'centre', 'quote', 'favourite', 'functioning', 'netflix', 'winners', 'jklf', 'spectacular', 'ark', '@coldplay', 'event', 'hostile', 'song', 'btw']))