# import csv
# from django.shortcuts import render
# from django.shortcuts import redirect, render
import json
import random
from django.shortcuts import render
import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import to_hex, Normalize
import gmaps.geojson_geometries
from sklearn.preprocessing import minmax_scale
# from ipywidgets.embed import embed_minimal_html
import gmaps
from geopy.geocoders import Nominatim


# import json
# Create your views here.


def heat(request):

    # Credentials can be changed here depending on the user
    api_key = "YJKQrvmFj4IOSv27nonp8aBGx"
    api_secret = "3wDOUZcAAeTGvH4cNBjgAGYJ2gqYqOEc80rUI3oanGl9igjqbG"
    access_token = "1364148883329220609-WEj8Ijit8g79xor6qCRqJ7pMHAqdIe"
    access_token_secret = "LNZWMyDykX0x2oHe5i8z3dLF1g4lMWQsSszaZDNNJsECE"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAGjjVgEAAAAArOMqLJZ0092d0YcI2z2FBEF6ZUg%3DYh0UCSP4kAYkYFmRkrJnuHMMwMhvz7VIyJtWdlbiT1PqOyXrHw"

    def authenticate(api_key, api_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        # print(api.verify_credentials().screen_name)
        return api

        # function to recieve data
    def auth():
        client = tweepy.Client(bearer_token)
        return client

    def tweet_data(api, keyword, max_r):
        tweets = tweepy.Cursor(api.search_tweets, q=keyword).items(max_r)
        # Changes can be made here to get more tweet fields
        tweets_list = [[tweet.user.id, tweet.user.name, tweet.id,
                        tweet.user.location, tweet.created_at, tweet.text] for tweet in tweets]
        # Converting the data to Pandas DataFrame
        tweets_df = pd.DataFrame(tweets_list, columns=[
            'UserID', 'Name', 'TweetID', 'User Location', 'Date and Time', 'Text'])
        # print(tweets_df)
        return tweets_df

    #  function to perform Sentiment Analysis

    def sentiments(tweets):
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

    def m():
        api = authenticate(api_key, api_secret,
                           access_token, access_token_secret)

        # To take name of the brand and max number of tweets from the user
        if 'kw' in request.GET and request.GET['kw']:
            keyword = request.GET['kw']
        else:
            keyword = "uber"
        # print(keyword)
        max_r = 80
        tweets = tweet_data(api, keyword, int(max_r))
        # print(tweets['User Location'])
        # analysed_tweets = sentiments(tweets)
        # print(analysed_tweets)
        return tweets
        # 'analysed_tweets' is the final data yet

    def m1():
        # Credentials can be changed here depending on the user
        api = authenticate(api_key, api_secret,
                           access_token, access_token_secret)

        # To take name of the brand and max number of tweets from the user
        if 'kw' in request.GET and request.GET['kw']:
            keyword = request.GET['kw']
        else:
            keyword = "uber"
        # print(keyword)
        max_r = 80

        tweets = tweet_data(api, keyword, int(max_r))
        # print(tweets['User Location'])
        analysed_tweets = sentiments(tweets)
        # print(analysed_tweets)
        return analysed_tweets
        # 'analysed_tweets' is the final data yet

    gmaps.configure('AIzaSyDYXPLgcTlgMqebFc_da8nO72--5XS5CZ8')

    def calculate_color(sentiment):
        norm = Normalize(vmin=-1, vmax=1)
        cmap = cm.RdYlGn
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        mpl_color = m.to_rgba(sentiment)
        gmaps_color = to_hex(mpl_color)

        return gmaps_color

    def scatter_plot(map_fig, tweets):
        info_box_template = """
            <d1>
            <dt>{Country}</dt>
            <dt>Sentiment</dt><dd>{Sentiment}</dd>
            </d1>

            """

        cluster_info_text = [info_box_template.format(Sentiment=str(tweets['compound'][i]),
                                                      Country=str(tweets['Country'][i])) for i in range(len(tweets['compound']))]
        colors = []
        for i in range(len(tweets['User Location'])):
            colors.append(calculate_color(tweets['compound'][i]))

        locations = tweets[['Latitude', 'Longitude']]

        scatter_layer = gmaps.symbol_layer(
            locations, fill_color=colors, stroke_color=colors, info_box_content=cluster_info_text)
        map_fig.add_layer(scatter_layer)

    def geocode_locations(tweet_data):
        geolocator = Nominatim(user_agent="plotting_points", timeout=300)
        lat, long, country = [], [], []
        for i in tweet_data1['User Location']:
            if i == '':
                lat.append(None)
                long.append(None)
                country.append(None)
            else:
                location = geolocator.geocode(i)
                if location:
                    lat.append(location.latitude)
                    long.append(location.longitude)
                    country.append(location.address.split(",")[-1])
                else:
                    lat.append(None)
                    long.append(None)
                    country.append(None)

        return lat, long, country

    def heatmap_layer(map_figure, tweets):
        heatmap = gmaps.heatmap_layer(tweets[['Latitude', 'Longitude']])
        heatmap.point_radius = 20
        heatmap.max_intensity = 25
        map_figure.add_layer(heatmap)

    def create_clusters(locations):
        kmeans = KMeans(n_clusters=35, init='k-means++',
                        random_state=10, max_iter=200)
        y_kmeans = kmeans.fit_predict(locations[['Longitude', 'Latitude']])
        locations['cluster'] = y_kmeans
        avg_sentiment, Lat, Long, cluster_sizes = [], [], [], []

        for i in range(kmeans.n_clusters):
            cluster_indicies = np.where(kmeans.labels_ == i)[0]
            sum_sentiment = 0
            for x in cluster_indicies:
                sum_sentiment += locations.compound[x]

            avg_sentiment.append(sum_sentiment / len(cluster_indicies))
            cluster_sizes.append(len(cluster_indicies) * 5)
            Long.append(kmeans.cluster_centers_[i][0])
            Lat.append(kmeans.cluster_centers_[i][1])

            clusters_data = {'Lat': Lat,
                             'Long': Long,
                             'Size': cluster_sizes,
                             'Sentiment': avg_sentiment}
            clusters = pd.DataFrame(clusters_data)

        return clusters

    def cluster_map(map_fig, tweets):
        clusters = create_clusters(tweets)
        sentiment_color = [calculate_color(color)
                           for color in clusters['Sentiment']]
        scales = clusters['Size'].tolist()
        minmax_scale(scales)
        for i in range(len(scales)):
            scales[i] = int(scales[i] / 3)
            if scales[i] <= 0:
                scales[i] = 1

        info_box_template = """
        <d1>
        <dt>Sentiment</dt><dd>{Sentiment}</dd>
        <button type="button">Stats</button>
        <button type="button">Pie</button>
        </d1>

        """

        cluster_info_text = [info_box_template.format(
            Sentiment=i) for i in clusters['Sentiment']]

        cluster_layer = gmaps.symbol_layer(clusters[['Lat', 'Long']],
                                           fill_color=sentiment_color,
                                           stroke_color=sentiment_color,
                                           scale=scales,
                                           fill_opacity=0.5,
                                           stroke_opacity=0,
                                           display_info_box=True,
                                           info_box_content=cluster_info_text)
        map_fig.add_layer(cluster_layer)

    '''figure_layout = {
        'width': '100%',
        'height': '75vh',
        'border': '2px solid white',
        'padding': '2px'
    }'''

    # m = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])
    tweet_data1 = m()
    lat_long = geocode_locations(tweet_data)
    tweet_sentiment = m1()
    tweet_data1['Latitude'] = lat_long[0]
    tweet_data1['Longitude'] = lat_long[1]
    tweet_data1['Country'] = lat_long[2]
    location = []
    customer(request)
    # print(lat_long)
    for i in range(40):
        if(lat_long[0][i] is not None and lat_long[1][i] is not None):
            location.append({"lat": lat_long[0][i],
                             "long": lat_long[1][i],
                             "sentiment": tweet_sentiment['comp_score'][i]})
        else:
            location.append({"lat": random.randint(0, 70),
                             "long": random.randint(-3, 120),
                             "sentiment": tweet_sentiment['comp_score'][i]})
    return render(request, 'googleHeatMap.html', {'loc': json.dumps(location)})
    # tweet_data.dropna(axis=0, inplace=True)
    # tweet_data = tweet_data.reset_index()
    # tweet_data.to_csv('tweet_data.csv')
    # scatter_plot(m, tweet_data)
    # embed_minimal_html('export.html', views=[m])


def customer(request):
    # Credentials can be changed here depending on the user
    api_key = "YJKQrvmFj4IOSv27nonp8aBGx"
    api_secret = "3wDOUZcAAeTGvH4cNBjgAGYJ2gqYqOEc80rUI3oanGl9igjqbG"
    access_token = "1364148883329220609-WEj8Ijit8g79xor6qCRqJ7pMHAqdIe"
    access_token_secret = "LNZWMyDykX0x2oHe5i8z3dLF1g4lMWQsSszaZDNNJsECE"
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAGjjVgEAAAAArOMqLJZ0092d0YcI2z2FBEF6ZUg%3DYh0UCSP4kAYkYFmRkrJnuHMMwMhvz7VIyJtWdlbiT1PqOyXrHw"

    def auth():
        client = tweepy.Client(bearer_token)
        return client

    def authenticate(api_key, api_secret, access_token, access_token_secret):
        auth = tweepy.OAuthHandler(api_key, api_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        # print(api.verify_credentials().screen_name)
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
        # print(tweets_df)
        return tweets_df

    #  function to perform Sentiment Analysis

    def sentiments(tweets):
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

    def m1():

        # To take name of the brand and max number of tweets from the user
        if 'kw' in request.GET and request.GET['kw']:
            keyword = request.GET['kw']
        else:
            keyword = "cardiff"
        max_r = 40

        api = authenticate(api_key, api_secret,
                           access_token, access_token_secret)
        tweets = tweet_data(api, keyword, int(max_r))
        # print(tweets['User Location'])
        analysed_tweets = sentiments(tweets)
        # print(analysed_tweets)
        return analysed_tweets
        # 'analysed_tweets' is the final data yet
    analysed_result = m1()
    p_customer = []
    for i in analysed_result.index:
        # print(analysed_result['comp_score'][i])
        if(analysed_result['comp_score'][i] == "positive"):
            p_customer.append(
                {"Name": analysed_result['Name'][i], "loc": analysed_result['User Location'][i]})
    # print(p_customer)
    return render(request, 'customer.html', {'customerList': json.dumps(p_customer)})
