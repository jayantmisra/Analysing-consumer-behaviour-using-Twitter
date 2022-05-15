from django.shortcuts import render
import tweepy
import pandas as pd
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import to_hex, Normalize
import gmaps.geojson_geometries
from sklearn.preprocessing import minmax_scale
from ipywidgets.embed import embed_minimal_html
import gmaps
from geopy.geocoders import Nominatim
from googletrans import Translator
# function for authentication
# Create your views here.


def plotting_points(request):
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

        # Translates all of the tweets into english for the sentiment analysis
        translator = Translator()
        for column in range(0, tweets['Text'].size):
            tweets.loc[column, 'Text'] = translator.translate(
                tweets['Text'].iloc[column]).text

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

    def ma():

        # To take name of the brand and max number of tweets from the user
        keyword = input("Enter the keyword :")
        max_r = input("Enter the max number of tweets required :")

        api = authenticate(api_key, api_secret,
                           access_token, access_token_secret)
        tweets = tweet_data(api, keyword, int(max_r))
        # print(tweets['User Location'])
        analysed_tweets = sentiments(tweets)
        # print(analysed_tweets)
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
        translator = Translator()
        info_box_template = """
                <d1>
                <dt>{Country}</dt>
                <dt>-----------------------</dt>
                <dt>{Tweet}</dt>
                <dt>-----------------------</dt>
                <dt>Sentiment</dt><dd>{Sentiment}</dd>
                </d1>

                """

        cluster_info_text = [info_box_template.format(Sentiment=str(tweets['compound'][i]),
                                                      Tweet=translator.translate(
                                                          (tweets['Text'][i])).text,
                                                      Country=str(tweets['Country'][i])) for i in range(len(tweets['compound']))]
        colors = []
        for i in range(len(tweets['User Location'])):
            colors.append(calculate_color(tweets['compound'][i]))

        locations = tweets[['Latitude', 'Longitude']]

        scatter_layer = gmaps.symbol_layer(
            locations, fill_color=colors, stroke_color=colors, info_box_content=cluster_info_text)
        map_fig.add_layer(scatter_layer)

    def countries_syntax(countries):
        changes = {'United States of America': 'United States', 'United Republic of Tanzania': 'Tanzania',
                   'Guinea Bissau': 'Guinea-Bissau', 'The Gambia': 'Gambia', 'Ivory Coast': 'CÃ´te d\'Ivoire',
                   'Republic of Congo': 'Liberia', 'Macedonia': 'North Macedonia', 'Republic of Serbia': 'Serbia',
                   'Czech Republic': 'Czechia'}

        for i in changes:
            if changes[i] in countries:
                countries[i] = countries.pop(changes[i])

        return countries

    def country_sentiment(tweets):

        countries = {}
        countries_count = {}
        for i in range(len(tweets['Country'])):
            if tweets['Country'][i] in countries:
                countries[tweets['Country'][i]] += tweets['compound'][i]
                countries_count[tweets['Country'][i]] += 1
            else:
                countries[tweets['Country'][i]] = tweets['compound'][i]
                countries_count[tweets['Country'][i]] = 1

        for x in countries_count:
            if countries_count[x] != 0:
                countries[x] = countries[x] / countries_count[x]

        countries = countries_syntax(countries)
        return countries

    def geojson_layer(map_figure, countries_geojson, tweet_data):
        countries = country_sentiment(tweet_data)
        colors = []
        for feature in countries_geojson['features']:
            country_name = feature['properties']['name']
            try:
                color = calculate_color(countries[country_name])
            except KeyError:
                color = '#e0e0e0'
            colors.append(color)

        gini_layer = gmaps.geojson_layer(countries_geojson, fill_color=colors, stroke_color='#000000',
                                         fill_opacity=0.8, stroke_weight=0.2)
        map_figure.add_layer(gini_layer)

    def geocode_locations(tweet_data):
        geolocator = Nominatim(user_agent="plotting_points", timeout=300)
        lat, long, country = [], [], []
        for i in tweet_data1['User Location']:
            if i == '':
                lat.append(None)
                long.append(None)
                country.append(None)
            else:
                location = geolocator.geocode(i, language='en')
                if location:
                    lat.append(location.latitude)
                    long.append(location.longitude)
                    location_country = location.address.split(", ")[-1]
                    country.append(location_country)
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

    def create_clusters(locations, number_clusters):
        kmeans = KMeans(n_clusters=number_clusters,
                        init='k-means++', random_state=10, max_iter=200)
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

    def cluster_map(map_fig, tweets, number_clusters):
        clusters = create_clusters(tweets, number_clusters)
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

    def tweet_prep(tweet_data):
        lat_long = geocode_locations(tweet_data)
        tweet_data['Latitude'] = lat_long[0]
        tweet_data['Longitude'] = lat_long[1]
        tweet_data['Country'] = lat_long[2]
        tweet_data.dropna(axis=0, inplace=True)
        tweet_data = tweet_data.reset_index()

        return tweet_data

    figure_layout = {
        'width': '100%',
        'height': '750px',
        'border': '2px solid white',
        'padding': '2px'
    }

    m = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])
    countries_geojson = gmaps.geojson_geometries.load_geometry(
        'countries-high-resolution')

    # tweet_data = pd.read_csv('tweet_data.csv')
    tweet_data1 = ma()
    # tweet_data = db.query("SELECT * FROM Tweets WHERE rowid >= 10110 AND rowid <= 10460")
    tweet_data1 = tweet_prep(tweet_data)
    tweet_data1.to_csv('tweet_data.csv')

    geojson_layer(m, countries_geojson, tweet_data)
    # cluster_map(m, tweet_data, int(math.sqrt(len(tweet_data.index))))
    # scatter_plot(m, tweet_data)

    embed_minimal_html('templates/export.html', views=[m])
    return render(request, 'export.html')
