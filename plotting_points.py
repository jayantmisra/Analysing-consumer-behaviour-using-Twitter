import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from matplotlib.colors import to_hex, Normalize
import gmaps.geojson_geometries
from sklearn.preprocessing import minmax_scale
from ipywidgets.embed import embed_minimal_html
import gmaps
from geopy.geocoders import Nominatim
import Tweets
import math

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
            <dt>-----------------------</dt>
            <dt>{Tweet}</dt>
            <dt>-----------------------</dt>
            <dt>Sentiment</dt><dd>{Sentiment}</dd>
            </d1>

            """

    cluster_info_text = [info_box_template.format(Sentiment=str(tweets['compound'][i]),
                                                  Tweet=(tweets['Text'][i]),
                                                  Country=str(tweets['Country'][i])) for i in range(len(tweets['compound']))]
    colors = []
    for i in range(len(tweets['User Location'])):
        colors.append(calculate_color(tweets['compound'][i]))

    locations = tweets[['Latitude', 'Longitude']]

    scatter_layer = gmaps.symbol_layer(locations, fill_color=colors, stroke_color=colors, info_box_content=cluster_info_text)
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
    geolocator = Nominatim(user_agent="plotting_points", timeout=20)
    lat, long, country = [], [], []
    for i in tweet_data['User Location']:
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
    kmeans = KMeans(n_clusters=number_clusters, init='k-means++', random_state=10, max_iter=200)
    y_kmeans = kmeans.fit_predict(locations[['Longitude', 'Latitude']])
    locations['cluster'] = y_kmeans
    avg_sentiment, Lat, Long, cluster_sizes = [], [], [], []
    clusters_indicies = [[] for _ in range(number_clusters)]

    for i in range(kmeans.n_clusters):
        sum_sentiment = 0
        inside_clusters = []
        for ind, row in locations.loc[(locations['cluster'] == i)].iterrows():
            sum_sentiment += row.compound
            inside_clusters.append(row['index'])
        clusters_indicies[i] = (inside_clusters)

        avg_sentiment.append(sum_sentiment / len(inside_clusters))
        cluster_sizes.append(len(inside_clusters) * 5)
        Long.append(round(kmeans.cluster_centers_[i][0], 3))
        Lat.append(round(kmeans.cluster_centers_[i][1], 3))

    clusters_data = {'Lat': Lat,
                     'Long': Long,
                     'Size': cluster_sizes,
                     'Sentiment': avg_sentiment}
    clusters = pd.DataFrame(clusters_data)

    return clusters, clusters_indicies


def cluster_map(map_fig, tweets, number_clusters):
    clusters = create_clusters(tweets, number_clusters)
    cluster_indicies = clusters[1]
    clusters = clusters[0]
    sentiment_color = [calculate_color(color) for color in clusters['Sentiment']]
    scales = clusters['Size'].tolist()
    minmax_scale(scales)
    for i in range(len(scales)):
        scales[i] = int(scales[i] / 3)
        if scales[i] <= 0:
            scales[i] = 5

    info_box_template = """
    <d1>
    <dt>Location</dt><dd>[{Latitude} , {Longitude}]</dd>
    <dt>Sentiment</dt><dd>{Sentiment}</dd>
    <button type="button" id= {i} >Details</button>
    </d1>

    """

    cluster_info_text = [info_box_template.format(Sentiment=clusters['Sentiment'][i],
                                                  Latitude=clusters['Lat'][i],
                                                  Longitude=clusters['Long'][i],
                                                  i=i) for i in range(len(clusters['Sentiment']))]

    cluster_layer = gmaps.symbol_layer(clusters[['Lat', 'Long']],
                                       fill_color=sentiment_color,
                                       stroke_color=sentiment_color,
                                       scale=scales,
                                       fill_opacity=0.5,
                                       stroke_opacity=0,
                                       display_info_box=True,
                                       info_box_content=cluster_info_text)
    map_fig.add_layer(cluster_layer)
    return cluster_indicies


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

countries_geojson = gmaps.geojson_geometries.load_geometry('countries-high-resolution')

#tweet_data = pd.read_csv('tweet_data.csv')
tweet_data = Tweets.main()
tweet_data = tweet_prep(tweet_data)
#tweet_data.to_csv('tweet_data.csv')

m = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])
n = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])
o = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])

geojson_layer(m, countries_geojson, tweet_data)
cluster_data = cluster_map(n, tweet_data, int(math.sqrt(len(tweet_data.index))))
scatter_plot(o, tweet_data)

embed_minimal_html('geojson.html', views=[m])
embed_minimal_html('cluster_map.html', views=[n])
embed_minimal_html('scatter_plot.html', views=[o])

