import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import to_hex, Normalize
import gmaps.geojson_geometries
from sklearn.preprocessing import minmax_scale
from ipywidgets.embed import embed_minimal_html
import gmaps
from geopy.geocoders import Nominatim
import Tweets


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

    scatter_layer = gmaps.symbol_layer(locations, fill_color=colors, stroke_color=colors, info_box_content=cluster_info_text)
    map_fig.add_layer(scatter_layer)


def geocode_locations(tweet_data):
    geolocator = Nominatim(user_agent="plotting_points", timeout=300)

    lat, long, country = [], [], []
    for i in tweet_data['User Location']:
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
    kmeans = KMeans(n_clusters=35, init='k-means++', random_state=10, max_iter=200)
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
    sentiment_color = [calculate_color(color) for color in clusters['Sentiment']]
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

    cluster_info_text = [info_box_template.format(Sentiment=i) for i in clusters['Sentiment']]

    cluster_layer = gmaps.symbol_layer(clusters[['Lat', 'Long']],
                                       fill_color=sentiment_color,
                                       stroke_color=sentiment_color,
                                       scale=scales,
                                       fill_opacity=0.5,
                                       stroke_opacity=0,
                                       display_info_box=True,
                                       info_box_content=cluster_info_text)
    map_fig.add_layer(cluster_layer)


figure_layout = {
    'width': '100%',
    'height': '75vh',
    'border': '2px solid white',
    'padding': '2px'
}

m = gmaps.figure(zoom_level=1, layout=figure_layout, center=[0, 0])
tweet_data = Tweets.main()
lat_long = geocode_locations(tweet_data)
tweet_data['Latitude'] = lat_long[0]
tweet_data['Longitude'] = lat_long[1]
tweet_data['Country'] = lat_long[2]
tweet_data.dropna(axis=0, inplace=True)
tweet_data = tweet_data.reset_index()
tweet_data.to_csv('tweet_data.csv')
scatter_plot(m, tweet_data)

embed_minimal_html('export.html', views=[m])
