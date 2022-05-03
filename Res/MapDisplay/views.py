from django.shortcuts import redirect, render
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.cm import RdYlGn
from matplotlib.colors import to_hex
import gmaps.geojson_geometries
from sklearn.preprocessing import minmax_scale
from ipywidgets.embed import embed_minimal_html
import gmaps
import gmaps.datasets
# import json
# Create your views here.


def md(request):
    return render(request, 'map.html', {'msg': None})


def plotting(request):
    gmaps.configure('AIzaSyDYXPLgcTlgMqebFc_da8nO72--5XS5CZ8')

    figure_layout = {
        'width': '400px',
        'height': '400px',
        'border': '1px solid black',
        'padding': '1px'
    }
    gmaps.figure(layout=figure_layout)

    def create_clusters(locations):
        # number of clusters adjusted here
        kmeans = KMeans(n_clusters=20, init='k-means++',
                        random_state=10, max_iter=200)
        y_kmeans = kmeans.fit_predict(locations[['Longitude', 'Latitude']])
        locations['cluster'] = y_kmeans
        avg_sentiment, Lat, Long, cluster_sizes = [], [], [], []

        for i in range(kmeans.n_clusters):
            cluster_indicies = np.where(kmeans.labels_ == i)[0]
            sum_sentiment = 0
            for x in cluster_indicies:
                sum_sentiment += locations.Sentiment[x]

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

    def calculate_color(sentiment):
        inverse_sentiment = 1.0 - sentiment
        mpl_color = RdYlGn(inverse_sentiment)
        gmaps_color = to_hex(mpl_color, keep_alpha=False)

        return gmaps_color

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

    def heatmap_layer(map_figure, tweets):
        heatmap = gmaps.heatmap_layer(tweets[['Latitude', 'Longitude']])
        heatmap.point_radius = 20
        heatmap.max_intensity = 25
        map_figure.add_layer(heatmap)

    def geojson_layer(map_figure, tweets):
        countries_geojson = gmaps.geojson_geometries.load_geometry(
            'countries-high-resolution')
        # with open('world-administrative-boundaries.geojson') as f:
        #   geometry = json.load(f)

        countries = {}
        countries_count = {}
        for i in range(len(tweets['Location'])):
            if tweets['Location'][i] in countries:
                countries[tweets['Location'][i]] += tweets['Sentiment'][i]
                countries_count[tweets['Location'][i]] += 1
            else:
                countries[tweets['Location'][i]] = tweets['Sentiment'][i]
                countries_count[tweets['Location'][i]] = 1
        for x in countries_count:
            countries[x] = countries[x]/countries_count[x]

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

    def scatter_plot(map_fig, tweets):
        info_box_template = """
                <d1>
                <dt>{Country}</dt>
                <dt>Sentiment</dt><dd>{Sentiment}</dd>
                </d1>

                """

        cluster_info_text = [info_box_template.format(Sentiment=tweets['Sentiment'][i],
                                                      Country=tweets['Location'][i]) for i in range(len(tweets['Sentiment']))]
        colors = []
        for i in range(len(tweets['Location'])):
            colors.append(calculate_color(tweets['Sentiment'][i]))
        tweets = tweets[['Latitude', 'Longitude']]

        scatter_layer = gmaps.symbol_layer(
            tweets, fill_color=colors, stroke_color=colors, info_box_content=cluster_info_text)
        map_fig.add_layer(scatter_layer)

    m = gmaps.Map()
    # The path here need to be change
    tweet_data = pd.read_csv(
        'D:\python code\Register\Res\MapDisplay\location_test.csv')

    # ###UNCOMMENT FOR LAYER####

    # cluster_map(m, tweet_data)
    # heatmap_layer(m, tweet_data)
    geojson_layer(m, tweet_data)
    # pie_chart_sentiment(clusters.iloc[2])
    # scatter_plot(m, tweet_data)
    embed_minimal_html('templates/map.html', views=[m])
