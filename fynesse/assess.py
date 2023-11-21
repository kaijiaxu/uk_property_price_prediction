from .config import *

from . import access

import osmnx as ox
import matplotlib.pyplot as plt
import mlai
import mlai.plot as plot
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, fixed
import numpy as np
import geopandas as gpd
from geopandas.tools import sjoin
from IPython.display import display
import shapely.geometry

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


### OSM ###

def data_validation_df(latitude, longitude, scale):
    """
    Matches property data from OSM to prices_coordinates_data
    """
    # Query OSM for residential data. Includes 'apartments', 'terrace', 'house', 'detached' or 'semidetached_house'.
    tags = {"residential": True}
    (north, south, east, west) = access.get_bounding_box(latitude, longitude, scale, scale)
    osm_df = access.get_pois(north, south, east, west, tags)
    print(f"Number of properties from OSM: {len(osm_df)}")
    # Query prices_coordinates_data
    prices_coordinates_df = access.togpd(access.get_prices_coordinates_df_by_coordinates(north, south, east, west))
    print(f"Number of properties from prices_coordinates_data: {len(prices_coordinates_df)}")
    # Join the two dataframes based on geometry
    joined_df = sjoin(osm_df, prices_coordinates_df, how='inner')
    print(f"Number of matches: {len(joined_df)}")
    return display(joined_df)


def data_validation_interaction(scale):
    """
    Allows interaction in the Jupyter Notebook
    """
    latitude_slider = widgets.FloatSlider(min=49.89517100, max=55.79741500, step=scale, value=50)
    longitude_slider = widgets.FloatSlider(min=-6.35264700, max=1.76277300, step=scale, value=0)
    _ = interact(data_validation_df,
            latitude=latitude_slider,
            longitude=longitude_slider,
            scale=fixed(scale))


def osm_plot(tags):
    """
    Plots the data from OSM for the whole of the UK to see the distribution
    """
    tag_names = '-'.join(tags.keys())
    # Plot UK outline
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    # Get OSM data
    data = access.get_pois(55.79741500, 49.89517100, 1.76277300, -6.35264700, tags)
    node_data = data.loc['node']

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax=ax, color='white', edgecolor='black')
    node_data.plot(ax=ax, color='b', alpha=0.05)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'OSM data for {tag_names}') 
    plt.tight_layout() 
    mlai.write_figure(f'osm-{tag_names}.jpg', directory='./ml')


### Prices Coordinates Data ###

def price_coord_df(year):
    # Plot UK outline
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    prices_coordinates_df = access.togpd(access.get_prices_coordinates_df_by_year(year))

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax=ax, color='white', edgecolor='black')
    prices_coordinates_df.plot(ax=ax, color='b', alpha=0.05)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'Prices Coordinates data for {year}') 
    plt.tight_layout() 
    mlai.write_figure(f'Prices-coord-{year}.jpg', directory='./ml')

def price_coord_by_year(year):
    """
    Plots the latitude by longitude graph to see the distribution of prices_coordinates_data
    """
    _ = interact(price_coord_df,
            year=fixed(year))
    

def plot_price_distribution(year):
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    results = access.run_query_return_results(f"SELECT latitude, longitude, price FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'price'])
    prices_coordinates_df = access.togpd(df)

    BOXES = 50
    a, b, c, d = prices_coordinates_df.total_bounds

    # create a grid for UK
    gdf_grid = gpd.GeoDataFrame(
        geometry=[
            shapely.geometry.box(minx, miny, maxx, maxy)
            for minx, maxx in zip(np.linspace(a, c, BOXES), np.linspace(a, c, BOXES)[1:])
            for miny, maxy in zip(np.linspace(b, d, BOXES), np.linspace(b, d, BOXES)[1:])
        ],
        crs="epsg:4326",
    )

    # remove grid boxes created outside actual geometry
    gdf_grid = gdf_grid.sjoin(prices_coordinates_df).drop(columns="index_right")

    # get earthquakes that have occured within one of the grid geometries
    prices_coordinates_df_temp = prices_coordinates_df.loc[:, ["geometry", "price"]]
    # get median magnitude of eargquakes in grid
    gdf_grid = gdf_grid.join(
        prices_coordinates_df_temp.dissolve(by="index_right", aggfunc="median").drop(columns="geometry")
    )
    # how many earthquakes in the grid
    gdf_grid = gdf_grid.join(
        prices_coordinates_df_temp.dissolve(by="index_right", aggfunc=lambda d: len(d))
        .drop(columns="geometry")
        .rename(columns={"mag": "number"})
    )

    # drop grids geometries that have no measures and create folium map
    m = gdf_grid.dropna().explore(column="mag")
    # for good measure - boundary on map too
    prices_coordinates_df["geometry"].apply(lambda g: shapely.geometry.MultiLineString([p.exterior for p in g.geoms])).explore(m=m)

def plot_property_type_distribution(year):
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    results = access.run_query_return_results(f"SELECT latitude, longitude, property_type FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'property_type'])

    prices_coordinates_df = access.togpd(df)
    detached = prices_coordinates_df[prices_coordinates_df['property_type'] == 'D']
    semidetached = prices_coordinates_df[prices_coordinates_df['property_type'] == 'S']
    terraced = prices_coordinates_df[prices_coordinates_df['property_type'] == 'T']
    flats = prices_coordinates_df[prices_coordinates_df['property_type'] == 'F']
    other = prices_coordinates_df[prices_coordinates_df['property_type'] == 'O']

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax=ax, color='white', edgecolor='black')
    detached.plot(ax=ax, color='r', alpha=0.05)
    semidetached.plot(ax=ax, color='g', alpha=0.05)
    terraced.plot(ax=ax, color='b', alpha=0.05)
    flats.plot(ax=ax, color='c', alpha=0.05)
    other.plot(ax=ax, color='m', alpha=0.05)
    ax.legend(["detached", "semidetached", "terraced", "flats", "other"])
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'Property Type distribution for {year}') 
    plt.tight_layout() 
    mlai.write_figure(f'property-type-{year}.jpg', directory='./ml')


def plot_new_build_distribution(year):
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    results = access.run_query_return_results(f"SELECT latitude, longitude, new_build_flag FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'new_build_flag'])

    prices_coordinates_df = access.togpd(df)
    old_build = prices_coordinates_df[prices_coordinates_df['new_build_flag'] == 'N']
    new_build = prices_coordinates_df[prices_coordinates_df['new_build_flag'] == 'Y']

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax=ax, color='white', edgecolor='black')
    old_build.plot(ax=ax, color='g', alpha=0.05)
    new_build.plot(ax=ax, color='r', alpha=0.05)
    ax.legend(["old builds", "new builds"])
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    fig.suptitle(f'New Build distribution for {year}') 
    plt.tight_layout() 
    mlai.write_figure(f'new-build-{year}.jpg', directory='./ml')


def plot_tenure_type_distribution(year):
    world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world_gdf.crs = "EPSG:4326"
    uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

    results = access.run_query_return_results(f"SELECT latitude, longitude, tenure_type FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'tenure_type'])

    prices_coordinates_df = access.togpd(df)
    freehold = prices_coordinates_df[prices_coordinates_df['tenure_type'] == 'F']
    lease = prices_coordinates_df[prices_coordinates_df['tenure_type'] == 'L']
    other = prices_coordinates_df[(prices_coordinates_df['tenure_type'] != 'L') & (prices_coordinates_df['tenure_type'] != 'F')]

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax=ax, color='white', edgecolor='black')
    freehold.plot(ax=ax, color='r', alpha=0.05)
    lease.plot(ax=ax, color='g', alpha=0.05)
    other.plot(ax=ax, color='b', alpha=0.05)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.legend(["freehold", "lease", "others"])
    fig.suptitle(f'Tenure Type distribution for {year}') 
    plt.tight_layout() 
    mlai.write_figure(f'tenure-type-{year}.jpg', directory='./ml')



def plot_average_price_by_year():
    results = access.run_query_return_results("SELECT EXTRACT(year FROM date_of_transfer) AS year, AVG(price) AS average_price FROM prices_coordinates_data GROUP BY EXTRACT(year FROM date_of_transfer);")
    df = pd.DataFrame(results, columns=['year', 'average_price'])
    print(df)
    plt.plot(df['year'], df['average_price'])
    plt.title('Average property prices across the years')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.show()

# def plot_correlation_price_property_type(year):
#     world_gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#     world_gdf.crs = "EPSG:4326"
#     uk_gdf = world_gdf[(world_gdf['name'] == 'United Kingdom')]

#     results = access.run_query_return_results(f"SELECT AVG(price), property_type FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31' GROUP BY property_type;")
#     df = pd.DataFrame(results, columns=['average_price', 'property_type'])



#     fig, ax = plt.subplots(figsize=plot.big_figsize)
#     uk_gdf.plot(ax=ax, color='white', edgecolor='black')
#     freehold.plot(ax=ax, color='r', alpha=0.05)
#     lease.plot(ax=ax, color='g', alpha=0.05)
#     other.plot(ax=ax, color='b', alpha=0.05)
#     ax.set_xlabel('longitude')
#     ax.set_ylabel('latitude')
#     fig.suptitle(f'Tenure Type distribution for {year}') 
#     plt.tight_layout() 
#     mlai.write_figure(f'tenure-type-{year}.jpg', directory='./ml')

# def data():
#     """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
#     df = access.data()
#     raise NotImplementedError

# def query(data):
#     """Request user input for some aspect of the data."""
#     raise NotImplementedError

# # def view(north, south, east, west, tags):
# #     """Provide a view of the data that allows the user to verify some aspect of its quality."""
# #     graph = ox.graph_from_bbox(north, south, east, west)
# #     # Retrieve nodes and edges
# #     nodes, edges = ox.graph_to_gdfs(graph)
# #     area = ox.geocode_to_gdf("United Kingdom")
# #     fig, ax = plt.subplots(figsize=plot.big_figsize)
# #     # Plot the footprint
# #     area.plot(ax=ax, facecolor="white")
# #     # Plot street edges
# #     edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
# #     ax.set_xlim([west, east])
# #     ax.set_ylim([south, north])
# #     ax.set_xlabel("longitude")
# #     ax.set_ylabel("latitude")

# #     # Plot all POIs
# #     pois = access.get_pois(north, south, east, west, tags)
# #     pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
# #     plt.tight_layout()
# #     mlai.write_figure(directory="./maps", filename="pois.jpg")

# def labelled(data):
#     """Provide a labelled set of data ready for supervised learning."""
#     raise NotImplementedError
