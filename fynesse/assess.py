from .config import *

from . import access
from . import address

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
import matplotlib as mpl

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


### Helper functions ###
def get_uk_outline():
    """
    Return a dataframe that allows us to plot the UK map background
    """
    uk_gdf = ox.geocoder.geocode_to_gdf('United Kingdom')
    uk_gdf.crs = "EPSG:4326"
    return uk_gdf


def split_gdf_into_boxes(gdf):
    """
    Split the UK into smaller boxes, allowing us to aggregate data within these boxes
    """
    BOXES = 50
    a, b, c, d = gdf.total_bounds
    # create a grid for UK
    gdf_grid = gpd.GeoDataFrame(
        geometry=[
            shapely.geometry.box(minx, miny, maxx, maxy)
            for minx, maxx in zip(np.linspace(a, c, BOXES), np.linspace(a, c, BOXES)[1:])
            for miny, maxy in zip(np.linspace(b, d, BOXES), np.linspace(b, d, BOXES)[1:])
        ],
        crs="EPSG:4326",
    )
    return gdf_grid


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
    Allows interaction in the Jupyter Notebook for OSM property data
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
    uk_gdf = get_uk_outline()

    # Get OSM data
    data = access.get_pois(55.79741500, 49.89517100, 1.76277300, -6.35264700, tags)

    # Plot points
    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax = ax, alpha = 0.5)
    data.plot(ax=ax, color='r', alpha=0.8, markersize = 1, label='Locations')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'OSM data for {tag_names}') 
    plt.tight_layout() 
    plt.legend()
    mlai.write_figure(f'osm-{tag_names}.jpg', directory='./ml')     


def osm_plot_specific_area(north, south, east, west, tags):
    """
    Plots the data from OSM for the whole of the UK to see the distribution
    """
    tag_names = '-'.join(tags.keys())
    # Retrieve nodes and edges
    graph = ox.graph_from_bbox(north, south, east, west)
    nodes, edges = ox.graph_to_gdfs(graph)
    
    # Plot points
    fig, ax = plt.subplots(figsize=plot.big_figsize)
    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
    for osm_key in tags:
        # Get OSM data
        data = access.get_pois(north, south, east, west, {osm_key: tags[osm_key]})
        data.plot(ax=ax, alpha=0.7, markersize = 10, label=f'{osm_key}')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle('OSM data for specific area') 
    plt.tight_layout() 
    plt.legend()
    mlai.write_figure(f'osm-specific-{tag_names}.jpg', directory='./ml') 


### Prices Coordinates Data ###

def price_coord_df(year):
    """
    Plots all data in prices_coordinates_data within given year on the map of UK
    """
    # Plot UK outline
    uk_gdf = get_uk_outline()

    prices_coordinates_df = access.togpd(access.get_prices_coordinates_df_by_year(year))

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax = ax, alpha = 0.5)
    prices_coordinates_df.plot(ax=ax, color='r', alpha=0.8, markersize = 1)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'Prices Coordinates data for {year}') 
    plt.tight_layout() 
    mlai.write_figure(f'Prices-coord-{year}.jpg', directory='./ml')


def plot_property_type_distribution(year):
    """
    Plots different property types within given year on the map of UK
    """
    uk_gdf = get_uk_outline()

    results = access.run_query_return_results(f"SELECT latitude, longitude, property_type FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'property_type'])

    prices_coordinates_df = access.togpd(df)
    detached = prices_coordinates_df[prices_coordinates_df['property_type'] == 'D']
    semidetached = prices_coordinates_df[prices_coordinates_df['property_type'] == 'S']
    terraced = prices_coordinates_df[prices_coordinates_df['property_type'] == 'T']
    flats = prices_coordinates_df[prices_coordinates_df['property_type'] == 'F']
    other = prices_coordinates_df[prices_coordinates_df['property_type'] == 'O']

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax = ax, alpha = 0.5)
    detached.plot(ax=ax, color='r', alpha=0.8, markersize = 1, label='Detached property locations')
    semidetached.plot(ax=ax, color='g', alpha=0.8, markersize = 1, label='Semi-detached property locations')
    terraced.plot(ax=ax, color='black', alpha=0.8, markersize = 1, label='Terraced property locations')
    flats.plot(ax=ax, color='c', alpha=0.8, markersize = 1, label='Flats property locations')
    other.plot(ax=ax, color='m', alpha=0.8, markersize = 1, label='Other property locations')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'Property Type distribution for {year}') 
    plt.tight_layout() 
    plt.legend()
    mlai.write_figure(f'property-type-{year}.jpg', directory='./ml')


def plot_new_build_distribution(year):
    """
    Plots new build and old build within given year on the map of UK
    """
    uk_gdf = get_uk_outline()

    results = access.run_query_return_results(f"SELECT latitude, longitude, new_build_flag FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'new_build_flag'])

    prices_coordinates_df = access.togpd(df)
    old_build = prices_coordinates_df[prices_coordinates_df['new_build_flag'] == 'N']
    new_build = prices_coordinates_df[prices_coordinates_df['new_build_flag'] == 'Y']

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax = ax, alpha = 0.5)
    old_build.plot(ax=ax, color='g', alpha=0.8, markersize = 1, label='Old build locations')
    new_build.plot(ax=ax, color='r', alpha=0.8, markersize = 1, label='New build locations')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    fig.suptitle(f'New/Old Build distribution for {year}') 
    plt.tight_layout() 
    plt.legend()
    mlai.write_figure(f'new-old-build-{year}.jpg', directory='./ml')


def plot_tenure_type_distribution(year):
    """
    Plots freehold, lease and other tenure types within given year on the map of UK
    """
    uk_gdf = get_uk_outline()

    results = access.run_query_return_results(f"SELECT latitude, longitude, tenure_type FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31';")
    df = pd.DataFrame(results, columns=['latitude', 'longitude', 'tenure_type'])

    prices_coordinates_df = access.togpd(df)
    freehold = prices_coordinates_df[prices_coordinates_df['tenure_type'] == 'F']
    lease = prices_coordinates_df[prices_coordinates_df['tenure_type'] == 'L']
    other = prices_coordinates_df[(prices_coordinates_df['tenure_type'] != 'L') & (prices_coordinates_df['tenure_type'] != 'F')]

    fig, ax = plt.subplots(figsize=plot.big_figsize)
    uk_gdf.plot(ax = ax, alpha = 0.5)
    freehold.plot(ax=ax, color='r', alpha=0.8, markersize = 1, label='Freehold locations')
    lease.plot(ax=ax, color='g', alpha=0.8, markersize = 1, label='Leasehold locations')
    other.plot(ax=ax, color='black', alpha=0.8, markersize = 1, label='Other tenure type locations')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.suptitle(f'Tenure Type distribution for {year}') 
    plt.tight_layout() 
    plt.legend()
    mlai.write_figure(f'tenure-type-{year}.jpg', directory='./ml')


def plot_average_price_by_year():
    """
    Plot line graph of average price in a year
    """
    results = access.run_query_return_results("SELECT EXTRACT(year FROM date_of_transfer) AS year, AVG(price) AS average_price FROM prices_coordinates_data GROUP BY EXTRACT(year FROM date_of_transfer);")
    df = pd.DataFrame(results, columns=['year', 'average_price'])
    print(df)
    plt.plot(df['year'], df['average_price'])
    plt.title('Average property prices across the years')
    plt.xlabel('Year')
    plt.ylabel('Average Price')
    plt.show()


def plot_num_house_distribution(year):
    """
    Groups prices_coordinates_data into boxes, showing the count of property prices within each box
    """
    uk_gdf = get_uk_outline()

    prices_coordinates_df = access.togpd(access.get_prices_coordinates_df_by_year(year))

    gdf_grid = split_gdf_into_boxes(prices_coordinates_df)

    merged = gpd.sjoin(prices_coordinates_df, gdf_grid, how='left', op='within')
    merged['n_houses'] = 1
    count_merged = merged.dissolve(by='index_right', aggfunc={"n_houses": "count"})
    gdf_grid.loc[count_merged.index, 'n_houses'] = count_merged.n_houses.values
    ax = gdf_grid.plot(column='n_houses', figsize=(12, 8), cmap='viridis', vmax=5000, edgecolor="grey")
    plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=0, vmax=5000), cmap='viridis'),
             ax=ax, orientation='vertical')
    plt.autoscale(False)
    uk_gdf.to_crs(gdf_grid.crs).plot(ax = ax, alpha = 0.5)
