from .config import *

from . import access

import osmnx as ox
import matplotlib.pyplot as plt
import mlai
import mlai.plot as plot
import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact
from geopandas.tools import sjoin
from IPython.display import display

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data_validation_df(latitude, longitude):
    """
    Matches property data from OSM to prices_coordinates_data
    """
    # Query OSM for residential data. Includes 'apartments', 'terrace', 'house', 'detached' or 'semidetached_house'.
    tags = {"residential": True}
    (north, south, west, east) = access.get_bounding_box(latitude, longitude, 0.02, 0.02)
    osm_df = access.get_pois(north, south, east, west, tags)
    print(f"Number of properties from OSM: {len(osm_df)}")
    # Query prices_coordinates_data
    prices_coordinates_df = access.get_prices_coordinates_df(north, south, east, west)
    print(f"Number of properties from prices_coordinates_data: {len(prices_coordinates_df)}")
    # Join the two dataframes based on geometry
    joined_df = sjoin(osm_df, prices_coordinates_df, how='inner')
    print(f"Number of matches: {len(joined_df)}")
    return display(joined_df)

def data_validation_interaction():
    """
    Allows interaction in the Jupyter Notebook
    """
    latitude_slider = widgets.FloatSlider(min=49.89517100, max=55.79741500, step=0.02, value=50)
    longitude_slider = widgets.FloatSlider(min=-6.35264700, max=1.76277300, step=0.02, value=0)
    _ = interact(data_validation_df, 
            latitude=latitude_slider,
            longitude=longitude_slider)

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
