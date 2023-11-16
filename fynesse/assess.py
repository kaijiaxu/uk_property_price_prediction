from .config import *

from . import access

import osmnx as ox
import matplotlib.pyplot as plt
# import mlai
# import mlai.plot as plot

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

# def view(north, south, east, west, tags):
#     """Provide a view of the data that allows the user to verify some aspect of its quality."""
#     graph = ox.graph_from_bbox(north, south, east, west)
#     # Retrieve nodes and edges
#     nodes, edges = ox.graph_to_gdfs(graph)
#     area = ox.geocode_to_gdf("United Kingdom")
#     fig, ax = plt.subplots(figsize=plot.big_figsize)
#     # Plot the footprint
#     area.plot(ax=ax, facecolor="white")
#     # Plot street edges
#     edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")
#     ax.set_xlim([west, east])
#     ax.set_ylim([south, north])
#     ax.set_xlabel("longitude")
#     ax.set_ylabel("latitude")

#     # Plot all POIs
#     pois = access.get_pois(north, south, east, west, tags)
#     pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
#     plt.tight_layout()
#     mlai.write_figure(directory="./maps", filename="pois.jpg")

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
