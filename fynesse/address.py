# This file contains code for suporting addressing questions in the data
from .config import *
from . import access
from . import assess

from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from statsmodels.api import add_constant
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import spatial
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""

### Helper functions ###
def validate_parse_date(date_text):
    # If date_text is not of the correct format, strptime will throw a ValueError
    dt = datetime.strptime(date_text, '%Y-%m-%d')
    return (dt.year, dt.month, dt.day)

def create_KDTree(gdf):
    """
    Create a KDTree from the given geodataframe, which speeds up the data processing and computation for features in consideration.
    """
    if len(gdf) == 0:
        return None
    # Convert geometry into a coordinates array
    longitude = gdf.geometry.apply(lambda x: x.x).values
    latitude = gdf.geometry.apply(lambda x: x.y).values
    coordinates = list(zip(latitude, longitude))

    # Create the KDTree
    tree = spatial.KDTree(coordinates)
    return tree

def closest_osm_features(tree, coordinates, top_k=[50], max_distance=0.2):
    """
    Returns a maximum of k rows of OSM data which are within a boundary of max_distance from the coordinates of the point we are interested in
    :param tree: the KDTree of the geopandas dataframe containing all OSM data that we want to filter
    :param coordinates: the coordinates of the point we are interested in
    :param top_k: maximum number of POIs we want to return (in a list -- to prevent special case of top_k = 1)
    :param max_distance: maximum distance of POI from the point we are interested in
    """
    if tree is None:
        return []
    results = tree.query((coordinates), k=top_k, distance_upper_bound=max_distance)
    zipped_results = list(zip(results[0], results[1]))
    # Removes missing neighbours (number of neighbours < top_k), which have infite distances
    zipped_results = [i for i in zipped_results if i[0] != np.inf]
    return zipped_results


def num_of_pois(prices_coord_gdf, pois_tree, neighbourhood_size, tag_name):
    """
    Adds a column to `prices_coord_gdf` stating the number of POIs with the specific `tag` in OSM, within the `neighbourhood_size` from the given `latitude` and `longitude`.
    :param prices_coord_gdf: the geopandas dataframe to append the column to
    :param pois_tree: KDTree of POIs
    :param neighbourhood_size: the bounding distance for POIs
    :param tag_name: used to generate the column name
    """
    prices_coord_gdf['number of ' + tag_name + ' in neighbourhood'] = prices_coord_gdf['geometry'].apply(lambda house: len(closest_osm_features(pois_tree, (house.y, house.x), max_distance=neighbourhood_size)))
    return prices_coord_gdf


def calculate_min_dist(pois_tree, coordinates, bbox_size):
    """
    Compute the minimum distance from coordinates to POIs
    :param pois_tree: KDTree of POIs
    :param coordinates: coordinates of the reference point
    :param bbox_size: used to generate the maximum distance
    """
    if pois_tree is None:
        return bbox_size * 40000/360
    results = closest_osm_features(pois_tree, coordinates, top_k=[1])
    if len(results) == 0:
        return bbox_size * 40000/360
    return results[0][0] * 40000/360


def min_dist_to_poi(prices_coord_gdf, pois_tree, osm_tag_name, bbox_size):
    """
    Adds a column to `prices_coord_gdf` stating the minimum distance to the specific `tag` in OSM, from the given `latitude` and `longitude`.
    :param prices_coord_gdf: the geopandas dataframe to append the column to
    :param pois_tree: KDTree of POIs
    :param tag_name: used to generate the column name
    :param bbox_size: used to generate the maximum distance
    """
    prices_coord_gdf['min distance to ' + osm_tag_name] = prices_coord_gdf['geometry'].apply(lambda house: calculate_min_dist(pois_tree, (house.y, house.x), bbox_size))
    return prices_coord_gdf


def generate_all_osm_columns(prices_coordinates_data_df, osm_tags, neighbourhood_size, bbox_size, latitude, longitude):
    """
    Adds all the columns to the dataframe based on osm_tags
    """
    if osm_tags is not None:
        for osm_key in osm_tags:
            for osm_value in osm_tags[osm_key]:
                (north, south, east, west) = access.get_bounding_box(latitude, longitude, bbox_size, bbox_size)
                pois_df = access.get_pois(north, south, east, west, {osm_key: osm_value})
                pois_tree = create_KDTree(pois_df)
                osm_tag_name = str(osm_key) + '-' + str(osm_value)
                prices_coordinates_data_df = num_of_pois(prices_coordinates_data_df, pois_tree, neighbourhood_size, osm_tag_name)
                prices_coordinates_data_df = min_dist_to_poi(prices_coordinates_data_df, pois_tree, osm_tag_name, bbox_size)
    return prices_coordinates_data_df


def build_df(latitude, longitude, date, property_type, bbox_size, osm_tags, neighbourhood_size, date_range=1):
    # 1. Select a bounding box around the housing location in latitude and longitude.
    (north, south, east, west) = access.get_bounding_box(latitude, longitude, bbox_size, bbox_size)

    # 2. Select a data range around the prediction date.
    (date_year, date_month, date_day) = validate_parse_date(date)

    # 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    prices_coordinates_data_df = access.togpd(access.get_prices_coordinates_df_for_prediction(date_year - date_range, date_year + date_range, property_type, north, south, east, west))

    if len(prices_coordinates_data_df) < 10: # Ensures testing_data is of at least length 2, so that we get a r2 score
        print("Not enough training data. Try increasing bbox_size or date_range.")
        return None
    
    prices_coordinates_data_df['new_build'] = prices_coordinates_data_df['new_build_flag'].apply(lambda x: 1 if x == 'Y' else 0)
    prices_coordinates_data_df['freehold'] = prices_coordinates_data_df['tenure_type'].apply(lambda x: 1 if x == 'F' else 0)
    prices_coordinates_data_df['num_of_year'] = prices_coordinates_data_df['date_of_transfer'].apply(lambda x: x.year - 1995)
    prices_coordinates_data_df['num_of_month'] = prices_coordinates_data_df['date_of_transfer'].apply(lambda x: x.month)
    
    # Incorporate features from OSM
    if osm_tags is not None:
        prices_coordinates_data_df = generate_all_osm_columns(prices_coordinates_data_df, osm_tags, neighbourhood_size, bbox_size, latitude, longitude)
    return prices_coordinates_data_df


def build_design_matrix(df, osm_tags):
    """
    Builds the design matrix for the model
    """
    df['const'] = 1
    column_names = ['const', 'new_build', 'freehold', 'num_of_year', 'num_of_month'] 
    if osm_tags is not None:
        for osm_key in osm_tags:
            for osm_value in osm_tags[osm_key]:
                osm_tag_name = str(osm_key) + '-' + str(osm_value)
                column_names += ['number of ' + osm_tag_name + ' in neighbourhood']
                column_names += ['min distance to ' + osm_tag_name]
    return df[column_names]


### Address functions ###

def correlation(latitude, longitude, date, property_type, bbox_size, osm_tags, neighbourhood_size, column_names):
    """
    Print the correlation matrix to help decide features for the model
    :param column_names: array of column names for correlation matrix
    """
    df = build_df(latitude, longitude, date, property_type, bbox_size, osm_tags, neighbourhood_size)
    print(df[column_names].corr())


# Example values for bbox_size and osm_tags parameters in predict_price

neighbourhood_size = 0.04

bbox_size = 0.5

osm_tags = {
    "amenity": ["restaurant", "kindergarten", "school", "bus_station"],
    "public_transport": ["platform", "stop_position"],
    "shop": ["convenience", "supermarket"],
    "office": [True]
}

def predict_price(latitude, longitude, date, property_type, bbox_size=bbox_size, neighbourhood_size=neighbourhood_size, osm_tags=osm_tags, date_range=1):
    """Price prediction for UK housing."""

    if bbox_size < neighbourhood_size:
        raise ValueError("Please input a neighbourhood size that's smaller than or equal to bbox_size.")
    
    (date_year, date_month, date_day) = validate_parse_date(date)

    # 1. Select a bounding box around the housing location in latitude and longitude.
    # 2. Select a data range around the prediction date.
    # 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    prices_coordinates_data_df = build_df(latitude, longitude, date, property_type, bbox_size, osm_tags, neighbourhood_size, date_range)

    if prices_coordinates_data_df is None:
        return None
    
    # Split data into training and validation set
    training_data, testing_data = train_test_split(prices_coordinates_data_df, test_size=0.2)
    print(f"Size of training set: {len(training_data)}\n")

    # 4. Train a linear model on the data set you have created.
    design = build_design_matrix(training_data, osm_tags)

    m = sm.OLS(training_data['price'], design)
    results = m.fit()
    print(results.summary())

    # 5. Validate the quality of the model.
    design_test = build_design_matrix(testing_data, osm_tags)

    test_results = results.predict(design_test)
    r2 = r2_score(testing_data['price'], test_results)
    print(f"R-squared value using testing data: {r2:.6f}\n")

    if r2 < 0.4 or len(testing_data['price']) <= 1:
        print("Warning: prediction quality is poor. Consider ways to improve the model - use more/different OSM tags and increase bounding box size.\n")

    # 6. Provide a prediction of the price from the model, warning appropriately if your validation indicates the quality of the model is poor.

    # Build design matrix to predict price
    avg_new_build = len(prices_coordinates_data_df[prices_coordinates_data_df['new_build_flag'] == 'Y']) / len(prices_coordinates_data_df)
    avg_tenure_freehold = len(prices_coordinates_data_df[prices_coordinates_data_df['tenure_type'] == 'F']) / len(prices_coordinates_data_df)

    prediction_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude], 'new_build': [avg_new_build], 'freehold': [avg_tenure_freehold], 'num_of_year': date_year-1995, 'num_of_month': date_month})
    prediction_df = access.togpd(prediction_df)
    prediction_df = generate_all_osm_columns(prediction_df, osm_tags, neighbourhood_size, bbox_size, latitude, longitude)
    design_pred = build_design_matrix(prediction_df, osm_tags)

    pred_price = results.predict(design_pred)
    print(f"The predicted price for a house at latitude={latitude}, logitude={longitude}, of property type {property_type} on {date} is predicted to be of £{pred_price[0]:.2f}.\n")

    return pred_price[0], r2


def pca_analysis(design_matrix, n_components):
    """
    Perform PCA analysis
    """
    X = design_matrix - design_matrix.mean(axis=0)
    pca = PCA(n_components=n_components)
    pca.fit_transform(X)
    print("PCA components: \n")
    print(pca.components_)
    print("Percentage of variance explained by selected components: \n")
    print(sum(pca.explained_variance_ratio_))