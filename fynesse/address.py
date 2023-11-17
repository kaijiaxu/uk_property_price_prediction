# This file contains code for suporting addressing questions in the data
from . import access

import datetime
import numpy as np
import statsmodels.api as sm

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
    try:
        datetime.date.fromisoformat(date_text)
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")
    dt = datetime.strptime(date_text, '%Y-%m-%d')
    return (dt.year, dt.month, dt.day)


def mean_distance(gdf_houses, gdf_osm_feature):
    return gdf_houses['geometry'].apply(lambda house: gdf_osm_feature.distance(house).mean())

def min_distance(gdf_houses, gdf_osm_feature):
    return gdf_houses['geometry'].apply(lambda house: gdf_osm_feature.distance(house).min())

### Address functions ###

def predict_price(latitude, longitude, date, property_type):
    """Price prediction for UK housing."""
    # Define some constants that can be used to tweak the model
    bbox_size = 0.2
    osm_tags = {}
    
    # 1. Select a bounding box around the housing location in latitude and longitude.
    (north, south, west, east) = access.get_bounding_box(latitude, longitude, bbox_size, bbox_size)

    # 2. Select a data range around the prediction date.
    (date_year, date_month, date_day) = validate_parse_date(date)
    prices_coordinates_data_df = access.join_on_the_fly(date_year - 1, date_year + 1, property_type, south, north, east, west)

    # 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    if len(prices_coordinates_data_df):
        print("No training data. Try increasing bbox_size.")
        return None
    
    pois = access.get_pois(north, south, east, west, osm_tags)

    # get average distance to schools, average and min distance to food, healthcare, etc. from OSM and append to prices_coordinates_data
    for osm_key in osm_tags.keys():
        osm_gdf = pois[pois[osm_key].notnull()]["geometry"]
        osm_gdf = osm_gdf[osm_gdf.notnull()]
        prices_coordinates_data_df['mean distance to ' + osm_key] = mean_distance(prices_coordinates_data_df, osm_gdf)
        prices_coordinates_data_df['min distance to ' + osm_key] = min_distance(prices_coordinates_data_df, osm_gdf)

    # 4. Train a linear model on the data set you have created.
    design = np.concatenate((
        np.where(prices_coordinates_data_df['new_build_flag'] == 'Y', 1, 0), 
        np.where(prices_coordinates_data_df['new_build_flag'] == 'N', 1, 0), 
        np.where(prices_coordinates_data_df['tenure_type'] == 'F', 1, 0), 
        np.where(prices_coordinates_data_df['tenure_type'] == 'L', 1, 0), 
                             ),axis=1)
    for osm_key in osm_tags.keys():
        design = np.concatenate((
            design,
            prices_coordinates_data_df['mean distance to ' + osm_key],
            prices_coordinates_data_df['min distance to ' + osm_key]
            ), axis = 1)
    m = sm.OLS(prices_coordinates_data_df['price'], design)
    results = m.fit()
    # x_pred = np.linspace(-5,20,200).reshape(-1,1)
    # design_pred = np.concatenate((np.sin(x_pred), np.sin(x_pred**2/40), x_pred),axis=1)
    
    # 5. Validate the quality of the model.

    # 6. Provide a prediction of the price from the model, warning appropriately if your validation indicates the quality of the model is poor.
    pass