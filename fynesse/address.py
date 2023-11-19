# This file contains code for suporting addressing questions in the data
from .config import *
from . import access

from datetime import datetime
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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


def mean_distance(gdf_houses, gdf_osm_feature):
    return gdf_houses['geometry'].apply(lambda house: gdf_osm_feature.distance(house).mean())

def min_distance(gdf_houses, gdf_osm_feature):
    return gdf_houses['geometry'].apply(lambda house: gdf_osm_feature.distance(house).min())

### Address functions ###

# Suggested values for bbox_size and osm_tags parameters in predict_price
bbox_size = 0.02
osm_tags = {
    "amenity": "bar",
    "amenity": "fast_food",
    "amenity": "pub",
    "amenity": "restaurant",
    "amenity": "kindergarten",
    "amenity": "school",
    "amenity": "bus_station",
    "public_transport": "station",
    "shop": "convenience",
    "shop": "deparment_store",
    "shop": "supermarket",
    "tourism": True,
}

def predict_price(latitude, longitude, date, property_type, bbox_size, osm_tags):
    """Price prediction for UK housing."""
    
    # 1. Select a bounding box around the housing location in latitude and longitude.
    (north, south, west, east) = access.get_bounding_box(latitude, longitude, bbox_size, bbox_size)

    # 2. Select a data range around the prediction date.
    (date_year, date_month, date_day) = validate_parse_date(date)

    # 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    prices_coordinates_data_df = access.join_on_the_fly(date_year - 1, date_year + 1, property_type, south, north, east, west)

    if len(prices_coordinates_data_df):
        print("No training data. Try increasing bbox_size.")
        return None
    
    pois = access.get_pois(north, south, east, west, osm_tags)

    # Compute average distance and min distance to features from OSM and append to prices_coordinates_data
    for (osm_key, osm_value) in osm_tags.items():
        osm_gdf = pois[pois[osm_key].notnull()]["geometry"]
        osm_gdf = osm_gdf[osm_gdf.notnull()]
        prices_coordinates_data_df['mean distance to ' + osm_key + '-' + osm_value] = mean_distance(prices_coordinates_data_df, osm_gdf)
        prices_coordinates_data_df['min distance to ' + osm_key + '-' + osm_value] = min_distance(prices_coordinates_data_df, osm_gdf)

    # Split data into training and validation set
    training_data, testing_data = train_test_split(prices_coordinates_data_df, test_size=0.2)

    # 4. Train a linear model on the data set you have created.
    design = np.concatenate((
        np.where(training_data['new_build_flag'] == 'Y', 1, 0), 
        np.where(training_data['new_build_flag'] == 'N', 1, 0), 
        np.where(training_data['tenure_type'] == 'F', 1, 0), 
        np.where(training_data['tenure_type'] == 'L', 1, 0), 
                             ),axis=1)
    
    for osm_key in osm_tags.keys():
        design = np.concatenate((
            design,
            training_data['mean distance to ' + osm_key],
            training_data['min distance to ' + osm_key]
            ), axis = 1)
    m = sm.OLS(training_data['price'], design)
    results = m.fit_regularized(alpha=0.10,L1_wt=0.0)


    # 5. Validate the quality of the model.
    design_test = np.concatenate((
        np.where(testing_data['new_build_flag'] == 'Y', 1, 0), 
        np.where(testing_data['new_build_flag'] == 'N', 1, 0), 
        np.where(testing_data['tenure_type'] == 'F', 1, 0), 
        np.where(testing_data['tenure_type'] == 'L', 1, 0), 
                             ),axis=1)
    
    for (osm_key, osm_value) in osm_tags.items():
        design_test = np.concatenate((
            design_test,
            testing_data['mean distance to ' + osm_key + '-' + osm_value],
            testing_data['min distance to ' + osm_key + '-' + osm_value]
            ), axis = 1)
    test_results = m.predict(design_test)

    r2 = r2_score(testing_data['price'], test_results)
    print(f"R-squared: {r2:.6f}\n")

    if r2 < 0.4:
        print("Warning: prediction quality is poor. Consider ways to improve the model - use more/different OSM tags and increase bounding box size.\n")

    # 6. Provide a prediction of the price from the model, warning appropriately if your validation indicates the quality of the model is poor.

    # Build design matrix to predict price
    avg_new_build = len(prices_coordinates_data_df[prices_coordinates_data_df['new_build_flag'] == 'Y']) / len(prices_coordinates_data_df)
    avg_tenure_freehold = len(prices_coordinates_data_df[prices_coordinates_data_df['tenure_type'] == 'F']) / len(prices_coordinates_data_df)
    avg_tenure_lease = len(prices_coordinates_data_df[prices_coordinates_data_df['tenure_type'] == 'L']) / len(prices_coordinates_data_df)

    design_pred = [avg_new_build, 1 - avg_new_build, avg_tenure_freehold, avg_tenure_lease]

    for (osm_key, osm_value) in osm_tags.items():
        design_pred = np.concatenate(
            design_pred,
            prices_coordinates_data_df['mean distance to ' + osm_key + '-' + osm_value],
            prices_coordinates_data_df['min distance to ' + osm_key + '-' + osm_value]
            )

    pred_price = results.predict(design_pred)
    print(f"The predicted price for a house at latitude={latitude}, logitude={longitude}, of property type {property_type} on {date} is predicted to be of £{pred_price}.\n")

    return pred_price, r2