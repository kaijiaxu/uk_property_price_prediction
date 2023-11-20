# This file contains code for suporting addressing questions in the data
from .config import *
from . import access

from datetime import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
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


def num_of_pois(prices_coord_gdf, osm_key, osm_value, neighbourhood_size):
    """
    Adds a column to `prices_coord_gdf` stating the number of POIs with the specific `tag` in OSM, within the `neighbourhood_size` from the given `latitude` and `longitude`.
    """
    prices_coord_gdf_copy = gpd.GeoDataFrame(prices_coord_gdf.copy(deep=True))
    prices_coord_gdf_copy['number of ' + osm_key + '-' + osm_value + ' in neighbourhood'] = prices_coord_gdf_copy['geometry'].apply(lambda house: len((access.get_pois(house.y + neighbourhood_size/2, house.y - neighbourhood_size/2, house.x + neighbourhood_size/2, house.x - neighbourhood_size/2, {osm_key: osm_value})).notna()))
    return prices_coord_gdf_copy


def generate_all_osm_columns(prices_coordinates_data_df, osm_tags, neighbourhood_size):
    for osm_key in osm_tags:
        for osm_value in osm_tags[osm_key]:
            prices_coordinates_data_df = num_of_pois(prices_coordinates_data_df, osm_key, osm_value, neighbourhood_size)
    return prices_coordinates_data_df


### Address functions ###

# Suggested values for bbox_size and osm_tags parameters in predict_price
bbox_size = 0.1

osm_tags = {
    "amenity": ["bar", "fast_food", "pub", "restaurant", "kindergarten", "school", "bus_station"],
    "public_transport": ["station"],
    "shop": ["convenience", "deparment_store", "supermarket"],
    "tourism": [True]
}

def predict_price(latitude, longitude, date, property_type, bbox_size, osm_tags):
    """Price prediction for UK housing."""
    # Bounding box size for calculating OSM features 
    neighbourhood_size = 0.02

    # 1. Select a bounding box around the housing location in latitude and longitude.
    (north, south, east, west) = access.get_bounding_box(latitude, longitude, bbox_size, bbox_size)

    # 2. Select a data range around the prediction date.
    (date_year, date_month, date_day) = validate_parse_date(date)

    # 3. Use the data ecosystem you have build above to build a training set from the relevant time period and location in the UK. Include appropriate features from OSM to improve the prediction.
    prices_coordinates_data_df = access.togpd(access.get_prices_coordinates_df_for_prediction(date_year, date_year, property_type, north, south, east, west))

    if len(prices_coordinates_data_df) < 10: # Ensures testing_data is of at least length 2, so that we get a r2 score
        print("Not enough training data. Try increasing bbox_size.")
        return None
    
    # Incorporate features from OSM
    prices_coordinates_data_df = generate_all_osm_columns(prices_coordinates_data_df, osm_tags, neighbourhood_size)

    # Split data into training and validation set
    training_data, testing_data = train_test_split(prices_coordinates_data_df, test_size=0.2)

    # 4. Train a linear model on the data set you have created.
    design = np.column_stack((
        np.where(training_data['new_build_flag'] == 'Y', 1, 0), 
        np.where(training_data['new_build_flag'] == 'N', 1, 0), 
        np.where(training_data['tenure_type'] == 'F', 1, 0), 
        np.where(training_data['tenure_type'] == 'L', 1, 0), 
                             ))

    m = sm.OLS(np.array(training_data['price']), design)
    results = m.fit()
    print(results.summary())

    # 5. Validate the quality of the model.
    design_test = np.column_stack((
        np.where(testing_data['new_build_flag'] == 'Y', 1, 0), 
        np.where(testing_data['new_build_flag'] == 'N', 1, 0), 
        np.where(testing_data['tenure_type'] == 'F', 1, 0), 
        np.where(testing_data['tenure_type'] == 'L', 1, 0), 
                             ))
    
    test_results = results.predict(design_test)
    r2 = r2_score(np.array(testing_data['price']), np.array(test_results))
    print(f"R-squared: {r2:.6f}\n")

    if r2 < 0.4 or len(testing_data['price']) <= 1:
        print("Warning: prediction quality is poor. Consider ways to improve the model - use more/different OSM tags and increase bounding box size.\n")

    # 6. Provide a prediction of the price from the model, warning appropriately if your validation indicates the quality of the model is poor.

    # Build design matrix to predict price
    avg_new_build = len(prices_coordinates_data_df[prices_coordinates_data_df['new_build_flag'] == 'Y']) / len(prices_coordinates_data_df)
    avg_tenure_freehold = len(prices_coordinates_data_df[prices_coordinates_data_df['tenure_type'] == 'F']) / len(prices_coordinates_data_df)
    avg_tenure_lease = len(prices_coordinates_data_df[prices_coordinates_data_df['tenure_type'] == 'L']) / len(prices_coordinates_data_df)

    prediction_df = pd.DataFrame({'latitude': [latitude], 'longitude': [longitude], 'new_build': [avg_new_build], 'old_build': 1 - avg_new_build, 'freehold': [avg_tenure_freehold], 'lease': avg_tenure_lease})
    prediction_df = access.togpd(prediction_df)
    prediction_df = generate_all_osm_columns(prediction_df, osm_tags, neighbourhood_size)
    design_pred = prediction_df.to_numpy()

    pred_price = results.predict(design_pred)
    print(f"The predicted price for a house at latitude={latitude}, logitude={longitude}, of property type {property_type} on {date} is predicted to be of £{pred_price[0]}.\n")

    return pred_price[0], r2