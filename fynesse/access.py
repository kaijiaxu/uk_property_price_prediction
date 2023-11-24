from .config import *
import yaml
from ipywidgets import interact_manual, Text, Password
import pymysql
import urllib.request
import pandas as pd
import geopandas as gpd
import osmnx as ox
import csv
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """


### Database Connection ###

def save_credentials():
    """ 
    Save credentials into a yaml file
    """
    @interact_manual(username=Text(description="Username:"),
                 password=Password(description="Password:"))
    def write_credentials(username, password):
        with open("credentials.yaml", "w") as file:
            credentials_dict = {'username': username,
                                'password': password}
            yaml.dump(credentials_dict, file)

def save_database_details():
    """ 
    Save credentials into a yaml file
    """
    @interact_manual(database=Text(description="Database URL:"),
                 port=Password(description="Port:"))
    def write_database_details(database, port):
        with open("database-details.yaml", "w") as file:
            database_dict = {'database': database,
                                'port': port}
            yaml.dump(database_dict, file)

def create_database_connection(user, password, host, database, port=3306):
    """ 
    Create a database connection to the MariaDB database
    specified by the host url and database name
    """
    conn = None
    try:
        conn = pymysql.connect(user=user, passwd=password, host=host, port=port, local_infile=1, db=database)
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

### Database operations ###

def run_query(query):
    """
    Run the given query on the database
    """
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
    with open("database-details.yaml") as file:
        database_details = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["database"]
    conn = create_database_connection(user=username, password=password, host=url, database="property_prices")
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()

def run_query_return_results(query):
    """
    Run the given query on the database and return results
    """
    with open("credentials.yaml") as file:
        credentials = yaml.safe_load(file)
    with open("database-details.yaml") as file:
        database_details = yaml.safe_load(file)
    username = credentials["username"]
    password = credentials["password"]
    url = database_details["database"]
    conn = create_database_connection(user=username, password=password, host=url, database="property_prices")
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    return rows


def load_data(filename, tablename):
    """
    Load data from a csv file to a database table 
    :param filename: name of csv file
    :param tablename: name of database table
    """
    run_query(f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE `{tablename}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"' LINES STARTING BY '' TERMINATED BY '\n';""")


### Price Paid Data ###

def load_ppdata_csvs(from_year, to_year):
    """
    Download Property Price Data onto local machine
    """
    for year in range(from_year, to_year + 1):
        for partnumber in range(1, 3):
            urllib.request.urlretrieve('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-part{partnumber}.csv'.format(year=year, partnumber=partnumber), 'pp-{year}-part{partnumber}.csv'.format(year=year, partnumber=partnumber))


def create_pp_data():
    """
    Create the schema for pp_data, including adding db_id
    """
    schema = ["CREATE TABLE IF NOT EXISTS `pp_data`",
            "(`transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,",
            "`price` int(10) unsigned NOT NULL,",
            "`date_of_transfer` date NOT NULL,",
            "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
            "`property_type` varchar(1) COLLATE utf8_bin NOT NULL,",
            "`new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,",
            "`tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,",
            "`primary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,",
            "`secondary_addressable_object_name` tinytext COLLATE utf8_bin NOT NULL,",
            "`street` tinytext COLLATE utf8_bin NOT NULL,",
            "`locality` tinytext COLLATE utf8_bin NOT NULL,",
            "`town_city` tinytext COLLATE utf8_bin NOT NULL,",
            "`district` tinytext COLLATE utf8_bin NOT NULL,",
            "`county` tinytext COLLATE utf8_bin NOT NULL,",
            "`ppd_category_type` varchar(2) COLLATE utf8_bin NOT NULL,",
            "`record_status` varchar(2) COLLATE utf8_bin NOT NULL,",
            "`db_id` bigint(20) unsigned NOT NULL)"
            "DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;"]
    schema = " ".join(schema)
    run_query("DROP TABLE IF EXISTS `pp_data`;")
    run_query(schema)
    run_query("ALTER TABLE `pp_data` ADD PRIMARY KEY (`db_id`);")
    run_query("ALTER TABLE `pp_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;")


### Postcode Data ###

def create_postcode_data():
    """
    Create the schema for postcode_data, including adding db_id
    """
    schema = ["CREATE TABLE IF NOT EXISTS `postcode_data`",
              "(`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
              "`status` enum('live','terminated') NOT NULL,",
              "`usertype` enum('small', 'large') NOT NULL,",
              "`easting` int unsigned,",
              "`northing` int unsigned,",
              "`positional_quality_indicator` int NOT NULL,",
              "`country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,",
              "`latitude` decimal(11,8) NOT NULL,",
              "`longitude` decimal(10,8) NOT NULL,",
              "`postcode_no_space` tinytext COLLATE utf8_bin NOT NULL,",
              "`postcode_fixed_width_seven` varchar(7) COLLATE utf8_bin NOT NULL,",
              "`postcode_fixed_width_eight` varchar(8) COLLATE utf8_bin NOT NULL,",
              "`postcode_area` varchar(2) COLLATE utf8_bin NOT NULL,",
              "`postcode_district` varchar(4) COLLATE utf8_bin NOT NULL,",
              "`postcode_sector` varchar(6) COLLATE utf8_bin NOT NULL,",
              "`outcode` varchar(4) COLLATE utf8_bin NOT NULL,",
              "`incode` varchar(3)  COLLATE utf8_bin NOT NULL,",
              "`db_id` bigint(20) unsigned NOT NULL)",
              "DEFAULT CHARSET=utf8 COLLATE=utf8_bin;"]
    schema = " ".join(schema)
    run_query("DROP TABLE IF EXISTS `postcode_data`;")
    run_query(schema)
    run_query("ALTER TABLE `postcode_data` ADD PRIMARY KEY (`db_id`);")
    run_query("ALTER TABLE `postcode_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;")


### Option A: Join on the fly ###

def join_on_the_fly(min_year, max_year, property_type, north, south, east, west):
    """
    Return a pandas dataframe after joining on the fly based on the date_of_transfer, property_type, and bounding box
    """
    # For slightly better readability, join_query was separated:
    join_query = [
        "SELECT pp_data.`price`, pp_data.`date_of_transfer`, pp_data.`postcode`, pp_data.`property_type`, pp_data.`new_build_flag`, pp_data.`tenure_type`, pp_data.`locality`, pp_data.`town_city`, pp_data.`district`, pp_data.`county`, postcode_data.`country`, postcode_data.`latitude`, postcode_data.`longitude`, pp_data.`db_id` FROM pp_data",
        "INNER JOIN postcode_data ON pp_data.`postcode` = postcode_data.`postcode`",
        f"WHERE pp_data.`property_type` = '{property_type}' AND pp_data.`date_of_transfer` >= '{min_year}-01-01' AND pp_data.`date_of_transfer` <= '{max_year}-12-31' AND postcode_data.`latitude` <= {north} AND postcode_data.`latitude` >= {south} AND postcode_data.`longitude` >= {west} AND postcode_data.`longitude` <= {east}"]
    join_query = " ".join(join_query)
    results = run_query_return_results(join_query)
    df = pd.DataFrame(results, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude', 'db_id'])
    print(f'Successfully joined on the fly\n')
    return df


### Option B: Prices Coordinates Data ###

def create_indices():
    """
    Index both the `pp_data` and `postcode_data` table.
    I suspect only the ppdata_date_postcode and postcodedata_postcode are needed, but I created indices for all fields in `prices_coordinates_data`.
    """
    query = [
        "CREATE INDEX IF NOT EXISTS ppdata_price ON pp_data (price);",
        "CREATE INDEX IF NOT EXISTS ppdata_date_of_transfer ON pp_data (date_of_transfer);",
        "CREATE INDEX IF NOT EXISTS ppdata_property_type ON pp_data (property_type);",
        "CREATE INDEX IF NOT EXISTS ppdata_new_build_flag ON pp_data (new_build_flag);",
        "CREATE INDEX IF NOT EXISTS ppdata_tenure_type ON pp_data (tenure_type);",
        "CREATE INDEX IF NOT EXISTS ppdata_locality ON pp_data (locality);",
        "CREATE INDEX IF NOT EXISTS ppdata_town_city ON pp_data (town_city);",
        "CREATE INDEX IF NOT EXISTS ppdata_district ON pp_data (district);",
        "CREATE INDEX IF NOT EXISTS ppdata_county ON pp_data (county);",
        "CREATE INDEX IF NOT EXISTS ppdata_postcode ON pp_data (postcode);",
        "CREATE INDEX IF NOT EXISTS postcodedata_postcode ON postcode_data (postcode);",
        "CREATE INDEX IF NOT EXISTS postcodedata_country ON postcode_data (country);",
        "CREATE INDEX IF NOT EXISTS postcodedata_latitude ON postcode_data (latitude);",
        "CREATE INDEX IF NOT EXISTS postcodedata_longitude ON postcode_data (longitude);"
    ]
    for line in query:
        run_query(line)


def join_one_year(year):
    """
    Join `pp_data` and `postcode_data` by year, and save the results into a csv
    """
    # For slightly better readability, join_query was separated:
    join_query = [
        "SELECT pp_data.`price`, pp_data.`date_of_transfer`, pp_data.`postcode`, pp_data.`property_type`, pp_data.`new_build_flag`, pp_data.`tenure_type`, pp_data.`locality`, pp_data.`town_city`, pp_data.`district`, pp_data.`county`, postcode_data.`country`, postcode_data.`latitude`, postcode_data.`longitude`, pp_data.`db_id` FROM `pp_data` INNER JOIN `postcode_data`",
        "ON pp_data.`postcode` = postcode_data.`postcode`",
        f"WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31'"]
    join_query = " ".join(join_query)
    results = run_query_return_results(join_query)
    fp = open(f'joined-{year}.csv', 'w')
    myFile = csv.writer(fp)
    myFile.writerows(results)
    fp.close()
    print(f'Successfully joined {year}\n')
    return results


def create_prices_coordinates_data():
    """
    Create the schema for prices_coordinates_data
    """
    schema = ["CREATE TABLE IF NOT EXISTS `prices_coordinates_data`",
              "(`price` int(10) unsigned NOT NULL,",
              "`date_of_transfer` date NOT NULL,",
              "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
              "`property_type` varchar(1) COLLATE utf8_bin NOT NULL,",
              "`new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,",
              "`tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,",
              "`locality` tinytext COLLATE utf8_bin NOT NULL,",
              "`town_city` tinytext COLLATE utf8_bin NOT NULL,",
              "`district` tinytext COLLATE utf8_bin NOT NULL,",
              "`county` tinytext COLLATE utf8_bin NOT NULL,",
              "`country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,",
              "`latitude` decimal(11,8) NOT NULL,",
              "`longitude` decimal(10,8) NOT NULL,",
              "`db_id` bigint(20) unsigned NOT NULL)",
              "DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;"]
    schema = " ".join(schema)
    run_query("DROP TABLE IF EXISTS `prices_coordinates_data`;")
    run_query(schema)


def get_prices_coordinates_df_by_coordinates(north, south, east, west):
    """
    Return a pandas dataframe of `prices_coordinates_data` filtered by a bounding box
    """
    sql_query = f"SELECT price, latitude, longitude FROM prices_coordinates_data WHERE latitude >= {south} AND latitude <= {north} AND longitude >= {west} AND longitude <= {east}"
    prices_coordinates_df = pd.DataFrame(run_query_return_results(sql_query), columns=['price', 'latitude', 'longitude'])
    return prices_coordinates_df


def get_prices_coordinates_df_by_year(year):
    """
    Return a pandas dataframe of `prices_coordinates_data` filtered by date_of_transfer
    """
    sql_query = f"SELECT price, latitude, longitude FROM prices_coordinates_data WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31'"
    prices_coordinates_df = pd.DataFrame(run_query_return_results(sql_query), columns=['price', 'latitude', 'longitude'])
    return prices_coordinates_df


def get_prices_coordinates_df_for_prediction(min_year, max_year, property_type, north, south, east, west):
    """
    Return a pandas dataframe of `prices_coordinates_data` filtered by date_of_transfer, property_type, and a bounding box
    """
    sql_query = f"SELECT * FROM prices_coordinates_data WHERE `property_type` = '{property_type}' AND `date_of_transfer` >= '{min_year}-01-01' AND `date_of_transfer` <= '{max_year}-12-31' AND `latitude` >= {south} AND `latitude` <= {north} AND `longitude` >= {west} AND `longitude` <= {east}"
    results = run_query_return_results(sql_query)
    df = pd.DataFrame(results, columns=['price', 'date_of_transfer', 'postcode', 'property_type', 'new_build_flag', 'tenure_type', 'locality', 'town_city', 'district', 'county', 'country', 'latitude', 'longitude', 'db_id'])
    return df


def create_indices_prices_coordinates():
    """
    Index the prices_coordinates_data table
    """
    query = [
        "CREATE INDEX IF NOT EXISTS pc_price ON prices_coordinates_data (price);",
        "CREATE INDEX IF NOT EXISTS pc_latitude ON prices_coordinates_data (latitude);",
        "CREATE INDEX IF NOT EXISTS pc_longitude ON prices_coordinates_data (longitude);"
    ]
    for line in query:
        run_query(line)

### Open Street Map data ###

def get_pois(north, south, east, west, tags):
  """
  Returns points of interest based on bounding box and tags
  """
  pois_df = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
  pois_df.crs = "EPSG:4326"
  try:
    pois_df = ox.features_from_bbox(north, south, east, west, tags)
    # Convert Polygons, etc. to Points
    pois_df['geometry'] = pois_df['geometry'].apply(lambda x: x.centroid)
    return pois_df
  except:
    return pois_df


def get_bounding_box(latitude, longitude, box_height, box_width):
  """
  Return edges of the bounding box based on the centre location, box height, and box width.
  :param latitude: latitude of centre point
  :param longitude: longitude of centre point
  :param box_height: total bounding box height
  :param box_width: total bounding box width
  """
  north = latitude + box_height/2
  south = latitude - box_height/2
  west = longitude - box_width/2
  east = longitude + box_width/2
  return (north, south, east, west)


def togpd(df):
    """
    Convert the given pandas dataframe to a geopandas dataframe 
    """
    geometry = gpd.points_from_xy(df.longitude, df.latitude)
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.crs = "EPSG:4326"
    return df
