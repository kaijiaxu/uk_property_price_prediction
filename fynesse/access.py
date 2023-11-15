from .config import *
import yaml
from ipywidgets import interact_manual, Text, Password
import pymysql
import urllib.request
import pandas as pd
import osmnx as ox
import csv

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

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

def load_ppdata_csvs(from_year, to_year):
    """
    Download Property Price Data 
    """
    for year in range(from_year, to_year + 1):
        for partnumber in range(1, 3):
            urllib.request.urlretrieve('http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-{year}-part{partnumber}.csv'.format(year=year, partnumber=partnumber), 'pp-{year}-part{partnumber}.csv'.format(year=year, partnumber=partnumber))

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


def create_pp_data():
    """
    Create the schema for pp_data, including adding db_id.
    """
    schema = ["DROP TABLE IF EXISTS `pp_data`;",    
            "CREATE TABLE IF NOT EXISTS `pp_data` (",
            "`transaction_unique_identifier` tinytext COLLATE utf8_bin NOT NULL,",
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
            "`db_id` bigint(20) unsigned NOT NULL"
            ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1;",
            "ALTER TABLE `pp_data` ADD PRIMARY KEY (`db_id`);",
            "ALTER TABLE `pp_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;"]
    schema = " ".join(schema)
    run_query(schema)


def create_postcode_data():
    """
    Create the schema for postcode_data, including adding db_id.
    """
    schema = ["DROP TABLE IF EXISTS `postcode_data`;",
              "CREATE TABLE IF NOT EXISTS `postcode_data` (",
              "`postcode` varchar(8) COLLATE utf8_bin NOT NULL,",
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
              "`db_id` bigint(20) unsigned NOT NULL",
              ") DEFAULT CHARSET=utf8 COLLATE=utf8_bin;",
              "ALTER TABLE `postcode_data` ADD PRIMARY KEY (`db_id`);",
              "ALTER TABLE `postcode_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;"]
    schema = " ".join(schema)
    run_query(schema)


def create_indices():
    """
    Index both the `pp_data` and `postcode_data` table.
    I suspect only the ppdata_date_of_transfer, ppdata_postcode and postcodedata_postcode are needed, but I created indices for all fields in `prices_coordinates_data`.
    """
    query = [
        "CREATE INDEX ppdata_price ON pp_data (price);",
        "CREATE INDEX ppdata_date_of_transfer ON pp_data (date_of_transfer);",
        "CREATE INDEX ppdata_property_type ON pp_data (property_type);",
        "CREATE INDEX ppdata_new_build_flag ON pp_data (new_build_flag);",
        "CREATE INDEX ppdata_tenure_type ON pp_data (tenure_type);",
        "CREATE INDEX ppdata_locality ON pp_data (locality);",
        "CREATE INDEX ppdata_town_city ON pp_data (town_city);",
        "CREATE INDEX ppdata_district ON pp_data (district);",
        "CREATE INDEX ppdata_county ON pp_data (county);",
        "CREATE INDEX ppdata_postcode ON pp_data (postcode);",
        "CREATE INDEX postcodedata_postcode ON postcode_data (postcode);",
        "CREATE INDEX postcodedata_country ON postcode_data (country);",
        "CREATE INDEX postcodedata_latitude ON postcode_data (latitude);",
        "CREATE INDEX postcodedata_longitude ON postcode_data (longitude);"
    ]
    for line in query:
        run_query(line)


def join_one_year(year):
    """
    Join by year, save results into a csv
    """
    # For slightly better readability, join_query was separated:
    join_query = [
        "SELECT pp_data_temp.`price`, pp_data_temp.`date_of_transfer`, pp_data_temp.`postcode`, pp_data_temp.`property_type`, pp_data_temp.`new_build_flag`, pp_data_temp.`tenure_type`, pp_data_temp.`locality`, pp_data_temp.`town_city`, pp_data_temp.`district`, pp_data_temp.`county`, postcode_data_temp.`country`, postcode_data_temp.`latitude`, postcode_data_temp.`longitude`, pp_data_temp.`db_id` FROM",
        f"(SELECT `price`, `date_of_transfer`, `postcode`, `property_type`, `new_build_flag`, `tenure_type`, `locality`, `town_city`, `district`, `county`, `db_id` FROM `pp_data` WHERE date_of_transfer >= '{year}-01-01' AND date_of_transfer <= '{year}-12-31') pp_data_temp",
        "INNER JOIN",
        "(SELECT `country`, `latitude`, `longitude`, `postcode` FROM `postcode_data`) postcode_data_temp",
        "ON pp_data_temp.`postcode` = postcode_data_temp.`postcode`"]
    join_query = " ".join(join_query)
    results = run_query_return_results(join_query)
    fp = open(f'joined-{year}.csv', 'w')
    myFile = csv.writer(fp)
    myFile.writerows(results)
    fp.close()
    # df = pd.DataFrame(results)
    # df.to_csv(f'joined-{year}.csv', index=False)
    print(f'Successfully joined {year}\n')
    return results


def generate_joined_csvs(from_year, to_year):
    for year in range(from_year, to_year + 1):
        join_one_year(year)


def create_prices_coordinates_data():
    """
    Create the schema for prices_coordinates_data.
    """
    schema = ["DROP TABLE IF EXISTS `prices_coordinates_data`;",
              "CREATE TABLE IF NOT EXISTS `prices_coordinates_data`",
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
    run_query(schema)

def load_data(filename, tablename):
    run_query(f"""LOAD DATA LOCAL INFILE '{filename}' INTO TABLE `{tablename}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"' LINES STARTING BY '' TERMINATED BY '\n';""")

def get_pois(north, south, west, east, tags):
  """Returns points of interest based on bounding box and tags"""
  return ox.geometries_from_bbox(north, south, east, west, tags)

def get_bounding_box(latitude, longitude, box_height, box_width):
  north = latitude + box_height/2
  south = latitude - box_height/2
  west = longitude - box_width/2
  east = longitude + box_width/2
  return (north, south, west, east)