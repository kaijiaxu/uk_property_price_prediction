from .config import *
import yaml
from ipywidgets import interact_manual, Text, Password
import pymysql

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
    url = database_details["url"]
    conn = create_database_connection(user=username, password=password, host=url, database="property_prices")
    cur = conn.cursor()
    cur.execute(query)
    conn.commit()
    cur.close()

