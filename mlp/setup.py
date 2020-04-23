import pandas as pd
import pyodbc
import yaml


def load_config():
    """
    Loads config file
    """

    with open("config.yaml", "r") as f:
        config = yaml.load(f, yaml.FullLoader)

    return config 

def extract():
    """
    Extract data from SQL database, returns pandas dataframe
    """

    ## Organize details in dictionary

    config = load_config()  
    connect_dict = config["Connection details"]

    assert len(pyodbc.drivers()) > 0, "Please install an ODBC driver"
    connect_dict['DRIVER'] = pyodbc.drivers()[-1]

    ## Make connection string

    conn_string = ";".join([f"{key}={value}" for key, value in connect_dict.items()])

    ## Connect to database

    connection = pyodbc.connect(conn_string)

    ## Instruction was given to write and SQL query to extract:
    ## - data between dates 2011 and 2012
    ## - all columns EXCEPT guest-bike and registered-bike

    col_names_substring = ",".join(config['Columns to extract'])

    df = pd.read_sql(sql = f"SELECT {col_names_substring} FROM rental_data WHERE date BETWEEN '20110101' and '20121231'",
                    con = connection,
                    parse_dates = "date")\
        .sort_values(["date", "hr"])

    return df