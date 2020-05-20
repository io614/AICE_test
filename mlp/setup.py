import pandas as pd
import pyodbc
import yaml
from sklearn.model_selection import train_test_split

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
    first_date = config['Date limits']['first']
    last_date = config['Date limits']['last']

    ## should be "query string"
    connection_string = f"SELECT {col_names_substring} FROM rental_data WHERE date BETWEEN '{first_date}' and '{last_date}' ORDER BY date, hr"

    print("Extracting data...")
    df = pd.read_sql(sql = connection_string,
                     con = connection)

    return df


def split(df):
    """
    splits dataframe into ratios defined in config.yaml.
    the random state is also defined in config.yaml
    """
    split_config_dict = load_config()['split_config']
    splits = split_config_dict['splits']
    train_df, test_df = train_test_split(df, train_size=splits['train'],
                                             random_state=split_config_dict['split_random_state'])

    return train_df, test_df
