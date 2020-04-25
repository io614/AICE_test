import pandas as pd
import mlp
import numpy as np
import matplotlib.pyplot as plt

config = mlp.setup.load_config()
preprocess_dict = config["Preprocessing"]


# helper functions


# drop duplicate values

def drop_duplicates(df):
    """
    drop duplicates in date, hr
    """
    return df.drop_duplicates(['date', 'hr'])


def correct_weather_typos(df):
    """
    correct typos discovered during eda
    """
    pairs_list = [(search_string,value) for value, search_string in preprocess_dict['weather_search_strings'].items()]
    to_replace, value = list(zip(*pairs_list))
    
    def correct_weather(df):
        return df.assign(weather = lambda x: x.weather.str.lower().\
                          replace(to_replace=to_replace, value=value, regex=True))
    
    return df.pipe(correct_weather)

def encode_weather_ordinal(df):
    """
    encode weather values as ordinals
    """
    weather_ord_encoding = preprocess_dict['weather_ord_encoding']
    weather, ordinal = list(zip(*weather_ord_encoding.items()))
    lookup_weather = pd.Series(index=weather,
                               data=ordinal,
                               name="weather_encoded")
    
    return df.merge(lookup_weather, left_on="weather", right_index=True).sort_values(["date", "hr"]).drop(columns="weather")


def remove_negative_scooters(df):
    """
    remove guest_scooter and registered_scooter values below 0
    """
    negative_mask = (df.guest_scooter < 0) | (df.registered_scooter < 0)
    return df.loc[~negative_mask]


def remove_anomalous_temps(df):
    """
    remove entries which have anomalous feels_like_temperature values
    """
    def get_z_score(df, col):
        df_new = df.copy()
        df_new[col+"_z_score"] = (df[col] - df[col].mean())/ df[col].std()
        return df_new

    df = df.pipe(get_z_score, "temperature").\
         pipe(get_z_score, "feels_like_temperature").\
         assign(z_score_diff_abs = lambda x: (x.temperature_z_score - x.feels_like_temperature_z_score).abs())
    
    z_threshold = preprocess_dict["z_threshold"]
    
    df["above_threshold"] = df["z_score_diff_abs"] > z_threshold
    
    return df.query("not above_threshold").drop(columns=["temperature_z_score",
                                                         "feels_like_temperature_z_score",
                                                         "z_score_diff_abs",
                                                         "above_threshold"])

def create_time_features(df):
    """
    create features based on time/date 
    """
    df = df.copy()
    df['datetime'] = pd.to_datetime(df.date + ' ' + df.hr.astype(str) + ":00")
    
    df['is_weekend'] = df.datetime.dt.dayofweek.isin([5,6])
    df['month'] = df.datetime.dt.month
    
    rush_hours = preprocess_dict['rush_hours']
    df['is_rush_hour'] = df.hr.isin(rush_hours)
    
    def encode_cyclic(df, var, period):
        df_new = df.copy()
        df_new[ var + "_y"] = np.sin(2*np.pi*df[var]/period)
        df_new[ var + "_x"] = np.cos(2*np.pi*df[var]/period)

        return df_new
    
    return df.pipe(encode_cyclic, "hr", 24)\
             .pipe(encode_cyclic, "month", 12)\
             .drop(columns=["date", "datetime", "hr", "month"])

def combine_scooter_vars(df):
    """
    combine guest_scooter and registered_scooter into a single feature total_scooter
    """
    return df.assign(total_scooter=lambda x: x.guest_scooter + x.registered_scooter)\
              .drop(columns = ["guest_scooter", "registered_scooter"])

def preprocess(raw_df):
    """
    run all helper functions defined above
    """
    return raw_df.pipe(drop_duplicates)\
                 .pipe(correct_weather_typos)\
                 .pipe(remove_negative_scooters)\
                 .pipe(remove_anomalous_temps)\
                 .pipe(encode_weather_ordinal)\
                 .pipe(create_time_features)\
                 .pipe(combine_scooter_vars)