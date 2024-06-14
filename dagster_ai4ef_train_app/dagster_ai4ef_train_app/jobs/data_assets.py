import numpy as np # type: ignore
import pandas as pd # type: ignore
import json
import pickle
from scipy.stats import zscore # type: ignore # type: ignore
import os
from typing import Tuple, List, Dict

from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from dagster import MetadataValue, Output # type: ignore
from dagster import multi_asset, AssetOut, AssetIn, graph_multi_asset # type: ignore
import requests
import json
from dotenv import load_dotenv
import logging
from urllib.parse import urlparse

# Construct the path to the .env file located two parent directories up
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(base_dir, '.env')
# Load the .env file
load_dotenv(dotenv_path=env_path)
storage_path = os.environ.get("SHARED_STORAGE_PATH")
api_key = os.environ.get("API_KEY")
consumer_agent_id = os.environ.get("CONSUMER_AGENT_ID")
provider_agent_id = os.environ.get("PROVIDER_AGENT_ID")

from dagster_ai4ef_train_app.resources.my_io_manager import CustomFilesystemIOManager

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def is_filepath(s):
    return os.path.isabs(s) or os.path.isfile(s) or os.path.isdir(s)

def is_url(s):
    try:
        result = urlparse(s)
        # Check if the scheme is HTTP, HTTPS, or FTP
        return all([result.scheme, result.netloc])
    except:
        return False

import json 

def request_data(path):
    # Headers (if any)
    headers = {
        'Authorization': 'Bearer' + api_key,
        'Forward-Id': provider_agent_id,         # reciever connector ID
        'Forward-Sender': consumer_agent_id      # Sender connector ID
    }

    print(path)
    # Sending GET request
    response = requests.get(path, headers=headers)

    # Check if request was successful (status code 200)
    if response.status_code == 200:
        try:
            data = response.json()  # Attempt to decode JSON
            # print(json.dumps(data, indent=4))
            return data
        except ValueError:  # includes simplejson.decoder.JSONDecodeError
            print("Response content is not valid JSON")
            print(response.text)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print("Response text:", response.text)

def extract_data_cols(data_cols):
    feature_cols = data_cols.feature_cols.split(",") #'The data,Primary energy consumption after ,Reduction of primary energy,CO2 emissions reduction'
        
    if(',' in data_cols.target_cols): # if multiple targets
        target_cols = data_cols.target_cols.split(",")
    else:
        target_cols = [data_cols.target_cols]

    # find categorical columns (string-based) 
    # categorical_cols = [col for col in data.columns if data[col].isin([0, 1]).all() or data[col].apply(lambda x: isinstance(x, str)).all()]
    data_cols = (feature_cols, target_cols)

    return data_cols

def data_scaling(dframe, categorical_scalers=None, train_scalers=None, train=False, target=False):
    """
    Used to transform and scale/encode dframes 
    Parameters:
        data: pandas.dataframe containing dataframe to scale
        categorical_scalers/train_scalers: dict containing scalers to use on categorical/training data
        train: boolean to signify if need for scaler creation or simple transformation
        target: if target dataframe, then its a single-column df, a series. Need to convert it back to df
    Returns:
        pandas.dataframe containing train/validation data
        pandas.dataframe containing test data
        1x2 sklearn scalers vector for X and Y respectively 
    """
    # if target dataframe, then its a single-column df, a series. Need to convert it back to df
    if(target): dframe = pd.DataFrame(dframe)

    categorical_cols = [col for col in dframe.columns if dframe[col].isin([0, 1]).all() or dframe[col].apply(lambda x: isinstance(x, str)).all()]

    # Categorical variables: label encoding
    # Initialize a dictionary to store the scalers    
    # if not training set, then use existing scalers
    if(train):
        categorical_scalers = {column: LabelEncoder() for column in categorical_cols}
        # Scale each column and store in the dictionary
        for column, scaler in categorical_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.fit_transform(dframe[column])
    else:
        for column, scaler in categorical_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.transform(dframe[column])

    # Continuous variables: scaling
    continuous_fields = [col for col in dframe.columns if col not in categorical_cols]

    # Initialize a dictionary to store the scalers      
    # if not training set, then use existing scalers
    if(train):
        train_scalers = {column: MinMaxScaler() for column in continuous_fields}
        # Scale each column and store in the dictionary
        for column, scaler in train_scalers.items():            
            if(column in dframe.columns):
                dframe[column] = scaler.fit_transform(dframe[[column]])
    else:
        for column, scaler in train_scalers.items():
            if(column in dframe.columns):
                dframe[column] = scaler.transform(dframe[[column]])

    if(train):
        return dframe, train_scalers, categorical_scalers
    else:
        return dframe

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Operations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@multi_asset(
    name="get_data",
    description="Op used to extract data from our defined source",
    group_name='data_pipeline',
    required_resource_keys={"config"},
    outs={"initial_data": AssetOut(dagster_type=pd.DataFrame, io_manager_key="io_manager")})
def get_data(context):

    initial_data = None
    config = context.resources.config

    print(CustomFilesystemIOManager(base_dir='test')._get_path(context))

    if(is_url(config.input_filepath)):
        initial_data = pd.DataFrame(request_data(config.input_filepath))
    elif(is_filepath(config.input_filepath)):
        initial_data = pd.read_csv(config.input_filepath) #,index_col=0
    else:
        initial_data = pd.DataFrame()

    return Output(value=initial_data, 
                  metadata={
                    "n_rows": len(initial_data),
                    "preview": MetadataValue.md(initial_data.head().to_markdown())})

@multi_asset(description="Multi asset that preprocess data for training and testing",
    name="data_preprocessing",
    required_resource_keys={"config"},
    group_name='data_pipeline',
    outs={"training_data": AssetOut(dagster_type=Tuple[ Tuple[pd.DataFrame, pd.DataFrame] , Tuple[pd.DataFrame, pd.DataFrame]], io_manager_key="io_manager"),
          "testing_data": AssetOut(dagster_type=Tuple[pd.DataFrame, pd.DataFrame], io_manager_key="io_manager"),
          "scalers": AssetOut(dagster_type=Dict, io_manager_key="io_manager")})
def data_preprocessing(context, initial_data: pd.DataFrame):
    """
    we choose to split data with validati,on/test data to be at the end of time series
    Parameters:
        data: pandas.dataframe containing dataframe to split
    Returns:
        pandas.dataframe containing train/validation data
        pandas.dataframe containing test data
        1x2 sklearn scalers vector for X and Y respectively 
    """
    config = context.resources.config

    # data, data_cols = data_arguments
    _, target_cols = extract_data_cols(config)

    scalers = {}

    categorical_cols = [col for col in initial_data.columns if initial_data[col].isin([0, 1]).all() or initial_data[col].apply(lambda x: isinstance(x, str)).all()]
    continuous_fields = [col for col in initial_data.columns if col not in categorical_cols]

    # Remove NaNs / duplicates / outliers
    initial_data = initial_data.dropna().reset_index(drop=True)
    initial_data.drop_duplicates(inplace=True)
    initial_data = initial_data[(np.abs(zscore(initial_data[continuous_fields])) <= 3).all(axis=1)]

    strat_df = initial_data[categorical_cols].copy()
    y = initial_data[target_cols].copy()

    strat_df = initial_data[categorical_cols].copy()

    initial_data.drop(target_cols,axis=1,inplace=True)

    train_X, test_X, train_Y, test_Y = train_test_split(initial_data, y, test_size=None, random_state=42, 
                                                        shuffle=True) #stratify=strat_df

    # train right now is both train and validation set
    train_X, scalers['X_continuous_scalers'], scalers['X_categorical_scalers'] = data_scaling(train_X, train=True)
    train_Y, scalers['Y_continuous_scalers'], scalers['Y_categorical_scalers'] = data_scaling(train_Y, train=True, target=True)

    test_X = data_scaling(test_X, scalers['X_categorical_scalers'], scalers['X_continuous_scalers'])
    test_Y = data_scaling(test_Y, scalers['Y_categorical_scalers'], scalers['Y_continuous_scalers'], target=True)

    train_stratify_cols = [item for item in train_X.columns if item in categorical_cols]
    strat_df = train_X[train_stratify_cols].copy()
    
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=42, 
                                                      shuffle=True) # , stratify=train_Y
    
    training_data = (train_X, train_Y), (validation_X, validation_Y)
    testing_data = (test_X, test_Y) 

    # Save the scalers
    with open(config.scalers_path, 'wb') as f: pickle.dump(scalers, f)

    # Print the deserialized data
    print("Training Data:", training_data)
    print("Testing Data:", testing_data)
    print("Scalers:", scalers)    # Extract training and testing data
    
    return  Output(value=training_data, 
                    metadata={
                        "n_rows": [len(training_data[0]), len(training_data[1])],
                        "train_X_preview": MetadataValue.md(training_data[0][0].head().to_markdown()),
                        "train_Y_preview": MetadataValue.md(training_data[0][1].head().to_markdown())}) ,\
            Output(value=testing_data, 
                    metadata={
                        "n_rows": [len(testing_data[0]), len(testing_data[1])],
                        "test_X_preview": MetadataValue.md(testing_data[0].head().to_markdown()),
                        "test_Y_preview": MetadataValue.md(testing_data[1].head().to_markdown())}) ,\
            scalers

@graph_multi_asset(
        name="data_pipeline",
        group_name='data_pipeline',
        outs={"training_data": AssetOut(dagster_type=Tuple[ Tuple[pd.DataFrame, pd.DataFrame] , Tuple[pd.DataFrame, pd.DataFrame]]),
          "testing_data": AssetOut(dagster_type=Tuple[pd.DataFrame, pd.DataFrame]),
          "scalers": AssetOut(dagster_type=Dict)})
def data_pipeline():
    initial_data = get_data()
    # data_cols = extract_data_cols() 
    training_data, testing_data, scalers = data_preprocessing(initial_data) #, scalers
    return {'training_data': training_data, 
            'testing_data': testing_data,
            "scalers": scalers}

# @job(
#     description="Job fetching and preprocessing data"
# )
# def data_fetch_and_preprocess_job():
#     data_pipeline()