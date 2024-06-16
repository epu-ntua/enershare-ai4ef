from fastapi import FastAPI
import httpx
import uvicorn
import numpy as np
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv

# from classifier import service_1_model_predict
from .MLPClassifier_HPO import service_1_model_predict
from .MLPClassifier_HPO import Classifier
from .MLPRegressor_HPO import service_2_model_predict
from .MLPRegressor_HPO import Regression
import os

load_dotenv()

tags_metadata = [
    {"name": "Service 1", "description": "REST APIs for service 1"},
    {"name": "Service 2", "description": "REST APIs for service 2"},
    {"name": "System Monitoring", "description": "REST APIs for monitoring the host machine of the API"},
]

app = FastAPI(
    title="Enershare AI4EF API",
    description="Collection of REST APIs for Serving Execution of Enershare AI4EF Service",
    version="0.0.1",
    openapi_tags=tags_metadata,
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service_1_features = ['Building total area', 'Reference area', 'Above ground floors', 'Underground floor', 
 'Initial energy class', 'Energy consumption before', 'Energy class after']
service_1_targets = ['Carrying out construction works','Reconstruction of engineering systems',
                     'Heat installation','Water heating system']
service_2_targets = ['Electricity produced by solar panels']
service_2_features = ['Region', 'Electricity consumption of the grid', 
                    'Current inverter set power', 'Inverter power in project']

# Add the parent directory to the system path
current_dir = os.getcwd()
shared_storage_dir = Path(os.environ.get("SHARED_STORAGE_PATH"))
parent_dir = os.path.join(current_dir, shared_storage_dir)

# Create path to models_scalers and json_files directory
models_scalers_dir = os.path.join(parent_dir, 'models-scalers')
json_files_dir = os.path.join(parent_dir, 'json_files')

# Create paths to the models and scalers
service_1_ml_path = os.path.join(models_scalers_dir, 'best_MLPClassifier.ckpt')
service_2_ml_path = os.path.join(models_scalers_dir, 'best_MLPRegressor.ckpt')
service_1_scalers_path = os.path.join(models_scalers_dir, 'MLPClassifier_scalers.pkl')
service_2_scalers_path = os.path.join(models_scalers_dir, 'MLPRegressor_scalers.pkl')
service1_outputs_path = os.path.join(json_files_dir, 'service1_outputs.json')
service2_outputs_path = os.path.join(json_files_dir, 'service2_outputs.json')

# Function to convert 1 or 0 to boolean
def convert_to_boolean(value):
    if value == 1:
        return True
    elif value == 0:
        return False
    else:
        return value

def replace_key_name(original_dict, key_to_replace, new_key):
    new_dict = {new_key if key == key_to_replace else key: value for key, value in original_dict.items()}
    
    return new_dict

# Function to check if the value is a float
def is_float(value):
    return isinstance(value, float)

# Function to check if the value is an empty string
def is_empty_string(value):
    return isinstance(value, str) and value == ''

def convert_to_title(variable_name):
    # Split the variable name into words
    words = variable_name.split('_')
    
    # Capitalize the first letter of the first word
    title_case = words[0].capitalize()
    
    # Capitalize the first letter of each subsequent word and join them back
    for word in words[1:]:
        title_case += ' ' + word
    
    return title_case

def calculate_self_consumption(parameters):
    average_energy_generated = parameters['electricity_produced_by_solar_panels'] 
    average_monthly_consumption = parameters['electricity_consumption_of_the_grid']

    try:
        # Check for division by zero
        if average_monthly_consumption == 0:
            return {"estimated_annual_self_consumption": ""}

        # Perform the division
        result = average_energy_generated / average_monthly_consumption
        # Check if the result is greater than 100%
        if result > 1:
            value = 100.0  # Return 100%
        else:
            # Round the result to 2 decimal places
            value = round(result * 100, 2)
            
        return {"estimated_annual_self_consumption": value}
    except Exception:
        # Return an empty string in case of an error
        return {"estimated_annual_self_consumption": ""}

def calculate_financial_savings(parameters):
    average_electricity_price = parameters['average_electricity_price']
    electricity_produced_by_solar_panels = parameters['electricity_produced_by_solar_panels']
    installation_costs = parameters['renewable_installation_cost']
    financial_savings = {}

    # Calculate the annual financial savings
    financial_savings['annual_financial_savings'] = average_electricity_price * (electricity_produced_by_solar_panels * 1000)
    
    # Calculate the payback period
    financial_savings['payback_period'] = installation_costs / financial_savings['annual_financial_savings']

    return financial_savings

def calculate_savings(forecasts, parameters):
    ECG = parameters['electricity_consumption_of_the_grid'] # Electricity consumption of the grid 
    
    savings_dict = forecasts.copy()

    EPSP = savings_dict['Electricity produced by solar panels'] # Electricity produced by solar panels

    PECB = ECG * 2.5 # Primary Energy consumption before
    real_EPSP = float(EPSP) * 2.5
    
    # make sure that PECA is not negative
    PECA = max(0, float(EPSP) + PECB - real_EPSP) # Primary energy consumption after 
    savings_dict['Primary energy consumption after'] = PECA
    
    PECR = PECB - PECA  #Reduction of primary energy consumption
    savings_dict['Reduction of primary energy consumption'] = PECR
    
    # CO2 emmisions = PECR (in Mwh) * coefficient of t C02
    CO2_emmisions = (real_EPSP / 2.5) * 0.109
    savings_dict['CO2 emissions reduction'] = CO2_emmisions

    return savings_dict

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Endpoints ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@app.post("/service_1/inference", tags=["Service 1"])
async def get_building_parameters_service_1(parameters: dict = {"building_total_area": 351.6, 
                                                                "reference_area": 277.4, 
                                                                "above_ground_floors": 3, 
                                                                "underground_floor": 0,
                                                                "initial_energy_class": "D",
                                                                "energy_consumption_before": 106.04,
                                                                "energy_class_after": "B"}):
    prediction = None

    # convert all parameter keys from variable format to title format
    parameters = {convert_to_title(key): value for key, value in parameters.items()}
    # keep only features that are in the model
    parameters = {key: parameters[key] for key in service_1_features}

    parameters = [parameters]

    prediction = service_1_model_predict(parameters, service_1_targets, service_1_ml_path, service_1_scalers_path)

    with open(service1_outputs_path, 'r') as json_file:
        json_template = json.load(json_file)
        properties = []

        # Update the JSON template with values from the prediction dictionary
        for property_dict in json_template["properties"]:
            title = property_dict["title"]
            if title in prediction:
                updated_property = {
                    "title": title,
                    "description": property_dict["description"],
                    "id": property_dict["id"],
                    "value": str(convert_to_boolean(prediction[title]))
                }
                properties.append(updated_property)

        return properties

# dict ={ "region": "Rīga", 
#         "average_monthly_electricity_consumption_before": 4.65, 
#         "current_inverter_set_power": 0.0, 
#         "planned_inverter_set_power": 10,
#         "renewable_installation_cost": 3000,
#         "renewable_energy_generated": 3.0,
#         "average_electricity_price": 0.23}

@app.post('/service_2/inference', tags=['Service 2'])
async def get_building_parameters_service_2(parameters: dict ={"average_monthly_electricity_consumption_before": 4.65, 
                                                               "average_electricity_price": 0.23,
                                                               "renewable_installation_cost": 3000,
                                                               "renewable_energy_generated": "",
                                                               "current_inverter_set_power": 0.0, 
                                                               "planned_inverter_set_power": 10,
                                                               "region": "Rīga"}):
    
    # make sure that certain vales are non-negative
    parameters['renewable_installation_cost'] = max(0, parameters['renewable_installation_cost'])

    # rename "planned_inverter_set_power" to "inverter_power_in_project" as in the csv file
    parameters = replace_key_name(parameters, "planned_inverter_set_power","inverter_power_in_project")

    # rename "current_electricity_consumption_before" to "electricity_consumption_of_the_grid" as in the csv file
    parameters = replace_key_name(parameters, "average_monthly_electricity_consumption_before", "electricity_consumption_of_the_grid")

    # rename "renewable_energy_generated" to "electricity_produced_by_solar_panels" as in the csv file
    parameters = replace_key_name(parameters, "renewable_energy_generated", "electricity_produced_by_solar_panels")

    # Input is given in kwh, but our data are in mwh (division by 1000)
    # to create a yearly average (12 months), we multiply by 12 
    parameters["electricity_consumption_of_the_grid"] *= 0.012 # 12 / 1000 

    api_output = None
    forecasts = None
    if is_empty_string(parameters['electricity_produced_by_solar_panels']):
        # convert all parameter keys from variable format to title format
        service_2_input = {convert_to_title(key): value for key, value in parameters.items()}
        # keep only features that are in the model
        service_2_input = {key: service_2_input[key] for key in service_2_features}

        service_2_input = [service_2_input]

        forecasts = service_2_model_predict(service_2_input, service_2_targets, service_2_ml_path, service_2_scalers_path)

        forecasts = {convert_to_title(key): value for key, value in forecasts.items()}
    else:
        forecasts = {"Electricity produced by solar panels": parameters['electricity_produced_by_solar_panels']}
    
    api_output = calculate_savings(forecasts, parameters)

    parameters["electricity_produced_by_solar_panels"] = api_output["Electricity produced by solar panels"]

    # Calculate self consumption
    self_consumtion = calculate_self_consumption(parameters)
    api_output.update(self_consumtion)

    # Calculate financial savings
    financial_savings = calculate_financial_savings(parameters)
    api_output.update(financial_savings)

    # Convert the keys to title case
    api_output = {convert_to_title(key): value for key, value in api_output.items()}
    # print(prediction)

    # Round all values to two digits
    api_output = {key: round(value, 2) for key, value in api_output.items()}

    with open(service2_outputs_path, 'r') as json_file:
        json_template = json.load(json_file)
        properties = []

        # Update the JSON template with values from the prediction dictionary
        for property_dict in json_template["properties"]:
            title = property_dict["title"]
            if title in api_output:
                updated_property = {
                    "title": title,
                    "description": property_dict["description"],
                    "id": property_dict["id"],
                    "unit": property_dict['unit'],
                    "value": str(api_output[title])
                }
                properties.append(updated_property)

        return properties

@app.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://127.0.0.1:8888/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, reload=True)
