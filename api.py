from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
import uvicorn
# import httpx
from enum import Enum
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware
import psutil
import nvsmi # takes a long time to install
# from fastapi import APIRouter

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

from classifier import service_1_model_predict
from MLPRegressor_HPO import service_2_model_predict
from MLPRegressor_HPO import Regression

service_1_targets = ['Carrying out construction works','Reconstruction of engineering systems',
                     'Heat installation','Water heating system']
service_2_targets = ['Electricity produced by solar panels']

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

@app.post("/service_1/inference", tags=["Service 1"])
async def get_building_parameters_service_1(parameters: dict):
    prediction = None
        
    # {"building_total_area": 351.6, 
    #     "reference_area": 277.4, 
    #     "above_ground_floors": 3, 
    #     "underground_floor": 0,
    #     "initial_energy_class": "D",
    #     "energy_consumption_before": 106.04,
    #     "energy_class_after": "B"}
    #
    # replace "_" with "-" said dictionary key
    # parameters['above-ground_floors'] = parameters.pop('above_ground_floors')
    # parameters['initial_energy_class '] = parameters.pop('initial_energy_class')
    feature_list = ["building_total_area","reference_area","above_ground_floors","underground_floor",
                    "initial_energy_class","energy_consumption_before","energy_class_after"]
    parameters = {key: parameters[key] for key in feature_list}

    # replace "_" with white spaces for all dictionary keys
    parameters = {key.replace("_", " "): value for key, value in parameters.items()}
    # capitalize all keys except first (see next command)
    parameters = {key.capitalize(): value for key, value in parameters.items()}
    # capitalize each letter after "_" in first key of dict (Reference Total Area) because the dataset does not have consistent capitalization
    # parameters = {'Building Total Area' if key == 'Building total area' else key: value for key, value in parameters.items()}
    parameters = [parameters]

    best_model = 'best_classifier.pkl'
    # prediction = {'prediction':'Service 1'}
    prediction = service_1_model_predict(best_model, service_1_targets, parameters)
    # print(f'pred: {prediction}')
    # convert key to lowercase
    # prediction = {key.lower(): value for key, value in prediction.items()}
    # # replace white spaces with "_" for all dictionary keys    
    # prediction = {key.replace(" ","_"): value for key, value in prediction.items()}

    with open('./json_files/EF_comp_outputs.json', 'r') as json_file:
        json_template = json.load(json_file)
        # print(json_template)
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

        # Create a new JSON structure with the updated fields
        # properties_json = {"properties": properties}

        # Print the resulting JSON
        # print(properties)
        return properties

@app.post('/service_2/inference', tags=['Service 2'])
async def get_building_parameters_service_2(parameters: dict):
    prediction = None

    # {"region": "Rīga", 
    #         "electricity_consumption_of_the_grid": 4.65, 
    #         "primary_energy_consumption_before": 11.63, 
    #         "current_inverter_set_power": 0.0, 
    #         "inverter_power_in_project": 10}

    # { "region": "Rīga", 
    #             "average_monthly_electricity_consumption_before": 4.65, 
    #             "current_inverter_set_power": 0.0, 
    #             "planned_inverter_set_power": 10}

    # rename "planned_inverter_set_power" to "inverter_power_in_project" as in the csv file
    parameters = replace_key_name(parameters, "planned_inverter_set_power","inverter_power_in_project")
    # parameters['inverter_power_in_project'] = parameters.pop('planned_inverter_set_power')

    # rename "current_electricity_consumption_before" to "electricity_consumption_of_the_grid" as in the csv file
    parameters = replace_key_name(parameters, "average_monthly_electricity_consumption_before", "electricity_consumption_of_the_grid")
    # parameters['electricity_consumption_of_the_grid'] = parameters.pop('average_monthly_electricity_consumption_before')

    # Input is given in kwh, but our data are in mwh (division by 1000)
    # to create a yearly average (12 months), we multiply by 12 
    parameters["electricity_consumption_of_the_grid"] *= 0.012 # 12 / 1000 

    # replace "_" with white spaces for all dictionary keys
    parameters = {key.replace("_", " "): value for key, value in parameters.items()}
    # parameters['primary_energy_consumption_before '] = parameters.pop('primary_energy_consumption_before')#

    # remove suffix space between 
    # Convert all keys to lowercase
    parameters = {key.capitalize(): value for key, value in parameters.items()}

    parameters = [parameters]
    
    filename='./models-scalers/best_regressor.ckpt'
    categorical_cols='Region'
    prediction = service_2_model_predict(parameters, service_2_targets, categorical_cols, filename)

    # Round all values to two digits
    prediction = {key: round(value, 2) for key, value in prediction.items()}
    # convert key to lowercase
    # prediction = {key.lower(): value for key, value in prediction.items()}
    # # replace white spaces with "_" for all dictionary keys    
    # prediction = {key.replace(" ","_"): value for key, value in prediction.items()}

    with open('./json_files/sol_pan_outputs.json', 'r') as json_file:
        json_template = json.load(json_file)
        # print(json_template)
        properties = []

        # Update the JSON template with values from the prediction dictionary
        for property_dict in json_template["properties"]:
            title = property_dict["title"]
            if title in prediction:
                updated_property = {
                    "title": title,
                    "description": property_dict["description"],
                    "id": property_dict["id"],
                    "unit": property_dict['unit'],
                    "value": str(prediction[title])
                }
                properties.append(updated_property)

    #     # Create a new JSON structure with the updated fields
    #     properties_json = {"properties": properties}

        # Print the resulting JSON
        # print(properties)
        return properties

    # return prediction # as json that george wants.

@app.get('/system_monitoring/get_cpu_usage', tags=['System Monitoring'])
async def get_cpu_usage():
    cpu_count_logical = psutil.cpu_count()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_usage = psutil.cpu_percent(percpu=True)
    cpu_percentage_response = {'labels': [f'CPU {i}' for i in range(1, len(cpu_usage)+1)], 'data': cpu_usage}
    response = {'barchart_1': cpu_percentage_response,
                'text_1': cpu_count,
                'text_2': cpu_count_logical}
    return response


@app.get('/system_monitoring/get_memory_usage', tags=['System Monitoring'])
async def get_memory_usage():
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    swap_memory_response = {
        'title': 'Swap memory usage (Mbytes)',
        'low': swap_memory.used // 1024**2,
        'high': swap_memory.total // 1024**2}
    virtual_memory_response = {
        'title': 'Virtual memory usage (Mbytes)',
        'low': virtual_memory.used // 1024**2,
        'high': virtual_memory.total // 1024**2}
    response = {
        'progressbar_1': virtual_memory_response,
        'progressbar_2': swap_memory_response}
    return response


@app.get('/system_monitoring/get_gpu_usage', tags=['System Monitoring'])
async def get_gpu_usage():
    gpus_stats = nvsmi.get_gpus()
    response = {}
    for gpu_stats in gpus_stats:
        response[gpu_stats.id] = {
           "progressbar_1": {'title': "GPU utilization (%)", 'percent': gpu_stats.gpu_util}, 
           "progressbar_2": {'title': "GPU memory utilization (Mbytes)",
                            'low':  gpu_stats.mem_used,
                            'high':  gpu_stats.mem_total}}
    print(response)
    return response

@app.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8888/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, reload=True)
