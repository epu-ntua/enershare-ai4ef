from fastapi import FastAPI
import httpx
import uvicorn
import pandas as pd
import json
from fastapi.middleware.cors import CORSMiddleware

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

# from classifier import service_1_model_predict
from MLPClassifier_HPO import service_1_model_predict
from MLPClassifier_HPO import Classifier
from MLPRegressor_HPO import service_2_model_predict
from MLPRegressor_HPO import Regression

service_1_targets = ['Carrying out construction works','Reconstruction of engineering systems',
                     'Heat installation','Water heating system']
service_2_targets = ['Electricity produced by solar panels']

service_1_ml_path: str = "./models-scalers/best_MLPClassifier.ckpt"
service_1_scalers_path: str = "./models-scalers/MLPClassifier_scalers.pkl"

service_2_ml_path: str = "./models-scalers/best_MLPRegressor.ckpt"
service_2_scalers_path: str = "./models-scalers/MLPRegressor_scalers.pkl"

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
async def get_building_parameters_service_1(parameters: dict = {"building_total_area": 351.6, 
                                                                "reference_area": 277.4, 
                                                                "above_ground_floors": 3, 
                                                                "underground_floor": 0,
                                                                "initial_energy_class": "D",
                                                                "energy_consumption_before": 106.04,
                                                                "energy_class_after": "B"}):
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
    feature_list = ["building_total_area","reference_area","above_ground_floors","underground_floor",
                    "initial_energy_class","energy_consumption_before","energy_class_after"]
    parameters = {key: parameters[key] for key in feature_list}

    # replace "_" with white spaces for all dictionary keys
    parameters = {key.replace("_", " "): value for key, value in parameters.items()}
    # capitalize all keys except first (see next command)
    parameters = {key.capitalize(): value for key, value in parameters.items()}
    # capitalize each letter after "_" in first key of dict (Reference Total Area) because the dataset does not have consistent capitalization
    parameters = [parameters]

    prediction = service_1_model_predict(parameters, service_1_targets, service_1_ml_path, service_1_scalers_path)

    with open('./json_files/EF_comp_outputs.json', 'r') as json_file:
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

@app.post('/service_2/inference', tags=['Service 2'])
async def get_building_parameters_service_2(parameters: dict ={ "region": "Rīga", 
                                                                "average_monthly_electricity_consumption_before": 4.65, 
                                                                "current_inverter_set_power": 0.0, 
                                                                "planned_inverter_set_power": 10}):
    prediction = None

    # { "region": "Rīga", 
    #             "average_monthly_electricity_consumption_before": 4.65, 
    #             "current_inverter_set_power": 0.0, 
    #             "planned_inverter_set_power": 10}

    # rename "planned_inverter_set_power" to "inverter_power_in_project" as in the csv file
    parameters = replace_key_name(parameters, "planned_inverter_set_power","inverter_power_in_project")

    # rename "current_electricity_consumption_before" to "electricity_consumption_of_the_grid" as in the csv file
    parameters = replace_key_name(parameters, "average_monthly_electricity_consumption_before", "electricity_consumption_of_the_grid")

    # Input is given in kwh, but our data are in mwh (division by 1000)
    # to create a yearly average (12 months), we multiply by 12 
    parameters["electricity_consumption_of_the_grid"] *= 0.012 # 12 / 1000 

    # replace "_" with white spaces for all dictionary keys
    parameters = {key.replace("_", " "): value for key, value in parameters.items()}

    # remove suffix space between 
    # Convert all keys to lowercase
    parameters = {key.capitalize(): value for key, value in parameters.items()}

    parameters = [parameters]
    
    # filename='./models-scalers/best_regressor.ckpt'
    # categorical_cols='Region'
    prediction = service_2_model_predict(parameters, service_2_targets, service_2_ml_path, service_2_scalers_path)

    # Round all values to two digits
    prediction = {key: round(value, 2) for key, value in prediction.items()}

    with open('./json_files/sol_pan_outputs.json', 'r') as json_file:
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
                    "unit": property_dict['unit'],
                    "value": str(prediction[title])
                }
                properties.append(updated_property)

        return properties

@app.get("/")
async def root():
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8888/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, reload=True)
