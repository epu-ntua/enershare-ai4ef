from fastapi import FastAPI
import uvicorn
# import httpx
from enum import Enum
import pandas as pd

# from utils import ConfigParser
from fastapi.middleware.cors import CORSMiddleware
import psutil, nvsmi
from dotenv import load_dotenv
# from fastapi import APIRouter
load_dotenv()
# explicitly set environment variables if any
# example:
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

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

service_1_targets = ['Carrying out construction works ','Reconstruction of engineering systems',
                     'Heat installation','Water heating system']
service_2_targets = ['Electricity produced by solar panels']

@app.post("/service_1/inference", tags=["Service 1"])
async def get_building_parameters_service_1(parameters: dict):
    # here you assume that you have a dictionary with the building parameters submitted by the user.
    # parameters = [{"name":..., "value":...}, {"name":..., "value":...}, ...]
    print(parameters)
    prediction = None
    # parameters = {"name1": value1, "name2": value2, ...}
    # input = [parameters]
    input = {"Building Total Area": 351.6, 
        "Reference area": 277.4, 
        "Above-ground floors": 3, 
        "Underground floor": 0,
        "Initial energy class ": "D",
        "Energy consumption before": 106.04,
        "Energy class after": "B"}

    if(not parameters):
        parameters = [input]
    else:
        parameters = [parameters]


    best_model = 'best_classifier.pkl'
    # prediction = {'prediction':'Service 1'}
    prediction = service_1_model_predict(best_model, service_1_targets, parameters)

    return prediction # as json that george wants.

@app.post('/service_2/inference', tags=['Service 2'])
async def get_building_parameters_service_2(parameters: dict):

    # here you assume that you have a dictionary with the building parameters submitted by the user.
    # parameters = [{"name":..., "value":...}, {"name":..., "value":...}, ...]
    prediction = None
    print(parameters)

    input = {"Region": "Rīga", 
        "Electricity consumption of the grid": 4.65, 
        "Primary energy consumption before ": 11.63, 
        "Current inverter set power": 0.0, 
        "Inverter power in project": 10}

    if(not parameters):
        parameters = [input]
    else:
        parameters = [parameters]
    
    filename='./models-scalers/best_regressor.ckpt'
    categorical_cols='Region'
    # prediction = {'prediction':'Service 2'}
    prediction = service_2_model_predict(parameters, service_2_targets, categorical_cols, filename)

    return prediction # as json that george wants.

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
    return {"message": "Congratulations! Your API is working as expected. Now head over to http://localhost:8080/docs"}

if __name__ == "__main__":
    uvicorn.run(app, host="enershare.epu.ntua.gr", port=8080)