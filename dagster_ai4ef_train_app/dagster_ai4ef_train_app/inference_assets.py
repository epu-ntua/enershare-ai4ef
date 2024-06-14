import numpy as np
import pandas as pd
import torch
import pickle

from dagster import multi_asset, AssetIn
from dagster_ai4ef_train_app.class_assets import Classifier, Regressor
from dagster_ai4ef_train_app.data_assets import data_scaling

@multi_asset(
    name="service_1_predict",
    group_name="ai4ef_assets",
    ins={'testing_data': AssetIn(key="data_preprocessing"),
         'data_cols': AssetIn(key="data_cols"),
         'scalers': AssetIn(key="scalers"),},
    deps=["optuna_optimize","store_models"])
def service_1_model_predict(test_X, service_1_targets, model_path, scalers_path):
    model = Classifier.load_from_checkpoint(checkpoint_path=model_path)

    with open(scalers_path, 'rb') as f: scalers = pickle.load(f)

    print(test_X)

    test_X = pd.DataFrame.from_dict(test_X)

    test_X = data_scaling(test_X, scalers['X_categorical_scalers'], scalers['X_continuous_scalers'])

    test_X_tensor = torch.tensor(test_X[:10].values, dtype=torch.float32)

    pred_Y = model(test_X_tensor).round().tolist()
    print(pred_Y)
    
    pred_dict = {service_1_targets[i]: int(value) for i, value in enumerate(pred_Y[0])}

    print(pred_dict)
    
    return pred_dict

@multi_asset(
    name="service_1_predict",
    group_name="ai4ef_assets",
    ins={'testing_data': AssetIn(key="data_preprocessing"),
         'data_cols': AssetIn(key="data_cols"),
         'scalers': AssetIn(key="scalers"),},
    deps=["optuna_optimize","store_models"])
def service_2_model_predict(test_X, service_2_targets, model_path, scalers_path):
    model = Regressor.load_from_checkpoint(checkpoint_path=model_path)

    with open(scalers_path, 'rb') as f: scalers = pickle.load(f)

    print(test_X)

    test_X = pd.DataFrame.from_dict(test_X)

    test_X = data_scaling(test_X, scalers['X_categorical_scalers'], scalers['X_continuous_scalers'])

    test_X_tensor = torch.tensor(test_X[:1].values, dtype=torch.float32)

    pred_Y = model(test_X_tensor).detach().numpy() #create and convert output tensor to numpy array
    print(pred_Y)

    unscaled_test_X = pd.DataFrame(); unscaled_pred_Y = pd.DataFrame()
    
    for column, scaler in scalers['X_categorical_scalers'].items():
        unscaled_test_X[column] = scaler.inverse_transform(test_X[[column]].values.ravel())
    for column, scaler in scalers['X_continuous_scalers'].items():
        unscaled_test_X[column] = scaler.inverse_transform(test_X[[column]])

    for column, scaler in scalers['Y_continuous_scalers'].items():
        unscaled_pred_Y = scaler.inverse_transform(pred_Y.reshape(-1,1)).ravel()

    for index, pred  in enumerate(unscaled_pred_Y):
        ECG = unscaled_test_X.iloc[index,1] # Electricity consumption of the grid 
        # print(ECG)
        PECB = ECG * 2.5 # Primary Energy consumption before
        # print(PECB)
        real_pred = float(pred) * 2.5
        # print(real_pred)
        
        PECA = float(pred) + PECB - real_pred 
        # print(PECA)
        pred = np.append(pred,PECA) # Primary energy consumption after
        service_2_targets = np.append(service_2_targets,'Primary energy consumption after')
        
        PECR = PECB - PECA  #Reduction of primary energy consumption
        # print(PECR)
        pred = np.append(pred,PECR)
        service_2_targets = np.append(service_2_targets,'Reduction of primary energy consumption')
        
        # CO2 emmisions = PECR (in Mwh) * coefficient of t C02
        CO2_emmisions = (real_pred / 2.5) * 0.109
        pred = np.append(pred, CO2_emmisions)       
        service_2_targets = np.append(service_2_targets,'CO2 emissions reduction')
    
    pred_dict = {service_2_targets[i]: pred[i] for i in range(len(service_2_targets))}
    print(pred_dict)
    
    return pred_dict