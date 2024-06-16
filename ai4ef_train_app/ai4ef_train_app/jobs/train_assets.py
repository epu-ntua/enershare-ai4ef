import pandas as pd
from matplotlib import pyplot as plt # type: ignore

import os
import warnings 

import optuna # type: ignore
from optuna.integration import PyTorchLightningPruningCallback # type: ignore
import pytorch_lightning as pl # type: ignore
from pytorch_lightning import Trainer # type: ignore
from pytorch_lightning.callbacks import EarlyStopping # type: ignore

from dagster import multi_asset, AssetOut, AssetIn, MetadataValue, Output# type: ignore
from dagster import AssetOut, graph_multi_asset
from .class_assets import *
from .data_assets import extract_data_cols
from typing import Tuple

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def extract_data_cols(config):
    
    feature_cols = config.feature_cols.split(",") #'The data,Primary energy consumption after ,Reduction of primary energy,CO2 emissions reduction'
        
    if(',' in config.target_cols): # if multiple targets
        target_cols = config.target_cols.split(",")
    else:
        target_cols = [config.target_cols]

    # find categorical columns (string-based) 
    # categorical_cols = [col for col in data.columns if data[col].isin([0, 1]).all() or data[col].apply(lambda x: isinstance(x, str)).all()]
    
    data_cols = (feature_cols, target_cols)

    return data_cols

def keep_best_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_trainer", value=trial.user_attrs["best_trainer"])

def objective(trial, config, feature_cols, train_data, validation_data):
    """
    Function used by optuna for hyperparameter tuning
    Each execution of this function is basically a new trial with its params
    changed as suggest by suggest_* commands
    Parameters: 
        trial object (default)
        feature_cols: list containing model's input features
        train/validation_data: pd.Dataframe containing transformed data 
    Returns: 
        validation loss of model used for checking progress of tuning 
    """
    
    n_layers = list(map(int, config.n_layers.split(',')))
    l_rate = list(map(float, config.l_rate.split(','))) 
    layer_sizes = list(map(int, config.layer_sizes.split(',')))

    n_layers = trial.suggest_int("n_layers", n_layers[0], n_layers[1])
    params = {
        'input_dim': len(feature_cols), # df also has target column
        'max_epochs': config.max_epochs,
        'seed': 42,
        'layer_sizes': [trial.suggest_categorical("n_units_l{}".format(i), layer_sizes) for i in range(n_layers)], 
        'l_rate':  trial.suggest_float('l_rate', l_rate[0], l_rate[1], log=True), # loguniform will become deprecated
        'activation': trial.suggest_categorical("activation", config.activation) if ',' in config.activation else config.activation, #SiLU (Swish) performs good
        'optimizer_name': trial.suggest_categorical("optimizer_name", config.optimizer_name) if ',' in config.optimizer_name else config.optimizer_name,
        'batch_size': int(trial.suggest_categorical('batch_size', config.batch_size.split(','))),
        'num_workers': int(config.num_workers)
    }

    print(params)
    
    # ~~~~~~~~~~~~~~ Setting up network ~~~~~~~~~~~~~~~~~~~~~~
    torch.set_num_threads(params['num_workers']) 
    pl.seed_everything(params['seed'], workers=True)  

    model = globals()[config.mlClass](**params) # double asterisk (dictionary unpacking)

    trainer = Trainer(max_epochs=int(params['max_epochs']), deterministic=True,
                      accelerator='auto', 
                    #   devices = 1 if torch.cuda.is_available() else 0,
                    # auto_select_gpus=True if torch.cuda.is_available() else False,
                    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"),
                               EarlyStopping(monitor="val_loss", mode="min", verbose=True)]) 

    trainer.logger.log_hyperparams(params)

    train_X,train_Y = train_data 
    validation_X,validation_Y = validation_data
    train_loader = model.train_dataloader(train_X,train_Y)
    val_loader = model.val_dataloader(validation_X,validation_Y)

    print("############################ Traim/Test/Validate ###############################")
    
    trainer.fit(model, train_loader, val_loader)

    # store each trial trainer and update it at objetive's callback function to keep best
    trial.set_user_attr(key="best_trainer", value=trainer)

    return trainer.callback_metrics["val_loss"].item()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Ops ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@multi_asset(description="Op that performs HPO/training",
    group_name='train_pipeline',
    required_resource_keys={"config"},
    outs={"study": AssetOut(dagster_type=optuna.study.Study)})
def optuna_optimize(context, training_data):
    """
    Function used to setup optuna for study
    Parameters: None
    Returns: study object containing info about trials
    """
    config = context.resources.config

    train_data, validation_data = training_data

    # The pruners module defines a BasePruner class characterized by an abstract prune() method, which, 
    # for a given trial and its associated study, returns a boolean value 
    # representing whether the trial should be pruned (aborted)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    feature_cols , _  = extract_data_cols(config)
    print(feature_cols)

    pruner: optuna.pruners.BasePruner = optuna.pruners.MedianPruner() # experimentally better performance
        
    # default sampler: TPMESampler    
    study = optuna.create_study(direction="minimize", pruner=pruner)
    """
    timeout (Union[None, float]) – Stop study after the given number of second(s). 
    None represents no limit in terms of elapsed time. 
    The study continues to create trials until: the number of trials reaches n_trials, 
                                                timeout period elapses, stop() is called or, 
                                                a termination signal such as SIGTERM or Ctrl+C is received.
    """
    study.optimize(lambda trial: objective(trial, config, feature_cols, train_data, validation_data),
                #  n_jobs=2,
                #    timeout=600, # 10 minutes
                   callbacks=[keep_best_model_callback],
                   n_trials=config.n_trials,
                   gc_after_trial=True)

    # image_metadata = optuna_visualize(study)
    # best_model =  study.user_attrs["best_trainer"]

    return Output(study, metadata={"trials_df": MetadataValue.md(study.trials_dataframe().to_markdown())})
    # return Output(best_model, metadata={
    #                         "trials_df": MetadataValue.md(study.trials_dataframe().to_markdown()),
    #                         "image_metadata": image_metadata})

@multi_asset(description="Op that stores best model",
    group_name='train_pipeline',
    required_resource_keys={"config"},
    outs={"best_model": AssetOut(dagster_type=pl.Trainer)})
def store_models(context, study: optuna.study.Study):
    """
    Function used to store models extracted from optuna HPO to filesystem
    Parameters: optuna study object containing info about trials
    Returns: best optuna model
    """

    config = context.resources.config

    print(f'Save best model at file: \"{config.ml_path}\"')
    best_model = study.user_attrs["best_trainer"]

    print(type(best_model))
    print(config.ml_path)
    best_model.save_checkpoint(config.ml_path)
    print(f'Save data scalers at files: \"categorical_scalers\" and \"train_scalers\"')
    
    return best_model

@graph_multi_asset(
        group_name='train_pipeline',
        ins={"training_data": AssetIn(key="training_data", dagster_type=Tuple[ Tuple[pd.DataFrame, pd.DataFrame] , Tuple[pd.DataFrame, pd.DataFrame]])},
        outs={"best_model": AssetOut(dagster_type=pl.Trainer),
              "study": AssetOut(dagster_type=optuna.study.Study)})
def train_pipeline(training_data):
    study = optuna_optimize(training_data)
    best_model = store_models(study)
    return {"study": study, "best_model": best_model}


# @job(
#     description="Job performing HPO and training of model"
# )
# def tune_train_and_store_best_model():
#     train_pipeline()