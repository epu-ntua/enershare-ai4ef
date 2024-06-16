import numpy as np
import pandas as pd
import warnings 
from matplotlib import pyplot as plt # type: ignore
import os
import optuna # type: ignore
import torch
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelHammingDistance

from darts import TimeSeries
from darts.metrics import mape as mape_darts
from darts.metrics import mae as mae_darts
from darts.metrics import rmse as rmse_darts
from darts.metrics import smape as smape_darts
from darts.metrics import mse as mse_darts

from .class_assets import Classifier, Regressor
from .data_assets import extract_data_cols
from dagster import multi_asset, AssetIn, AssetOut, MetadataValue, Output, graph_multi_asset 
from typing import Tuple

import base64
from io import BytesIO
import json

from sklearn.metrics import classification_report, multilabel_confusion_matrix
import seaborn as sns

def calculate_mape(actual, predicted):
    return np.mean(np.abs((np.array(actual) - np.array(predicted)) / np.array(actual))) * 100 if actual else 0

@multi_asset(
    name="calculate_metrics",
    group_name="evaluation_pipeline",
    required_resource_keys={"config"},
    ins={'testing_data': AssetIn(key="testing_data")},
    outs={"metrics": AssetOut()})
def calculate_metrics(context, testing_data):
    """
    Function used to calculate evaluation metrics to score model's performance
    Parameters:
        testing_data: pd.Dataframe
    Returns: dict containing evaluation metrics
    """
    config = context.resources.config

    # _, _, test_data, _ = data_split
    test_X, test_Y = testing_data
    
    best_model = globals()[config.mlClass].load_from_checkpoint(checkpoint_path=os.path.abspath(config.ml_path))
    test_X_tensor = torch.tensor(test_X.values, dtype=torch.float32)
    test_Y_tensor = torch.tensor(test_Y.values, dtype=torch.float32)
    metrics = {}; image_metadata = {}

    if(config.mlClass == "Classifier"):
        pred_Y = best_model(test_X_tensor).round() 

        # Define classification metrics
        accuracy = MultilabelAccuracy(num_labels=4)
        precision = MultilabelPrecision(num_labels=4, average='macro')
        recall = MultilabelRecall(num_labels=4, average='macro')
        f1 = MultilabelF1Score(num_labels=4, average='macro')
        ham = MultilabelHammingDistance(num_labels=4, average='macro')

        metrics = {
            'accuracy': accuracy(pred_Y, test_Y_tensor).item(),
            'precision': precision(pred_Y, test_Y_tensor).item(),
            'recall': recall(pred_Y, test_Y_tensor).item(),
            'f1_score': f1(pred_Y, test_Y_tensor).item(),
            'hamming': ham(pred_Y, test_Y_tensor).item()
        }

        image_metadata['metrics'] = json.dumps(metrics)

        pred_Y = pd.DataFrame(pred_Y.detach().numpy()) #create and convert output tensor to numpy array

        report = classification_report(test_Y, pred_Y, digits=5, output_dict=True)

        _, target_cols = extract_data_cols(config)
        plt.close() # close any mpl figures (important, doesn't work otherwise)

        confusion_matrices = multilabel_confusion_matrix(test_Y, pred_Y)
        for i, cm in enumerate(confusion_matrices):
            print(f"Confusion Matrix for Class \"{target_cols[i]}\":\n {cm}\n")
    
            buffer = BytesIO()

            # ax, labels, title and ticks
            ax= plt.subplot();
            sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap=plt.cm.Greens);  #annot=True to annotate cells, ftm='g' to disable scientific notation
            ax.set_xlabel('Predicted labels'); ax.set_ylabel('True labels');  
            ax.set_xticklabels(['No','Yes']); ax.set_yticklabels(['No','Yes'])
            plt.title(target_cols[i]); plt.show()
            print(f"True Positives: {cm[1,1]}, False Positives: {cm[0,1]}, True Negatives: {cm[0,0]}, False Negatives: {cm[1,0]} \n\n")
            
            plt.savefig(buffer, format='png'); plt.close(); 
            buffer.seek(0)  # Rewind buffer
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            image_metadata[target_cols[i]] = MetadataValue.md(f"![{target_cols[i]}](data:image/png;base64,{image_data})")
            
        image_metadata['report'] = MetadataValue.md(pd.DataFrame(report).transpose().to_markdown()) 

    elif(config.mlClass == "Regressor"):
        pred_Y = best_model(test_X_tensor).detach().numpy() #create and convert output tensor to numpy array

        pred_series = TimeSeries.from_values(pred_Y)
        actual_series = TimeSeries.from_values(test_Y)

        # ~~~~~~~~~~~~ For MAPE to work ~~~~~~~~~~~~
        # Define the extremely small number you want to add
        # small_number = 1e-10

        # # Add this small number to each value in the series
        # pred_series = pred_series + small_number
        # actual_series = actual_series + small_number
        # # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        pred_series = TimeSeries.from_values(pred_Y)
        actual_series = TimeSeries.from_values(test_Y)

        # Evaluate the best_model prediction
        metrics = {
            "SMAPE": smape_darts(actual_series,pred_series),
            "MAE": mae_darts(actual_series,pred_series),
            "MSE": mse_darts(actual_series,pred_series),
            "RMSE": rmse_darts(actual_series,pred_series)
            # ,
            # "MAPE": mape_darts(actual_series,pred_series)
        }
        
        plt.figure(figsize=(10, 6))
        actual_series.plot(label='actual series')
        pred_series.plot(label='pred series')
        plt.title("Comparison Plot")
        buffer = BytesIO()
        plt.savefig(buffer, format="png"); plt.close(); 
        buffer.seek(0)  # Rewind buffer
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_metadata['comparison_plot'] = MetadataValue.md(f"![comparison_plot](data:image/png;base64,{image_data})")
        image_metadata["metrics"] = json.dumps(metrics)
    else:
        print("This should not be printed!")
    
    return Output(metrics, metadata=image_metadata)

def print_optuna_report(study):
    """
    This function prints hyperparameters found by optuna
    """    
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")

    print("\n ~~~~~~~~~~~~~~~~~~~~~~~~~~ Optuna Report ~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print(" Number: {}".format(trial.number))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    return trial.params

@multi_asset(
    name="optuna_visualize",
    group_name="evaluation_pipeline",
    required_resource_keys={"config"},
    ins={'study': AssetIn(key="study")},
    outs={'optuna_report': AssetOut()})
def optuna_visualize(context, study):
    """
    Plot optuna visualization for trial progression
    """

    config = context.resources.config

    optuna_viz = config.optuna_viz
    image_metadata = {}

    # with tempfile.TemporaryDirectory(prefix=optuna_viz) as tmpdirname:
        # print('created temporary directory', tmpdirname)
    # if dir does not exit, make it
    if not os.path.exists(optuna_viz): os.makedirs(optuna_viz)

    plt.close() # close any mpl figures (important, doesn't work otherwise)
    
    buffer = BytesIO()
    
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(buffer, format='png'); plt.close(); 
    buffer.seek(0)  # Rewind buffer
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_metadata["plot_param_importances"] = MetadataValue.md(f"![plot_param_importances](data:image/png;base64,{image_data})")

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(buffer, format='png'); plt.close(); 
    buffer.seek(0)  # Rewind buffer
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_metadata["plot_optimization_history"] = MetadataValue.md(f"![plot_optimization_history](data:image/png;base64,{image_data})")

    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig(buffer, format='png'); plt.close(); 
    buffer.seek(0)  # Rewind buffer
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_metadata["plot_slice"] = MetadataValue.md(f"![plot_slice](data:image/png;base64,{image_data})")

    optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.savefig(buffer, format='png'); plt.close(); 
    buffer.seek(0)  # Rewind buffer
    image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    image_metadata["plot_intermediate_values"] = MetadataValue.md(f"![plot_intermediate_values](data:image/png;base64,{image_data})")
    
    optuna_report = print_optuna_report(study)
    image_metadata["optuna_report"] = optuna_report
    return Output(optuna_report, metadata=image_metadata)

@graph_multi_asset(
        group_name='evaluation_pipeline',
        ins={"testing_data": AssetIn(key="testing_data", dagster_type=Tuple[pd.DataFrame, pd.DataFrame]),
             "study": AssetIn(key="study", dagster_type=optuna.study.Study)},
        outs={'metrics': AssetOut(),
              'optuna_report': AssetOut()})
def evaluation_pipeline(testing_data, study):
    optuna_report = optuna_visualize(study)
    metrics = calculate_metrics(testing_data)
    return {'optuna_report': optuna_report , 'metrics': metrics}