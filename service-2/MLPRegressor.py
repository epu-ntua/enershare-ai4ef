import numpy as np
import pandas as pd
from enum import IntEnum
from datetime import datetime
from matplotlib import pyplot as plt

import os
import gc
import tempfile

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from scipy.stats import zscore

from darts.models import NaiveSeasonal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
import logging
import click

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.models.signature import infer_signature

# get environment variables
from dotenv import load_dotenv
load_dotenv()
# explicitly set MLFLOW_TRACKING_URI as it cannot be set through load_dotenv
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")

#### Define the model and hyperparameters
class Regression(pl.LightningModule):
    """
    Regression  Techniques are used when the output is real-valued based on continuous variables. 
                For example, any time series data. This technique involves fitting a line
    Feature: Features are individual independent variables that act as the input in your system. 
             Prediction models use features to make predictions. 
             New features can also be obtained from old features using a method known as ‘feature engineering’. 
             More simply, you can consider one column of your data set to be one feature. 
             Sometimes these are also called attributes. T
             The number of features are called dimensions
    Target: The target is whatever the output of the input variables. 
            In our case, it is the output value range of load. 
            If the training set is considered then the target is the training output values that will be considered.
    Labels: Label: Labels are the final output. You can also consider the output classes to be the labels. 
            When data scientists speak of labeled data, they mean groups of samples that have been tagged to one or more labels.

    ### The Model ### 
    Initialize the layers
    Here we have:
        one input layer (size 'lookback_window'), 
        one output layer (size 36 as we are predicting next 36 hours)
        hidden layers define by 'params' argument of init
    """
    def __init__(self, **params):
        super(Regression, self).__init__()

        # enable Lightning to store all the provided arguments 
        # under the self.hparams attribute. 
        # These hyperparameters will also be stored within the model checkpoint
        self.save_hyperparameters()
        
        # used by trainer logger (check log_graph flag)
        # example of input use by model (random tensor of same size)
        self.example_input_array = torch.rand(self.hparams.input_dim)

        # self.loss = MeanAbsolutePercentageError() #MAPE
        self.loss = nn.MSELoss()

        """
        feature_extractor: all layers before classifier
        classifier: last layer connecting output with rest of network (not always directly)
        We load proper pretrained model, and use its feauture_extractor for the new untrained one
        (Also check forward pass commentary)
        """
        self.feature_extractor = None        
        self.classifier = None

        feature_layers, last_dim = self.make_hidden_layers()
        self.feature_extractor = nn.Sequential(*feature_layers) #list of nn layers
        self.classifier = nn.Linear(last_dim, 1)

    def make_hidden_layers(self):
        """
        Each loop is the setup of a new layer
        At each iteration:
            1. add previous layer to the next (with parameters gotten from layer_sizes)
                    at first iteration previous layer is input layer
            2. add activation function
            3. set current_layer as next layer
        connect last layer with cur_layer
        Parameters: None
        Returns: 
            layers: list containing input layer through last hidden one
            cur_layer: size (dimension) of the last hidden layer      
        """
        layers = [] # list of layer to add at NN
        cur_layer = self.hparams.input_dim

        # layer_sizes must be iterable for creating model layers
        if(not isinstance(self.hparams.layer_sizes,list)):
            self.hparams.layer_sizes = [self.hparams.layer_sizes] 

        for next_layer in self.hparams.layer_sizes: 
            layers.append(nn.Linear(int(cur_layer), int(next_layer)))
            layers.append(getattr(nn, self.hparams.activation)()) # nn.activation_function (as suggested by Optuna)
            cur_layer = int(next_layer) #connect cur_layer with previous layer (at first iter, input layer)
            # print(f'({int(cur_layer)},{int(next_layer)},{self.hparams.layer_sizes})')

        return layers, cur_layer

    # Perform the forward pass
    def forward(self, x):
        """
        In forward pass, we pass input through (freezed or not) feauture extractor
        and then its output through the classifier 
        """
        representations = self.feature_extractor(x)
        return self.classifier(representations)

### The Data Loaders ###     
    # Define functions for data loading: train / validate / test

# If you load your samples in the Dataset on CPU and would like to push it during training to the GPU, 
# you can speed up the host to device transfer by enabling "pin_memory".
# This lets your DataLoader allocate the samples in page-locked memory, which speeds-up the transfer.    
    def train_dataloader(self,train_X,train_Y):
        feature = torch.tensor(train_X.values).float() #feature tensor train_X
        target = torch.tensor(train_Y.values).float().unsqueeze(1) #target tensor train_Y 
        train_dataset = TensorDataset(feature, target)  # dataset bassed on feature/target
        train_loader = DataLoader(dataset = train_dataset, 
                                  shuffle = True, 
                                  pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                  num_workers = self.hparams.num_workers,
                                  batch_size = self.hparams.batch_size)
        return train_loader
            
    def test_dataloader(self,test_X,test_Y):
        feature = torch.tensor(test_X.values).float()
        target = torch.tensor(test_Y.values).float().unsqueeze(1) # convert [x] -> [x,1] to match feature tensor
        test_dataset = TensorDataset(feature, target)
        test_loader = DataLoader(dataset = test_dataset, 
                                 pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                 num_workers = self.hparams.num_workers,
                                 batch_size = self.hparams.batch_size)
        return test_loader

    def val_dataloader(self,validation_X,validation_Y):
        feature = torch.tensor(validation_X.values).float()
        target = torch.tensor(validation_Y.values).float().unsqueeze(1)
        val_dataset = TensorDataset(feature, target)
        validation_loader = DataLoader(dataset = val_dataset,
                                       pin_memory=True if torch.cuda.is_available() else False, #for GPU
                                       num_workers = self.hparams.num_workers,
                                       batch_size = self.hparams.batch_size)
        return validation_loader

    def predict_dataloader(self):
        return self.test_dataloader()
    
### The Optimizer ### 
    # Define optimizer function: here we are using ADAM
    def configure_optimizers(self):
        return getattr(optim, self.hparams.optimizer_name)( self.parameters(),
                                                            # momentum=0.9, 
                                                            # weight_decay=1e-4,                   
                                                            lr=float(self.hparams.l_rate))

### Training ### 
    # Define training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        # Add logging
        logs = {'loss': loss}
        self.log("loss", loss, on_epoch=True) # computes train_loss mean at end of epoch        
        return {'loss': loss, 'log': logs}

### Validation ###  
    # Define validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss)
        self.log("avg_val_loss", loss, on_epoch=True)  # computes avg_loss mean at end of epoch
        return {'val_loss': loss}

### Testing ###     
    # Define test step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        correct = torch.sum(logits == y.data)
        # I want to visualize my predictions vs my actuals so here I'm going to 
        # add these lines to extract the data for plotting later on
        self.log('test_loss', loss, on_epoch=True)        
        return {'test_loss': loss, 'test_correct': correct, 'logits': logits}

### Prediction ###
    # Define prediction step
        # This method takes as input a single batch of data and makes predictions on it. 
        # It then returns predictions
    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.forward(x)
        
    def on_train_epoch_end(self):
        gc.collect()

    def on_validation_epoch_end(self):
        gc.collect()

from sklearn.preprocessing import LabelEncoder

def data_preprocess(dframe,needed_cols,onehot_fields,target_col):
        
    # dframe.drop([col for col in uneeded_cols if col != target_col], axis=1, inplace=True)

    dframe = dframe[needed_cols + [target_col]]

    # remove str prefix of "Project Number" and make each entry an int instead of object
    # dframe.set_index('Project number')
    # dframe['Project number'] = dframe['Project number'].str.replace('PME2-', '')
    # dframe['Project number'] = dframe['Project number'].astype(str).astype(np.int64)

    # Categorical variables: label encoding
    for field in onehot_fields:
        dframe[field] = LabelEncoder().fit_transform(dframe[field])
    
    # Categorical variables: one-hot encoding
    # for field in onehot_fields:
    #     dummies = pd.get_dummies(dframe[field], prefix=field, drop_first=False)
    #     dframe = pd.concat([dframe, dummies], axis=1)
    # dframe = dframe.drop(onehot_fields, axis=1)

    # Continuous variables: scaling
    continuous_fields = dframe.iloc[:, ~dframe.columns.isin(onehot_fields)]
    
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for field in continuous_fields:
        mean, std = dframe[field].mean(), dframe[field].std()
        scaled_features[field] = [mean, std]
        dframe.loc[:, field] = (dframe[field] - mean)/std
    
    # Remove NaNs / duplicates / outliers
    dframe = dframe.dropna().reset_index(drop=True)
    dframe.drop_duplicates(inplace=True)
    dframe = dframe[(np.abs(zscore(dframe)) <= 3).all(axis=1)]

    return dframe, scaled_features

def train_test_valid_split(dframe,statify_cols='Region',target_col='Electricity produced by solar panels'):
    """
    we choose to split data with validation/test data to be at the end of time series
    Parameters:
        pandas.dataframe containing dataframe to split
    Returns:
        pandas.dataframe containing train/test/valiation data
        pandas.dataframe containing valiation data
        pandas.dataframe containing test data
    """

    y = dframe.pop(target_col)

    # keep columns containing region data to use for stratifying
    strat_df = dframe.copy()
    for col in statify_cols:
        strat_df = strat_df.filter(regex=f'^{col}$',axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(dframe, y, test_size=0.2, random_state=1, 
                                                        shuffle=True, stratify=strat_df)
    
    # keep columns containing region data to use for stratifying
    strat_df = train_X.copy()
    for col in statify_cols:
        strat_df = strat_df.filter(regex=f'^{col}$',axis=1)

    # strat_df = train_X.filter(regex=f'^{statify_col}_',axis=1)
    train_X, validation_X, train_Y, validation_Y = train_test_split(train_X, train_Y, test_size=0.25, random_state=1, 
                                                      shuffle=True, stratify=strat_df) # 0.25 x 0.8 = 0.2
        
    return train_X, validation_X, test_X, train_Y, validation_Y, test_Y

def cross_plot_actual_pred(plot_pred, plot_actual):
    # And finally we can see that our network has done a decent job of estimating!
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(plot_pred, label='Prediction')
    ax.plot(plot_actual, label='Data')
    # ax.set_xticks(np.arange(len(datesx))[12::24])
    ax.legend()
    plt.show()

def calculate_metrics(actual,pred,target_col,scaled_features):

    mean = scaled_features[target_col][0]
    std = scaled_features[target_col][1]

    # Get predicted/actual points (scaled back to their original size)
    plot_pred = [x * std + mean for x in pred]
    plot_actual = [x * std + mean for x in actual]

    # Evaluate the model prediction
    metrics = {
        "MAE": mean_absolute_error(plot_actual,plot_pred),
        "MSE": mean_squared_error(plot_actual,plot_pred),
        "RMSE": np.sqrt(mean_squared_error(plot_actual,plot_pred))
    }
    
    print("  Metrics: ")
    for key, value in metrics.items():
        print("    {}: {}".format(key, value))

    cross_plot_actual_pred(plot_pred, plot_actual)
    
    return metrics

def model_predict(test_X,scaled_features,target_col,filename='model.ckpt'):
    model = Regression.load_from_checkpoint(checkpoint_path=filename)

    test = [4.67,8.8,0,11.04,4]
    test_X = test
    # mean = scaled_features[target_col][0]
    # std = scaled_features[target_col][1]
    
    # unscaled_test_X = [x * std + mean for x in test_X.values]
    # unscaled_test_X = [list(map(float, sublist)) for sublist in unscaled_test_X]

    # unscaled_test_X = [float(x) for x in unscaled_test_X]
    # test_X_tensor = torch.tensor(unscaled_test_X[:10])
    # pred_Y = model(test_X_tensor).detach().numpy() #create and convert output tensor to numpy array

    test_X_tensor = torch.tensor(test_X)
    pred_Y = model(test_X_tensor).detach().numpy() #create and convert output tensor to numpy array

    print(test_X)
    for index, pred  in enumerate(pred_Y):
        ECG = test_X[index] # Electricity consumption of the grid 
        PECB = ECG * 2.5 # Primary Energy consumption before
        real_pred = float(pred) * 2.5
        PECA = float(pred) + PECB - real_pred 

        pred = np.append(pred,PECA) # Primary energy consumption after (KW)

        PECR = PECB - PECA  #Reduction of primary energy consumption
        pred = np.append(pred,PECR)
        
        # CO2 emmisions = PECR (in Mwh) * coefficient of t C02
        CO2_emmisions = (real_pred / 2.5) * 0.109
        pred = np.append(pred, CO2_emmisions)
        
        print(pred)
    

# Remove whitespace from your arguments
@click.command(
    help= "Given a folder path for CSV files (see load_raw_data), use it to create a model, find\
            find ideal hyperparameters and train said model to reduce its loss function"
)

@click.option("--filepath", type=str, default='Sol_pan_comp.csv', help="File containing csv files used by the model")
@click.option("--seed", type=int, default=42, help='seed used to set random state to the model')
@click.option("--max_epochs", type=int, default=20, help='range of number of epochs used by the model')
# @click.option("--n_layers", type=str, default="1", help='range of number of layers used by the model')
@click.option("--layer_sizes", type=int, default=100, help='range of size of each layer used by the model')
@click.option("--l_rate", type=float, default=1e-4, help='range of learning rate used by the model')
@click.option("--activation", type=str, default="ReLU", help='activations function experimented by the model')
@click.option("--optimizer_name", type=str, default="Adam", help='optimizers experimented by the model') # SGD
@click.option("--batch_size", type=int, default=40, help='possible batch sizes used by the model') #16,32,
@click.option("--num_workers", type=int, default=2, help='accelerator (cpu/gpu) processesors and threads used') 
@click.option('--preprocess', type=int, default=1, help='data preprocessing and scaling')
@click.option('--needed_cols', type=str, default='Region,Electricity consumption of the grid,Primary energy consumption before ,Current inverter set power,Inverter power in project', help='Dataset columns not necesary for training')
# @click.option('--neededd_cols', type=str, default='The data,Primary energy consumption after ,Reduction of primary energy,CO2 emissions reduction,Project number,Granted support', help='Dataset columns not necesary for training')
@click.option('--target_col', type=str, default='Electricity produced by solar panels', help='Target column that we want to predict (model output)')
@click.option('--categorical_cols', type=str, default='Region', help='columns containing categorical data')
@click.option('--predict', type=int, default=0, help='predict value or not')
@click.option('--filename', type=str, default='model.ckpt', help='filename of best model')

def forecasting_model(**kwargs):
    """
    This is the main function of the script. 
    1. It loads the data from csvs
    2. splits to train/test/valid
    3. trains model based on hyper params set
    4. computes MAPE and plots graph of best model
    Parameters
        kwargs: dictionary containing click paramters used by the script
    Returns: None 
    """
    # with mlflow.start_run(run_name="train",nested=True) as train_start:

        # Auto log all MLflow entities
        # mlflow.pytorch.autolog()

    if not os.path.exists("./temp_files/"): os.makedirs("./temp_files/")
    # store mlflow metrics/artifacts on temp file
    with tempfile.TemporaryDirectory(dir='./temp_files/') as tmpdir: 

        # ~~~~~~~~~~~~ Data Collection & process ~~~~~~~~~~~~~~~~~~~~
        print("############################ Reading Data ###############################")
        df = pd.read_csv(kwargs['filepath'])
        df_backup = df.copy()
        scaler = None
        print("############################ Data Preprocess ###############################")
        
        # Remove date column (not needed)
        # neededd_cols = ['The data','Primary energy consumption after ',
                        # 'Reduction of primary energy','CO2 emissions reduction']
        needed_cols = kwargs['needed_cols'].split(",") #'The data,Primary energy consumption after ,Reduction of primary energy,CO2 emissions reduction'
        target_col = kwargs['target_col']
        categorical_cols = kwargs['categorical_cols'].split(",") 
        # target_col = 'Electricity produced by solar panels'
        # categorical_cols = 'Region'    
        print(df.info())

        if(kwargs['preprocess']):
            df, scaler = data_preprocess(df, needed_cols, categorical_cols, target_col)

        print("############################ Train Test Spit ###############################")
        
        train_X, validation_X, test_X, train_Y, validation_Y, test_Y = train_test_valid_split(df,categorical_cols,target_col)

        print(df.info())

        if(kwargs['predict']):
            model_predict(test_X,scaler,target_col,kwargs['filename'])
            return
        
        print("############################ Setting up network ###############################")
        
        
        # ~~~~~~~~~~~~~~ Setting up network ~~~~~~~~~~~~~~~~~~~~~~
        torch.set_num_threads(kwargs['num_workers']) #################################
        pl.seed_everything(kwargs['seed'], workers=True)  

        model_args = kwargs.copy(); model_args['input_dim'] = len(df.columns)

        model = Regression(**model_args) # double asterisk (dictionary unpacking)

        trainer = Trainer(max_epochs=kwargs['max_epochs'], deterministic=True,
                        profiler=SimpleProfiler(dirpath=f'{tmpdir}', filename='profiler_report'), #add simple profiler
                        logger= CSVLogger(save_dir=tmpdir),
                        #   accelerator='auto', 
                        #   devices = 1 if torch.cuda.is_available() else 0,
                        auto_select_gpus=True if torch.cuda.is_available() else False,
                        check_val_every_n_epoch=2,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True, patience=10)]) 

        train_loader = model.train_dataloader(train_X,train_Y)
        test_loader = model.test_dataloader(test_X,test_Y)
        val_loader = model.val_dataloader(validation_X,validation_Y)

        print("############################ Traim/Test/Validate ###############################")
        
        #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Traim/Test/Validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        trainer.fit(model, train_loader, val_loader)

        trainer.save_checkpoint("model.ckpt")
        
        pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
        
        # Either best or path to the checkpoint you wish to test. 
        # If None and the model instance was passed, use the current weights. 
        # Otherwise, the best model from the previous trainer.fit call will be loaded.
        trainer.test(ckpt_path='best', dataloaders=test_loader)

        trainer.validate(ckpt_path='best', dataloaders=val_loader)

        preds = trainer.predict(ckpt_path='best', dataloaders=test_loader)

        print(preds)
        #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Extract data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        actuals = []
        for x,y in test_loader: 
            actuals.append(y)

        # torch.vstack: convert list of tensors to (rank 2) tensor
        # .tolist(): convert (rank 2) tensor to list of lists
        # final outcome: list of floats
        preds = [item for sublist in torch.vstack(preds).tolist() for item in sublist]
        actuals = [item for sublist in torch.vstack(actuals).tolist() for item in sublist]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        metrics = calculate_metrics(actuals,preds,target_col,scaler)

        #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Store to Mlflow ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #store trained model to mlflow with input singature
        # print("\nUploading training csvs and metrics to MLflow server...")
        # logging.info("\nUploading training csvs and metrics to MLflow server...")
        # signature = infer_signature(train_X.head(1), pd.DataFrame(preds))
        # mlflow.pytorch.log_model(model, "model", signature=signature)
        # mlflow.log_params(kwargs)
        # mlflow.log_artifacts(train_tmpdir, "train_results")
        # mlflow.log_metrics(metrics)
        # # mlflow.set_tag("run_id", train_start.info.run_id)        

if __name__ == '__main__':
    print("\n=========== Forecasing Model =============")
    logging.info("\n=========== Forecasing Model =============")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)    
    forecasting_model()
