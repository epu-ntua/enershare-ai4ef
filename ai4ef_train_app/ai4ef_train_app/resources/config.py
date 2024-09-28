from dagster import ConfigurableResource
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

# Construct the path to navigate three levels up (outside of dagster)
current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
shared_storage_dir = Path(os.environ.get("SHARED_STORAGE_PATH"))
parent_dir = os.path.join(current_dir, shared_storage_dir)

class TrainConfig(ConfigurableResource):
    input_filepath: str = "http://enershare.epu.ntua.gr/consumer-data-app/openapi/0.5/efcomp" # "../datasets/EF_comp.csv" 
    # api_url: str = "http://localhost:7654"
    authorization: str = os.environ.get('API_KEY')
    provider_agent_id: str = os.environ.get("PROVIDER_AGENT_ID") # Forward-Id
    consumer_agent_id: str = os.environ.get("CONSUMER_AGENT_ID") # Forward-Sender
    seed: int = 42
    max_epochs: int = 10
    n_layers: str = "2,6"
    layer_sizes: str = "128,256,512,1024,2048"
    l_rate: str = "0.0001,0.001"
    activation: str = "ReLU"
    optimizer_name: str = "Adam"
    batch_size: str = "256,512,1024"
    num_workers: int = 2
    n_trials: int = 3
    preprocess: int = 1
    feature_cols: str = "Building total area,Reference area,Above ground floors,Underground floor,Initial energy class,Energy consumption before,Energy class after"
    target_cols: str = "Carrying out construction works,Reconstruction of engineering systems,Heat installation,Water heating system"
    predict: int = 0
    ml_path: str = f"{parent_dir}/models-scalers/best_MLPClassifier.ckpt"
    scalers_path: str = f"{parent_dir}/models-scalers/MLPClassifier_scalers.pkl"
    optuna_viz: str = f"{parent_dir}/optuna_viz/classifier/"
    mlClass: str = "Classifier"

    def extract_data_cols(self):
        
        feature_cols = self.feature_cols.split(",") #'The data,Primary energy consumption after ,Reduction of primary energy,CO2 emissions reduction'
            
        if(',' in self.target_cols): # if multiple targets
            target_cols = self.target_cols.split(",")
        else:
            target_cols = [self.target_cols]

        data_cols = (feature_cols, target_cols)

        return data_cols

    def to_dict(self):
        return {
            "input_filepath": self.input_filepath,
            "api_url": self.api_url,
            "seed": self.seed,
            "max_epochs": self.max_epochs,
            "n_layers": self.n_layers,
            "layer_sizes": self.layer_sizes,
            "l_rate": self.l_rate,
            "activation": self.activation,
            "optimizer_name": self.optimizer_name,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "n_trials": self.n_trials,
            "preprocess": self.preprocess,
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
            "predict": self.predict,
            "ml_path": self.ml_path,
            "scalers_path": self.scalers_path,
            "optuna_viz": self.optuna_viz,
            "mlClass": self.mlClass,
        }