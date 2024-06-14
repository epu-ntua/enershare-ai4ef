from dagster import ConfigurableResource
from dotenv import load_dotenv
import os
load_dotenv()
storage_path = os.getenv("SHARED_STORAGE_PATH")

class TrainConfig(ConfigurableResource):
    input_filepath: str = "http://enershare.epu.ntua.gr/consumer-data-app/openapi/0.5/efcomp" # "../datasets/EF_comp.csv" 
    api_url: str = "http://localhost:7654"
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
    ml_path: str = f"{storage_path}/models-scalers/best_MLPClassifier.ckpt"
    scalers_path: str = f"{storage_path}/models-scalers/MLPClassifier_scalers.pkl"
    optuna_viz: str = f"{storage_path}/optuna_viz/classifier/"
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