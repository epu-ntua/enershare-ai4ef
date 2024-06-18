from dagster import Definitions, FilesystemIOManager, define_asset_job

from .resources.config import TrainConfig
from .jobs.data_assets import data_pipeline #, data_fetch_and_preprocess_job 
from .jobs.train_assets import train_pipeline #, tune_train_and_store_best_model
from .jobs.evaluation_assets import evaluation_pipeline #, evaluate_study_and_model
from .resources.my_io_manager import CustomFilesystemIOManager

data_fetch_and_preprocess_job = define_asset_job("data_fetch_and_preprocess_job", selection=[data_pipeline])
tune_train_and_store_best_model = define_asset_job("tune_train_and_store_best_model", selection=[train_pipeline])
evaluate_study_and_model = define_asset_job("evaluate_study_and_model", selection=[evaluation_pipeline])
ml_pipeline = define_asset_job("ml_pipeline", selection=[data_pipeline, train_pipeline, evaluation_pipeline])

defs = Definitions(
    assets=[data_pipeline, train_pipeline, evaluation_pipeline],
    jobs=[data_fetch_and_preprocess_job, tune_train_and_store_best_model, evaluate_study_and_model, ml_pipeline],
    resources={
        "config": TrainConfig(),
        # "io_manager": FilesystemIOManager(base_dir="generated"),
        "io_manager": CustomFilesystemIOManager(base_dir="my_data"),
    },
)