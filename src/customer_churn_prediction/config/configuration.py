from customer_churn_prediction.constants import *
import os
from pathlib import Path
from customer_churn_prediction.utils.common import read_yaml, create_directories
from customer_churn_prediction.entity.config_entity import (DataIngestionConfig,
                                                DataPreprocesserConfig,
                                                TrainingConfig)



class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config



    def get_preprocesser_config(self) -> DataPreprocesserConfig:
        config = self.config.preprocessor
        preprocessor_config = DataPreprocesserConfig(
            data_dir=config.data_dir,
            root_dir=config.root_dir,
            preprocessor_dir=config.preprocessor_dir,
        )
        return preprocessor_config


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
        )

        return training_config
    


      