from dataclasses import dataclass
import numpy as np 
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from customer_churn_prediction.utils.common import save_bin
from customer_churn_prediction.entity.config_entity import DataPreprocesserConfig
from sklearn.model_selection import train_test_split


class DataTransformation:
    def __init__(self,config: DataPreprocesserConfig):
        self.data_transformation_config=config

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        numerical_columns = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "EstimatedSalary"
        ]
        categorical_columns = [
            "Geography",
            "Gender",
            "HasCrCard",
            "IsActiveMember"
        ]

        num_pipeline= Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler())
            ]
        )

        cat_pipeline=Pipeline(
            steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder",OneHotEncoder()),
            ("scaler",StandardScaler(with_mean=False))
            ]
        )

        preprocessor=ColumnTransformer(
            [
            ("num_pipeline",num_pipeline,numerical_columns),
            ("cat_pipelines",cat_pipeline,categorical_columns)
            ]
        )

        return preprocessor            
        
        
    def initiate_data_transformation(self):

        df = pd.read_csv(self.data_transformation_config.data_dir)
        df.drop(columns = ['RowNumber','CustomerId','Surname'], inplace= True)
        X=df.drop("Exited", axis=1)
        y=df["Exited"]
        train_x, test_x, train_y, test_y = train_test_split(X,y,test_size=0.2)
        df.dropna(axis=1, inplace=True)
        preprocessing_obj=self.get_data_transformer_object()
        sm = SMOTE(random_state = 2)
        train_x=preprocessing_obj.fit_transform(train_x)
        test_x=preprocessing_obj.transform(test_x)
        train_x, train_y = sm.fit_resample(train_x, train_y)
        test_x, test_y = sm.fit_resample(test_x, test_y)
        save_bin(path=self.data_transformation_config.preprocessor_dir,
                data=preprocessing_obj
        )

        return train_x, test_x, train_y, test_y