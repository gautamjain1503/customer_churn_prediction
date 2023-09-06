import pandas as pd
from customer_churn_prediction.components.pre_processer import DataTransformation
from customer_churn_prediction.entity.config_entity import DataPreprocesserConfig
from customer_churn_prediction.utils.common import load_bin
from pathlib import Path

class PredictPipeline:
    def __init__(self,config: DataPreprocesserConfig, model_path: Path):
        self.data_transformation_config=config
        self.model_path=model_path
        self.preprocessor_obj=load_bin(self.data_transformation_config.preprocessor_dir)

    def predict_churn(self,df):
        df=self.preprocessor_obj.transform(df)
        model=load_bin(self.model_path)
        result=model.predict(df)
        return result[0]




class CustomData:
    def __init__(  self,
                 creditscore: int,
                 age: int,
                 tenure: int,
                 balance: float,
                 numberofproducts: int,
                 estimatedsalary: float,
                 geography: str,
                 gender: str,
                 hascrcard: str,
                 isactivemember: str):
        
        self.creditscore=creditscore
        self.age=age
        self.tenure=tenure
        self.balance=balance
        self.numberofproducts=numberofproducts
        self.estimatedsalary=estimatedsalary
        self.geography=geography
        self.gender=gender
        self.hascrcard=hascrcard
        self.isactivemember=isactivemember


    def get_data_as_data_frame(self):
        custom_data_input_dict = {
            "CreditScore":[self.creditscore],
            "Age":[self.age],
            "Tenure":[self.tenure],
            "Balance":[self.balance],
            "NumOfProducts":[self.numberofproducts],
            "EstimatedSalary":[self.estimatedsalary],
            "Geography":[self.geography],
            "Gender":[self.gender],
            "HasCrCard":[self.hascrcard],
            "IsActiveMember":[self.isactivemember]
        }

        return pd.DataFrame(custom_data_input_dict)
