from customer_churn_prediction.config.configuration import ConfigurationManager
from customer_churn_prediction.components.predict import PredictPipeline, CustomData
from customer_churn_prediction import logger
from pathlib import Path

STAGE_NAME = "Prediction Pipeline"

class Predict:
    def __init__(self):
        pass

    def main(self, dictionary):
        config = ConfigurationManager()
        preprocessor_config = config.get_preprocesser_config()
        model_path=Path("artifacts/training/model.pkl")
        data=CustomData(creditscore=dictionary["creditscore"],
                        age=dictionary["age"],
                        tenure=dictionary["tenure"],
                        balance=dictionary["balance"],
                        numberofproducts=dictionary["numberofproducts"],
                        estimatedsalary=dictionary["estimatedsalary"],
                        geography=dictionary["geography"],
                        gender=dictionary["gender"],
                        hascrcard=dictionary["hascrcard"],
                        isactivemember=dictionary["isactivemember"])
        df=data.get_data_as_data_frame()
        model = PredictPipeline(config=preprocessor_config, model_path=model_path)
        result=model.predict_churn(df=df)
        logger.info(f">>>>>>  {result}  <<<<<<\n\nx==========x")
        return result




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        dictionary={}
        dictionary["creditscore"]=619
        dictionary["age"]=42
        dictionary["tenure"]=2
        dictionary["balance"]=0
        dictionary["numberofproducts"]=1
        dictionary["estimatedsalary"]=101348.88
        dictionary["geography"]="France"
        dictionary["gender"]="Female"
        dictionary["hascrcard"]=1
        dictionary["isactivemember"]=1
        print(dictionary)

        obj = Predict()
        obj.main(dictionary=dictionary)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
