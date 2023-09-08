from customer_churn_prediction.config.configuration import ConfigurationManager
from customer_churn_prediction.components.pre_processer import DataTransformation
from customer_churn_prediction.components.trainer import ModelTrainer
from customer_churn_prediction import logger
from customer_churn_prediction.utils.common import save_json
from pathlib import Path

STAGE_NAME = "Data Preprocessing and Training"

class Trainer:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessor_config = config.get_preprocesser_config()
        data_trans = DataTransformation(config=preprocessor_config)
        x_train, x_test, y_train, y_test= data_trans.initiate_data_transformation()
        training_config = config.get_training_config()
        trainer = ModelTrainer(config=training_config)
        score=trainer.initiate_model_trainer(X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test)
        self.save_score(score=score)
        logger.info(f">>>>>> best model score {score} completed <<<<<<\n\nx==========x")
    
    def save_score(self, score):
        save_json(path=Path("scores.json"), data=score)




if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = Trainer()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
