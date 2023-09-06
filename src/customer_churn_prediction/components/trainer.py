import numpy as np
from customer_churn_prediction.entity.config_entity import TrainingConfig
from customer_churn_prediction.utils.common import save_bin
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

class ModelTrainer:
    def __init__(self, config:TrainingConfig):
        self.trainer_config=config
        self.models = {
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression()
        }
        self.params={
            "Random Forest":{
                'n_estimators': [25, 50, 100, 150],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [3, 6, 9],
                'max_leaf_nodes': [3, 6, 9],
            },
            "Gradient Boosting":{
                'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                'criterion':['squared_error', 'friedman_mse'],
                'max_features':['sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{
                "penalty":["l1","l2"]
            }
            
        }
        
    def evaluate_models(self, X_train, y_train,X_test,y_test):
        report = {}

        for i in range(len(list(self.models))):
            model = list(self.models.values())[i]
            para=self.params[list(self.models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            self.models[list(self.models.keys())[i]]=model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(self.models.keys())[i]] = test_model_score

        return report


    def initiate_model_trainer(self,X_train,y_train,X_test,y_test):

        model_report:dict=self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test)
        
        best_model_score = max(sorted(model_report.values()))

        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = self.models[best_model_name]

        save_bin(
            path=self.trainer_config.trained_model_path,
            data=best_model
        )

        predicted=best_model.predict(X_test)

        r2_square = r2_score(y_test, predicted)
        accuracy=accuracy_score(y_test, predicted)
        precision=precision_score(y_test, predicted)
        recall=recall_score(y_test, predicted)
        result={"r2_square": r2_square,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall}
        return result