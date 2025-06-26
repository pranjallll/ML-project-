print("MODEL TRAINER SCRIPT HAS STARTED")
import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging 

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        print("Inside initiate_model_trainer...")  # Debug print
        try:
            logging.info("Split training and test input data")
            print("Splitting train and test arrays...")  # Debug print
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            print(f"Shapes X_train: {X_train.shape}, y_train: {y_train.shape}")  # Debug print
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNeighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            print("Evaluating models...")  # Debug print
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, param={})
            print("Model report:", model_report)  # Debug print
            
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                print("Best model score below threshold.")  # Debug print
                raise CustomException('No best model found')
            
            logging.info(f"Best found model on both training and testing dataset: {best_model_name} (R2={best_model_score:.4f})")

            print(f"Saving model: {best_model_name}")  # Debug print
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            print(f"R2 score: {r2_square}")
            print(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")
            return r2_square
        except Exception as e:
            print("Exception in initiate_model_trainer:", e)
            raise CustomException(e, sys)

if __name__ == "__main__":
    print("model_trainer.py is running...")  # Debug print
    import pandas as pd
    import dill
    import numpy as np

    try:
        train_df = pd.read_csv("artifacts/train.csv")
        test_df = pd.read_csv("artifacts/test.csv")
        print("Dataframes loaded.")  # Debug print

        # Load preprocessor
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = dill.load(f)
        print("Preprocessor loaded.")  # Debug print

        target_col = "math_score"
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        print("Transforming train data...")  # Debug print
        X_train_processed = preprocessor.transform(X_train)
        print("Transforming test data...")  # Debug print
        X_test_processed = preprocessor.transform(X_test)

        train_array = np.c_[X_train_processed, y_train]
        test_array = np.c_[X_test_processed, y_test]

        print("Calling ModelTrainer...")  # Debug print
        trainer = ModelTrainer()
        r2 = trainer.initiate_model_trainer(train_array, test_array)
        print(f"R2 score: {r2}")
        print(f"Model saved at: {trainer.model_trainer_config.trained_model_file_path}")
    except Exception as e:
        print("Exception during model training:", e)