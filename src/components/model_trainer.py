import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from statsmodels.regression.quantile_regression import QuantReg

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.logger import logging
from src.exception import CustomException
from src.utilis import save_object, load_object


@dataclass
class ModelTrainingConfig:
    trained_model_path = os.path.join("artifacts", 'model.pkl')


class ModelTrainer:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self):
        logging.info("Model Training Started.")

        try:
            logging.info(f"Loading data from: {self.data_ingestion_config.train_path}")
            train_df = pd.read_csv(self.data_ingestion_config.train_path)
            
            # Dropping unnecessary columns
            train_df = train_df.drop(columns=["_id"], axis=1)

            target_column = "Price"
            x = train_df.drop(columns=[target_column], axis=1)
            y = train_df[target_column]

            # Load preprocessor
            preprocessor_path = DataTransformationConfig().preprocessor_path
            preprocessor = load_object(preprocessor_path)

            # Apply preprocessor before splitting
            x = preprocessor.transform(x)

            # Split the data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

            logging.info(f"Training data shape: {x_train.shape}, {y_train.shape}")
            logging.info(f"Testing data shape: {x_test.shape}, {y_test.shape}")

            # Base Model for Stacking
            base_models = [
                ('lgbm', LGBMRegressor()),
                ('rf', RandomForestRegressor()),
                ('lasso', LassoCV())
            ]

            # Stacking Regressor
            stack_regressor = StackingRegressor(estimators=base_models,
                                                final_estimator=LinearRegression())

            # Hyperparameter tuning for stacking regressor
            stacking_params = {'final_estimator__fit_intercept': [True, False]}
            stacking_grid = GridSearchCV(estimator=stack_regressor,
                                         param_grid=stacking_params,
                                         scoring='r2',
                                         cv=5)
            stacking_grid.fit(x_train, y_train)

            best_stacking_model = stacking_grid.best_estimator_
            stacking_prediction = best_stacking_model.predict(x_test)
            stacking_r2_sq = r2_score(y_test, stacking_prediction)

            # CatBoost Regressor 
            catboost_param = {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            }

            catboost = CatBoostRegressor(verbose=False)
            catboost_grid = GridSearchCV(estimator=catboost,
                                         param_grid=catboost_param,
                                         scoring='r2',
                                         cv=5)
            catboost_grid.fit(x_train, y_train)

            best_catboost_model = catboost_grid.best_estimator_
            catboost_prediction = best_catboost_model.predict(x_test)
            catboost_r2_sq = r2_score(y_test, catboost_prediction)

            # Linear Regression
            linear = LinearRegression()
            linear.fit(x_train, y_train)

            linear_prediction = linear.predict(x_test)
            linear_r2_sq = r2_score(y_test, linear_prediction)

            # Quantile Regression
            quantile_regressor = QuantReg(y_train, x_train)
            quantile_result = quantile_regressor.fit(q=0.5, max_iter=1000)

            quantile_prediction = quantile_result.predict(x_test)
            quantile_r2_sq = r2_score(y_test, quantile_prediction)

            # Finalizing the best model
            models = {
                "Stacking Regressor": (best_stacking_model, stacking_r2_sq),
                "CatBoost Regressor": (best_catboost_model, catboost_r2_sq),
                "Linear Regressor": (linear, linear_r2_sq),
                "Quantile Regressor": (quantile_result, quantile_r2_sq)
            }

            best_model_name = max(models, key=lambda k: models[k][1])
            best_model, best_r2_score = models[best_model_name]

            # Saving the model
            save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)
            logging.info(f"Best model {best_model_name} with R2 score: {best_r2_score} saved.")

            return best_model_name, best_r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    model_trainer = ModelTrainer(DataIngestionConfig(train_path, test_path))
    best_model_name, best_r2_score = model_trainer.initiate_model_training()

    print("Best Model:", best_model_name)
    print("Best R2 Score:", best_r2_score)

    logging.info("Model Evaluation has Started.")
