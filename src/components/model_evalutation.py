import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.logger import logging
from src.exception import CustomException
from src.utilis import load_object


class ModelEvaluation:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        logging.info("Model Evaluation has Started.")


    def eval_metrics(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)

        logging.info("Evaluation metrics has been captured.")
        return mae, rmse, r2
    
        # return accuracy, rmse, mse, r2
    

    def initiate_model_evaluation(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_config.train_path)

            target_column = "Price"
            x = train_df.drop(columns=[target_column, "_id"], axis=1)
            y = train_df[target_column]

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            predictions = model.predict(x_test)
            rmse, mae, r2 = self.eval_metrics(predictions, y_test)

            logging.info(f"Evaluation results -, RMSE: {rmse}, MAE: {mae}, R2: {r2}")

            return {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                }

        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":

#     data_ingestion = DataIngestion()
#     train_path, test_path = data_ingestion.initiate_data_ingestion()
    
#     evaluator = ModelEvaluation(DataIngestionConfig(train_path, test_path))
    
#     evaluation_results = evaluator.initiate_model_evaluation()
#     print(evaluation_results)
#     logging.info("Model Evaluation has Ended.")

