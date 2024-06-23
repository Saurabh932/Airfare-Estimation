import os
import sys
from dataclasses import dataclass

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainingConfig
from src.components.model_evalutation import ModelEvaluation
from src.logger import logging
from src.exception import CustomException

@dataclass
class TrainingPipeline:
    def __init__(self):
        logging.info("Training Pipeline has started.")
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainingConfig()

    def start_data_ingestion(self):
        logging.info("Entered the start_data_ingestion method of TrainingPipeline class")
        try:
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()
            logging.info("Exited the start_data_ingestion method of TrainingPipeline class")
            return train_path, test_path
        except Exception as e:
            logging.error(f"Error in start_data_ingestion: {e}")
            raise CustomException(e, sys)

    def start_data_transformation(self, train_path, test_path):
        logging.info("Entered the start_data_transformation method of TrainingPipeline class")
        try:
            data_transformation = DataTransformation(DataIngestionConfig(train_path, test_path))
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
            logging.info("Exited the start_data_transformation method of TrainingPipeline class")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            logging.error(f"Error in start_data_transformation: {e}")
            raise CustomException(e, sys)

    def start_model_training(self, train_path, test_path):
        logging.info("Entered the start_model_training method of TrainingPipeline class")
        try:
            model_trainer = ModelTrainer(DataIngestionConfig(train_path=train_path, test_path=test_path))
            best_model_name, best_r2_score = model_trainer.initiate_model_training()
            
            logging.info("Exited the start_model_training method of TrainingPipeline class")
            return best_model_name, best_r2_score
        except Exception as e:
            logging.error(f"Error in start_model_training: {e}")
            raise CustomException(e, sys)

    def start_model_evaluation(self, train_path, test_path):
        logging.info("Entered the start_model_evaluation method of TrainingPipeline class")
        try:
            model_evaluation = ModelEvaluation(DataIngestionConfig(train_path, test_path))
            evaluation_results = model_evaluation.initiate_model_evaluation()
            logging.info("Exited the start_model_evaluation method of TrainingPipeline class")
            return evaluation_results
        except Exception as e:
            logging.error(f"Error in start_model_evaluation: {e}")
            raise CustomException(e, sys)

    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainingPipeline class")
        try:
            train_path, test_path = self.start_data_ingestion()
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(train_path, test_path)
            best_model_name, best_r2_score = self.start_model_training(train_arr, test_arr)
            evaluation_results = self.start_model_evaluation(train_path, test_path)
            logging.info("Exited the run_pipeline method of TrainingPipeline class")
            logging.info(f"Best Model: {best_model_name} with R2 Score: {best_r2_score}")
            logging.info(f"Evaluation Results: {evaluation_results}")
        except Exception as e:
            logging.error(f"Exception occurred in run_pipeline: {e}")
            raise CustomException(e, sys)



# if __name__ == "__main__":
#     pipeline = TrainingPipeline()
#     pipeline.run_pipeline()
