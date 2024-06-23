import os
import sys
import numpy as np
import pandas as pd
from src.utilis import save_object
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.data_transformation_config = DataTransformationConfig()

    def performing_data_transformation(self):
        logging.info("Data Transformation has started.")

        try:
            numerical_columns = [
                'Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
                'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
                'Duration_mins'
            ]
            
            categorical_columns = [
                'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo', 'Airline_Jet Airways',
                'Airline_Jet Airways Business', 'Airline_Multiple carriers', 'Airline_Multiple carriers Premium economy',
                'Airline_SpiceJet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
                'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
                'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi'
            ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ('one-hot-encoding', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_config.train_path)
            test_df = pd.read_csv(self.data_ingestion_config.test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtained preprocessing object")

            preprocessor_obj = self.performing_data_transformation()

            target_column = 'Price'

            input_feature_train = train_df.drop(columns=[target_column, "_id"], axis=1)
            target_feature_train = train_df[target_column]

            input_feature_test = test_df.drop(columns=["_id"], axis=1)

            logging.info(f"Applying preprocessing object on train_dataframe and test_dataframe.")

            input_feature_train = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test = preprocessor_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train, np.array(target_feature_train)]

            logging.info('Saved preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                input_feature_test,
                self.data_transformation_config.preprocessor_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation(DataIngestionConfig(train_path, test_path))
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation()
    
    logging.info(f"Data saved in preprocessor path: {preprocessor_path}.")
    logging.info("Data Transformation has Ended.")
