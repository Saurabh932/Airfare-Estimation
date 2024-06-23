import os
import sys

import pandas as pd
import pymongo
from dataclasses import dataclass

from src.configure.configure import MongoDBConfig
from src.logger import logging
from src.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join('artifacts', 'preprocessed_train.csv')
    test_path: str = os.path.join('artifacts', 'preprocessed_test.csv')



class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion has started.")
        
        try:
            logging.info("Import train raw data from MongoDB.")
            
            client = pymongo.MongoClient(MongoDBConfig.CLIENT)
            db = client["aifare_dataset"]
            collection = db["preprocessed_train_data"]
            mongo_data = collection.find({})
            train_set = pd.DataFrame(list(mongo_data))
            pd.set_option('display.max_columns', None)

            logging.info("Train data imported and saved.")


            logging.info("Import test raw data from MongoDB.")

            client = pymongo.MongoClient(MongoDBConfig.CLIENT)
            db = client["aifare_dataset"] 
            collection = db["preprocessed_test_data"]
            mongo_data = collection.find({})
            test_set = pd.DataFrame(list(mongo_data))
            pd.set_option('display.max_columns', None)

            logging.info("Test data imported and saved.")


            os.makedirs(os.path.dirname(self.data_ingestion_config.train_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.train_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_path, index=False)


            return (self.data_ingestion_config.train_path,
                    self.data_ingestion_config.test_path)
        

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    logging.info(f"Data saved in train path: {train_path} and test path: {test_path}")
    logging.info("Data Ingestion has ended.")
