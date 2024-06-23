import sys
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline

# Starting the pipeline
def start_training():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()

    except Exception as e:
        raise TrainingPipeline(e, sys)
    
    
if __name__ == "__main__":
    start_training()