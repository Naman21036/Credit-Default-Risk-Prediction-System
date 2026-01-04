import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training pipeline started")

            # Step 1: Data Ingestion
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data ingestion completed")

            # Step 2: Data Transformation
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_path,
                    test_path
                )
            )

            logging.info("Data transformation completed")

            # Step 3: Model Training
            model_trainer = ModelTrainer()
            roc_auc = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info(f"Model training completed. Final ROC AUC: {roc_auc}")

            return roc_auc

        except Exception as e:
            logging.error("Error occurred in training pipeline")
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    final_score = pipeline.run_pipeline()
    print("Training completed. Test ROC AUC:", final_score)
