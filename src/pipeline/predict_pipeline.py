import sys
import os
import pandas as pd

from src.logger import logging
from src.exception import CustomException

from src.utiles import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","trained_model.pkl")
            preprocessor_path=os.path.join("artifacts","pre_processor.pkl")

            
            #print("Before Loading")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            #print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(sys,e)