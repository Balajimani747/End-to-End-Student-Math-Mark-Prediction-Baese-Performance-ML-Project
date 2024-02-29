import os
import sys
from src.logger import logging
from src.exception import CustomException
import pickle


def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
             pickle.dump(obj, file_obj)
        logging.info("Pickle file for Transformer created")
    except Exception as e:
        raise CustomException(e,sys)

