from dataclasses import dataclass
import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utiles import save_object
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationconfig:
     pre_processor_obj_file_path=os.path.join('artifacts','pre_processor.pkl')

class  Data_Transformation:
    def __init__(self):
         self.Data_Transformation_config=DataTransformationconfig()

    def pipeline_built(self):
        try:
            numerical_columns =["writing_score", "reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numeric_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ("scaler",StandardScaler())
                       ]
            )
            categorical_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                       ("One_hot_encoding",OneHotEncoder()),
                       ("Scaler",StandardScaler(with_mean=False))
                       ]
            )
            
            logging.info(f"Categorical pipline done columns: {categorical_columns}")
            logging.info(f"Numerical pipline done columns: {numerical_columns}")
            
            pre_processor_transformer=ColumnTransformer([
                ("Numerical Pipeline",numeric_pipeline,numerical_columns),
                ("Categorical Pipeline",categorical_pipeline,categorical_columns)
            ])

            logging.info("data Pre-processing pipeline built sucessfully")
            
            return pre_processor_transformer 
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def Initiate_data_to_Transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Reading the Train and Test data")

            data_pipeline_obj=self.pipeline_built()

            target_column_name=["math_score"]
            feaure_column_name=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "writing_score", 
                "reading_score"
            ]

            train_input_feature_df=train_df.drop(columns=target_column_name,axis=1)
            train_target_feature_df=train_df["math_score"]

            test_input_feature_df=test_df.drop(columns=target_column_name,axis=1)
            test_target_feature_df=test_df["math_score"]

            logging.info("slpiting train and test data is done")

            final_train_input_feature_arr= data_pipeline_obj.fit_transform(train_input_feature_df)
            final_test_input_feature_arr= data_pipeline_obj.transform(test_input_feature_df)

            train_arr = np.c_[
                final_train_input_feature_arr, np.array(train_target_feature_df)
            ]
            test_arr = np.c_[final_test_input_feature_arr, np.array(test_target_feature_df)]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe")

            save_object(
                file_path=self.Data_Transformation_config.pre_processor_obj_file_path,
                obj=data_pipeline_obj
            )
            return(
                train_arr,
                test_arr,
                self.Data_Transformation_config.pre_processor_obj_file_path
            )        
        except Exception as e:
            raise CustomException(e,sys)

     
