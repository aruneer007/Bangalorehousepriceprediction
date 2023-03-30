import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            print("Entered predict")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path =os.path.join("artifacts","preprocessor.pkl")
            print("Before loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features).toarray()
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 location : str,
                 total_sqft : float,
                 bath : float,
                 bhk : int
                 ):
        self.location = location
        self.total_sqft = total_sqft
        self.bath = bath
        self.bhk = bhk
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "location" : [self.location],
                "total_sqft" : [self.total_sqft],
                "bath" : [self.bath],
                "bhk" : [self.bath],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)