import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

def predict_datapoint():
    
    data=CustomData(
        location = input("Enter location "),
        total_sqft = float(input("Enter total sqft ")),
        bath = float(input("Enter no of bath ")),
        bhk = int(input('Enter no of bhk ')),
    )
    pred_df=data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline=PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return results
    
if __name__ == "__main__":
    predicted_price = predict_datapoint()
    print(f"Predicted Price {predicted_price[0]}")
