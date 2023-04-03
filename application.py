from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method== 'GET':
        return render_template('home.html')
    else:
       
        data=CustomData(
            location = request.form.get('location'),
            total_sqft = request.form.get('total_sqft'),
            bath = request.form.get('bath'),
            bhk = request.form.get('bhk'),
        )
        
        print(data)
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        p_bhk = int(pred_df['bhk'][0])
        p_bath = int(pred_df['bath'][0])
        p_sqft = int(pred_df['total_sqft'][0])

        if (p_bhk >= p_bath) and ((p_sqft/p_bhk) >= 300):
            predict_pipeline=PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print(results)
            return render_template('home.html',results=f"Estimated Price in Lakhs : {results[0]}")
        
        elif (p_bhk < p_bath) or ((p_sqft/p_bhk) < 300):
            return render_template('home.html',results = "-BHK must be greater than or equal to no of bathrooms --Area per BHK must be greater than 300 sqft")
        
        else:
            return render_template('home.html')
        
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")
