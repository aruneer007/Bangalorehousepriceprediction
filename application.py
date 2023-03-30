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

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        print(results)
        return render_template('home.html',results=results[0])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")
