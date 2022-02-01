import pandas as pd
import pickle
import numpy as np
from flask import Flask, render_template, request
app = Flask(__name__)
data=pd.read_csv('cleaned_Data.csv')
pipe = pickle.load(open("RidgeModel.PKR",'rb'))
@app.route('/')
def index():

    city = sorted(data['city'].unique())
    return render_template('index.html', city=city)
@app.route('/predict', methods=['POST'])
def predict():
    city = request.form.get('city')
    bedroom = request.form.get('bedrooms')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    print(city,bedroom,bath,sqft)
    input= pd.DataFrame([[city,sqft,bath,bedroom]],columns=['city','Total_Area','baths','bedrooms'])
    prediction = pipe.predict(input)[0]

    return str(np.round(prediction,0))
if __name__=="__main__":
    app.run(debug=True, port=5001)
