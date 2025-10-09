import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
ss=StandardScaler()

m_o_d_e_l = pickle.load(open('pipeline_model.pkl','rb'))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    SepalLengthCm = request.form['SepalLengthCm']
    SepalWidthCm = request.form['SepalWidthCm']
    PetalLengthCm = request.form['PetalLengthCm']
    PetalWidthCm = request.form['PetalWidthCm']
    
    features = np.array([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    prediction = m_o_d_e_l.predict(features).reshape(1,-1)

    return render_template('index.html',output = prediction[0])

if __name__ == '__main__':
    app.run(debug=True)