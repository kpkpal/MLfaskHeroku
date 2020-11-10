import os
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, url_for
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
model_linr = pickle.load(open('model_linr.pkl', 'rb'))
model_logr = pickle.load(open('model_logr.pkl', 'rb'))

@app.route('/')
@app.route("/home")
def home():
    return render_template('index.html')

@app.route("/logr")
def logr():
    return render_template('logr.html', title='logistic_regression')

@app.route("/contact")
def contact():
    return render_template('contact.html', title='contact')

# Linear Regression:
@app.route('/predicted_linrOP',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model_linr.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

# Logistic Regression
@app.route('/predict_logrOP',methods=['POST'])
def predict_logr():
    '''
    For rendering results on HTML GUI
    '''
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model_logr.predict(final_features) # making prediction

    return render_template('logr.html', prediction_text_logr='Predicted Species: {}'.format(prediction)) # rendering the predicted result


if __name__ == "__main__":
    app.run(debug=False, threaded=True)

