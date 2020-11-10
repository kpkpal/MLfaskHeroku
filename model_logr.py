# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import pickle

# Reading the data
iris = pd.read_csv("iris.csv")
#iris.drop("Id", axis=1, inplace = True)
y = iris['Species']
iris.drop(columns='Species', inplace=True)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model_logr = LogisticRegression()
model_logr.fit(x_train,y_train)

pickle.dump(model_logr,open('model_logr.pkl','wb'))

# Loading model to compare the results
model_logr = pickle.load(open('model_logr.pkl','rb'))
print(model_logr.predict([[1, 2, 2, 4]]))
