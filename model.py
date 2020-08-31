import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import pickle

data=pd.read_csv("./predic.csv")

data['Kw [m/bar/min]']=data['Kw [m/bar/min]']*10000

data=data.dropna()

X=data.iloc[:,:4]
y=data['Kw [m/bar/min]']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
from sklearn.preprocessing import StandardScaler

regr = MLPRegressor(random_state=1,solver='adam', max_iter=500, activation='relu',learning_rate_init=0.01,hidden_layer_sizes=100)

model = Pipeline([('scaler', StandardScaler()),("regr", regr) ])

model.fit(X_train, y_train)

pickle.dump(model, open('model.pkl','wb'))