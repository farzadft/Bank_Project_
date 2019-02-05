
import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
import seaborn as sb 

companies= pd.read_csv("1000_Companies.csv")

sb.heatmap(companies.corr())
x=companies.iloc[ : , :-1].values

x

y= companies.iloc[ : , 4].values

y

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

encoder= LabelEncoder()
x[:,3]= encoder.fit_transform(x[:,3])

sb.heatmap(companies.corr())

x=(x[:, :-1])

x

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.5, random_state=0)


from sklearn.linear_model import LinearRegression

model= LinearRegression()

model.fit(x_train, y_train)

pd.DataFrame(x_train)

pred=model.predict(x_train)

pred=model.predict(x_train)


model.coef_



model.intercept_




from sklearn.metrics import r2_score





r2_score(y_train,pred)



