#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
companies= pd.read_csv("Churn_Modelling.csv")

x=companies.iloc[ : , 3:12].values
y=companies.iloc[:, 12].values


#we have no worde in x as you can see because it cnnot be processed by the ANN
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode= LabelEncoder()
x[:,1]=encode.fit_transform(x[:,1])
x[:,2]= encode.fit_transform(x[:,2])
#Now to delete the words off the dataset we do a one hot encoder to do th
#Now we use model_selection function to introducer our training and testing dsets of variables
from sklearn.model_selection import train_test_split
#these are the sets and we choose the random stae to be 0.2 because we want to use 20% of the entire data size

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#you can see from the side bar that the number of values for x_train is now 20% of 10000 




#This part is new: The standard scalar function will ignore datas that are far from the mean (also putting the STD =1, to allow data points to be closer to each other) as a resul best results are found from the ANN.
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test= scale.fit_transform(x_test)

#Building ANN
from keras.models import Sequential
from keras.layers import Dense, Activation

#first main layer which contains data
seq=Sequential()
#Hidden layer #1

#The unis=6, the 6 comesfrom 11 outputs into the hidden layer, becase x_test has 11 output values and we have 1 output for each of the people either 1 or 0 (either they are staying or leaving the bank)

#kernel_initializer is the first itteration of randomly assigning the weights 
seq.add(Dense(units=6, activation='relu', kernel_initializer='uniform', input_dim=10))
#make sure you put input_dim because you are telling the number of inputs it is recieving


#Hidden layer #2
seq.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
#Output layer
seq.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

#Gradient descent is to find where the minimum cost function occurs
#running the ANN we need to udse the graduient discend and then use cross entropy to minimize these values within the result to get the most accurate one. 
#adam uses gradient descent to minimize the weights
#The loss function gives us an idea of how wrong we are compared to the actual answer 
seq.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#batch_size = the number of pepole we want to analyze at once and epoch corresponds to the number of itterations we want our model to do

seq.fit(x_train, y_train, batch_size=10, epochs=200)
y_pred=seq.predict(x_test)

#We can now use a boolean to limit the results into simple true/false statements
y_pred=(y_pred>0.5)

#Now if we want to know which of the results correspond to correct output we can use a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




















