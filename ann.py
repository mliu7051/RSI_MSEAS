#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:03:53 2022

@author: marianneliu
"""
#%%

#----------------------- import packages --------------------#
#importing required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.layers as tfl

from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV


from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, accuracy_score


#%%

#----------------------- Data Pre-processing ----------------------#
# Checking the tensorflow version
print(tf.__version__)

def standardize(col, df):
    mean = df[col].mean()
    std = df[col].std()
    new_vals = []
    for x in df[col]:
        new = (x - mean)/std
        new_vals.append(new)
    df[col] = new_vals
    return df

# Loading the data
sample_data = pd.read_csv("sample.csv")
sample_data = standardize('depth', sample_data)
sample_data = standardize('temp', sample_data)
sample_data = standardize('salinity', sample_data)
sample_data = standardize('nitraite', sample_data)


dataset = sample_data.values
X = dataset[:,0:3]
y = dataset[:,3]


#%%

# lr=0.001, b1=0.9, b2=0.999, epsi=1e-7


#----------------------- Building Base Model ----------------------#
def baseline_model():
    model = Sequential()
    model.add(Dense(3, input_dim=3, activation='relu', use_bias=True))
    model.add(Dense(7, input_dim=3, activation='relu', use_bias=True))
    model.add(Dense(15, activation='relu', use_bias=True))
    model.add(Dense(5, activation='relu', use_bias=True))
    model.add(Dense(1))
    
    
    #Z1 = tfl.Dense(3, activation="relu", use_bias=True)
    #Z2 = tfl.Dense(20, activation="relu", use_bias=True)(Z1)
    #Z3 = tfl.Dense(10, activation="relu", use_bias=True)(Z2)
    #Z4 = tfl.Dense(5, activation="relu", use_bias=True)(Z3)
    
    #oututs = tfl.Dense(units=1)(Z4)
    #model = tf.keras.Model(inputs=X, outputs=outputs)
              
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model


#------------------------------------------------------------------------#

estimator = KerasRegressor(build_fn=baseline_model, batch_size=5, verbose=0)
# split into train, test val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
history = estimator.fit(X_train, y_train, validation_split=0.33, epochs=200, batch_size=20, verbose=0)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('Model Training/Validation Loss')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%

#----------------------- Cross-Validation ----------------------#
#kfold = KFold(n = X.shape[0], n_folds=5, random_state=7)
kfold = KFold(n_splits=5)

# function to return mean loss
def score(self, X, y, **kwargs):
    '''Returns the mean loss on the given test data and labels.

    # Arguments
        X: array-like, shape `(n_samples, n_features)`
            Test samples where n_samples in the number of samples
            and n_features is the number of features.
        y: array-like, shape `(n_samples,)`
            True labels for X.
        kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate`.

    # Returns
        score: float
            Mean accuracy of predictions on X wrt. y.
    '''
    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
    loss = self.model.evaluate(X, y, **kwargs)
    if type(loss) is list:
        return -loss[0]
    return -loss


results = cross_val_score(estimator, X_test, y_test, cv=kfold)
score = estimator.score(X_test, y_test)

print("Score:" + str(score))
#print("Results mean: %.2f" % results.mean())
#print("Results std: %.3f" % results.std())


predictions = estimator.predict(X_test)
sum = 0
for i in range(len(predictions)):
    sum += abs(predictions[i]-y_test[i])
    
mae = sum/len(predictions)*3.963361443700686+4.5265442

print("MAE: "+str(mae))

#%%
"""
#----------------------- Hyperparameter Optimization ----------------------#

space = dict()
space['batch_size'] = [int(x) for x in np.linspace(1, 500, num = 10)]
space['epochs'] = [int(x) for x in np.linspace(1,500, num = 10)]
#space['learning_rate'] = [0.00001, 1]
#space['beta_1'] = [0.5, 0.9999]
#space['beta_2'] = [0.5, 0.9999]
#space['epsilon'] = [1e-11, 1.0]


search = RandomizedSearchCV(estimator = estimator, param_distributions = space, n_iter=500, scoring='neg_mean_absolute_error', n_jobs=-1, cv=kfold)
result = search.fit(X_train, y_train)

print('Best Score: %s' % result.best_score_)
print('Best Hyperparameters: %s' % result.best_params_)
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

"""
#%%

"""
#----------------------- Standardizing Dataset ----------------------#
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
#kfold = KFold(n=X.shape[0], n_folds=10, random_state=7)
kfold = KFold(n_splits=10)

results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

"""












#%%

#-------------------------- Extra ---------------------------------#

"""
# Taking  all rows and all columns in the data except the last column as X (feature matrix)
#the row numbers and customer id's are not necessary for the modelling so we get rid of and start with credit score
X = sample_data.iloc[:,:-1].values
print("Independent variables are:", X)
#taking all rows but only the last column as Y(dependent variable)
y = sample_data.iloc[:, -1].values
print("Dependent variable is:", y)



# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#printing the dimensions of each of those snapshots to see amount of rows and columns i each of them
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# Data Scaling/normalization of the features that will go to the NN
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#----------------------- Building the model -----------------------#

# Initializing the ANN by calling the Sequential class fromm keras of Tensorflow
ann = tf.keras.models.Sequential()

#----------------------------------------------------------------------------------
# Adding "fully connected" INPUT layer to the Sequential ANN by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


#----------------------------------------------------------------------------------
# Adding "fully connected" SECOND layer to the Sequential AMM by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 6 and Activation Function = Rectifier
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))


#----------------------------------------------------------------------------------
# Adding "fully connected" OUTPUT layer to the Sequential ANN by calling Dense class
#----------------------------------------------------------------------------------
# Number of Units = 1 and Activation Function = Sigmoid
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#----------------------- Training the model -----------------------#
# Compiling the ANN
# Type of Optimizer = Adam Optimizer, Loss Function =  crossentropy for binary dependent variable, and Optimization is done w.r.t. accuracy
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN model on training set  (fit method always the same)
# batch_size = 32, the default value, number of epochs  = 100
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#----------------------- Evaluating the Model ---------------------#
# the goal is to use this ANN model to predict the probability of the customer leaving the bank
# Predicting the churn probability for single observations

#Geography: French
#Credit Score:600
#Gender: Male
#Age: 40 years old
#Tenure: 3 years
#Balance: $60000
#Number of Products: 2
#with Credit Card
#Active member
#Estimated Salary: $50000

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# this customer has 4% chance to leave the bank

 
#show the vector of predictions and real values
#probabilities
y_pred_prob = ann.predict(X_test)

#probabilities to binary
y_pred = (y_pred_prob > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", confusion_matrix)
print("Accuracy Score", accuracy_score(y_test, y_pred))
"""