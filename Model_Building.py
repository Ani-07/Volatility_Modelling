# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 18:45:13 2020

@author: Anirudh Raghavan
"""

import pandas as pd
import numpy as np
import os

seed_value= 0

os.environ['PYTHONHASHSEED']=str(seed_value)

import random
random.seed(seed_value)

np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)

# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


############################################################################

#from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#############################################################################
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

######################################################################################

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from numpy.random import randint

###############################################################################

new_loc = r"C:\Users\Anirudh Raghavan\Desktop\Stevens - Courses\Fall 2020\FE 690 - Machine Learning\Assignment_3_Volatility_Modelling\Data"

os.chdir(new_loc)

file = open("train_labels.txt", 'r')
y_train = file.readlines()
y_train  = np.array(y_train).astype(np.float64)


file = open("test_labels.txt", 'r')
y_test = file.readlines()
y_test  = np.array(y_test).astype(np.float64)


X_train  = pd.read_csv("train_features.csv")
X_test  = pd.read_csv("test_features.csv")

column_names = ["Name","Trial 1","Trial 2","Trial 3","Trial 4","Trial 5","CV"]
cross_val_summary = pd.DataFrame(columns = column_names)

def cv_append(cv, name,column_names):
    row_dict = {}
    for i in range(len(column_names)):
    
        if i == 0:
            row_dict[column_names[i]] = name
        
        elif i == 6:
            row_dict[column_names[i]] = np.mean(cv)
        
        else:
            row_dict[column_names[i]] = cv[i-1]
    
    return row_dict


acc_table = pd.DataFrame(columns = ["Name","Accuracy Rate"])


##############################################################################
# MODEL BUILDING
##############################################################################
## Model 1 - Gaussian Naive Bayes

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

# In order to measure the performance of our classifer, we shall use a cross
# validation performance score

gnb_cv = cross_val_score(gnb, X_train, y_train, cv=5, scoring="accuracy")

gnb_cv_dict = cv_append(gnb_cv, "Naive Bayes",column_names)
    
cross_val_summary = cross_val_summary.append(gnb_cv_dict, ignore_index = True)



#Predict the response for test dataset
y_pred = gnb.predict(X_test)

# Build Confusion Matrix

nb_conf = confusion_matrix(y_test, y_pred)

nb_acc = accuracy_score(y_pred,y_test)
print('Accuracy is:', round(nb_acc *100,4))


acc_table = acc_table.append({'Name':"Naive Bayes",'Accuracy Rate':nb_acc}, 
                             ignore_index = True)


###############################################################################

# Model 2 - Logistic Regression

# Fit Model

log_reg = LogisticRegression(random_state=0, max_iter = 2000)

log_reg.fit(X_train, y_train)

# Measure general accuracy of model with cross validation

log_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring="accuracy")

log_cv_dict = cv_append(log_cv, "Logistic Regression", column_names)
    
cross_val_summary = cross_val_summary.append(log_cv_dict, ignore_index = True)

# Predict the response for test dataset
y_pred_log = log_reg.predict(X_test)


# Build Confusion Matrix
log_conf = confusion_matrix(y_test, y_pred_log)

log_acc = accuracy_score(y_pred_log,y_test)
print('Accuracy is:', round(log_acc*100,4))

acc_table = acc_table.append({'Name':"Logistic Regression",'Accruacy Rate':log_acc}, 
                             ignore_index = True)




###########################################################################

SVM_Linear = SVC(kernel = "linear")

SVM_Linear.fit(X_train, y_train)  

svml_cv = cross_val_score(SVM_Linear, X_train, y_train, cv=5, scoring="accuracy")

svml_cv_dict = cv_append(svml_cv, "Linear_SVM",column_names)
    
cross_val_summary = cross_val_summary.append(svml_cv_dict, ignore_index = True)

y_pred_svml = SVM_Linear.predict(X_test)

svml_conf = confusion_matrix(y_test, y_pred_svml)

svml_acc = accuracy_score(y_pred_svml,y_test)
print('Accuracy is:', round(svml_acc*100,4))

acc_table = acc_table.append({'Name':"Linear_SVM",'Accruacy Rate':svml_acc}, 
                             ignore_index = True)


###########################################################################

SVM_radial = SVC()

SVM_radial.fit(X_train, y_train)  

svmr_cv = cross_val_score(SVM_radial, X_train, y_train, cv=5, scoring="accuracy")

svmr_cv_dict = cv_append(svmr_cv, "Radial_SVM",column_names)
    
cross_val_summary = cross_val_summary.append(svmr_cv_dict, ignore_index = True)

y_pred_svmr = SVM_radial.predict(X_test)

svmr_conf = confusion_matrix(y_test, y_pred_svmr)

svmr_acc = accuracy_score(y_pred_svmr,y_test)
print('Accuracy is:', round(svmr_acc*100,4))

acc_table = acc_table.append({'Name':"Radial_SVM",'Accruacy Rate':svmr_acc}, 
                             ignore_index = True)

#############################################################################

###################################################################################

# Build Neural Network on the train set and test accuracy on test set
# We shall now begin building the model using the keras library

#y_nn_tr = pd.get_dummies(y_train).values.astype(np.float64)
#y_nn_te = pd.get_dummies(y_test).values.astype(np.float64)


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

n = X_train.shape[0]
DNN_cv = []

for i in range(5):
    tr = randint(0, n, int(0.8*n))
    DNN_X_tr = X_train[tr]
    DNN_y_tr = y_train[tr]
    
    DNN_X_te = X_train[-tr]
    DNN_y_te = y_train[-tr]



    # Building Model Architecture
    model = Sequential()
    model.add(Dense(1000, input_dim=14381, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Model Loss and Optimizer with learning rate 0.1
    opt = keras.optimizers.SGD(learning_rate=0.0001)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Using the loss and optimizer to learn the weights
    
    # We shall use 6 epochs as a limit for the number of passes of the training data

    epoch = 6
    
    history = model.fit(DNN_X_tr, DNN_y_tr, epochs=epoch, batch_size=20)

    # Now, we shall apply this model on the test set and obtain the accuracy
    y_pred = model.predict(DNN_X_te)
    
    #Converting predictions to labels
    pred = [1 if i >= 0.5 else 0 for i in y_pred]
                #Measure accuracy
    acc = accuracy_score(pred,DNN_y_te)
    print('Accuracy is:', round(acc*100,4))
    
    DNN_cv.append(acc)




DNN_cv_dict = cv_append(DNN_cv, "Deep Neural Net", column_names)   
cross_val_summary = cross_val_summary.append(DNN_cv_dict , ignore_index = True)


model = Sequential()
model.add(Dense(1000, input_dim=14381, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
    
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
    
# Model Loss and Optimizer with learning rate 0.1
opt = keras.optimizers.SGD(learning_rate=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# Using the loss and optimizer to learn the weights
    
# We shall use 6 epochs as a limit for the number of passes of the training data

epoch = 6
    
history = model.fit(X_train, y_train, epochs=epoch, batch_size=20)

# Now, we shall apply this model on the test set and obtain the accuracy
y_pred = model.predict(X_test)
    
#Converting predictions to labels
pred = [1 if i >= 0.5 else 0 for i in y_pred]
                #Measure accuracy
acc = accuracy_score(pred,y_test)
print('Accuracy is:', round(acc*100,4))
    
acc_table = acc_table.append({'Name':"Deep Neural Net",'Accruacy Rate':acc}, 
                             ignore_index = True)


#############################################################################
# ROC Curve
############################################################################

