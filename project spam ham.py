# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:22:15 2019

@author: reddymv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

data = pd.read_csv("sms_spam_data.csv", encoding='latin-1')
data.head()
data.tail()
#checking for null values
data.isna().sum()
#renaming columns
data=data.rename(columns={'type':'label','text':'message'})
data.describe()
data.groupby('message').describe()
data.groupby('label').describe()
data=pd.DataFrame(data)
data['length'] = data['message'].apply(len)
#histogram to analyze ham and spam 
data.hist(by='label',column='length',bins=30,figsize=[15,5])
#ham and spam to 0 and 1
data['num']=data.label.map({'ham':0,'spam':1})
#choosing independent and dependent variable
X=data["message"]
Y=data["num"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
vect = CountVectorizer()
#converting features into numeric vector
X_train = vect.fit_transform(x_train)
#converting target into numeric vector
X_test = vect.transform(x_test)

svc = SVC(kernel = 'linear')
mnb = MultinomialNB(alpha =0.2)
gnb  = GaussianNB()
lr = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=100,random_state=11)
abc = AdaBoostClassifier(n_estimators =100,random_state=11)

def training(clf,x_train,Y_train):
    clf.fit(x_train,Y_train)
    
#function for predicting labels

def predict(clf,X_test):
    return clf.predict(X_test)

classifier={'SVM': svc , 'MultinomialNB': mnb,'GaussianNB': gnb,'logistic': lr,'RandomForest': rfc,'Adaboost': abc}

score = []
for n,c in classifier.items():
    training(c,X_train.toarray(),y_train)
    pred = predict(c,X_test.toarray())
    score.append((n,[accuracy_score(y_test,pred,normalize=True)]))
    
accu=pd.DataFrame.from_items(score,orient='index',columns=['scores'])
#Adding accuracy column
accu['Accuracy (%)']=accu['scores']*100
accu

#Naive bayes classifier