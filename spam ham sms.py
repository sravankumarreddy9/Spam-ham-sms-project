# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 11:54:00 2019

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
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

data = pd.read_csv("sms_spam_data.csv", encoding='latin-1')
data=data.rename(columns={'type':'label','text':'message'})
data.head()
data.tail()
data.describe()
data.groupby('message').describe()
data['length'] = data['message'].apply(len)
data.hist(by='label',column='length',bins=30,figsize=[15,5])
data['num']=data.label.map({'ham':0,'spam':1})
data.shape
lower_case_documents=[]
for i in data['message']:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

#removing punctuations
punctuation_documents=[]
for i in lower_case_documents:
    punctuation_documents.append(i.translate(str.maketrans('', '', string.punctuation)))
    
spaceremove_documents = []
for i in punctuation_documents:
    spaceremove_documents.append(i.split(' '))
print(spaceremove_documents)

frequency_list = []
import pprint
from collections import Counter

for i in spaceremove_documents:
    frequency_count = Counter(i)
    frequency_list.append(frequency_count)
pprint.pprint(frequency_list)

from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
count_vector.fit(data['message'])
count_vector.get_feature_names()

array_list = count_vector.transform(data['message']).toarray()
doc_array
frequency = pd.DataFrame(array_list, columns = count_vector.get_feature_names())
frequency

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['message'], 
                                                    data['label'], 
                                                    random_state=1)
print('Number of rows in the total set: {}'.format(data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
count_vector = CountVectorizer()
training_data = count_vector.fit_transform(X_train)
testing_data = count_vector.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)
predictions=pd.DataFrame(predictions)
predictions=predictions.rename(columns={'0':'label'})
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(predictions, y_test)))

print ('confusion matrix\n', confusion_matrix(y_test['label'], predictions))
print ('row=expected, col=predicted')

print('Precision score: ', format(precision_score(predictions, y_test, pos_label="ham")))
print('Precision score: ', format(precision_score(predictions, y_test, pos_label="spam")))

print('Recall score: ', format(recall_score(predictions, y_test, pos_label="ham")))
print('Recall score: ', format(recall_score(predictions, y_test, pos_label="spam")))

print('F1 score: ', format(f1_score(predictions, y_test, pos_label="ham")))
print('F1 score: ', format(f1_score(predictions, y_test, pos_label="spam")))


print('accuracy', accuracy_score(data['label'], testMessages['prediction']))
print('confusion matrix\n', confusion_matrix(testMessages['label'], testMessages['prediction']))
print('(row=expected, col=predicted)')

print("\n\n[+] lets test with other unique msgs other than the datasets used: ") 
t=[input("[+]enter a text msg to test : ")]
t=np.array(t)
t=count_vector.transform(t)
prediction = naive_bayes.predict(t)
if prediction == 0:
    print("ham")
else:
    print("Spam")