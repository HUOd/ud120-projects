#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
clf = svm.SVC(kernel='rbf', C=10000.)

#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

clf.fit(features_train, labels_train)

prediction = clf.predict(features_test)
print("The predict for 10th:", prediction[10])
print("The predict for 26th:", prediction[26])
print("The predict for 50th:", prediction[50])

count = 0
for p in prediction:
    if p == 1:
        count += 1

print("Number of 1s:", count)

accuracy = accuracy_score(prediction, labels_test)
print("Accuracy is:", accuracy)
#########################################################
