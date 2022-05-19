#!/usr/bin/python

import sys
import pickle
import os
sys.path.append(os.path.abspath(("../tools/")))

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'deferred_income', 'total_stock_value', 'expenses', 'poi_mail_ratio']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

removed_outliers = []
for name, values in data_dict.items():
    if values["from_messages"] != "NaN" and float(values["from_poi_to_this_person"]) / float(values["from_messages"]) < 0.01:
        removed_outliers.append(name)

for name in removed_outliers:
    data_dict.pop(name)

print("Number of data:", len(data_dict))

### Task 3: Create new feature(s)

for name, values in data_dict.items():
    values["poi_mail_ratio"] = 0.0
    if values["from_messages"] != "NaN":
        pm_ratio = float(values["from_poi_to_this_person"]) / float(values["from_messages"])
        values["poi_mail_ratio"] = pm_ratio

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()

# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators=5, random_state=5)

# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(min_samples_split=10)

from sklearn.svm import SVC
clf = SVC(kernel='sigmoid', C=10000000.)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

from time import time
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

prediction = clf.predict(features_test)

#Evaluation metrics
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, prediction)
print("accuracy:", accuracy)

from sklearn.metrics import precision_score
precision = precision_score(labels_test, prediction)
print("precision:", precision)

from sklearn.metrics import recall_score
recall = recall_score(labels_test, prediction)
print("recall:", recall)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)