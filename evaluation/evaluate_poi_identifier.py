#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

### split the data!  
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size = .3, random_state = 42)

### create Decision Tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### print accuracy of prediction
print "Accuracy: ", accuracy_score(pred, labels_test)

### print No of POIs in the test data
count_POIs = 0
for i in pred:
	if i == 1:
		count_POIs +=1
print "No of POIs in the test data: ", count_POIs

### print No of people in the test set
print "No of total population in test data: ", len(pred)

### print accuracy if there was no POI in the test set
pred2 = [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
print "Accuracy 2: ", accuracy_score(pred2, labels_test)

### print matrix
print confusion_matrix(labels_test, pred)

### print precision
print "Precision: ", precision_score(labels_test, pred)

### print recall
print "Recall: ", recall_score(labels_test, pred)
