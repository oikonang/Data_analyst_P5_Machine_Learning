#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from pprint import pprint
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit



###################################################################################
print "### Task 1: Select what features you'll use."
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
target = 'poi'

features_list_email = ['from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages']

features_list_financial = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value']

features_list = [target] + features_list_financial + features_list_email
print "Features_list: \n", features_list
print ""

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)    

# count the total No of observations and total Persons of Interest
def count_poi():
    count_observations = 0
    count_poi = 0
    for person in data_dict:
        count_observations = count_observations + 1
        if data_dict[person]['poi'] == True:
            count_poi = count_poi + 1
    return count_observations, count_poi
print "The total number of observations and total Persons of Interest in the dataset: ", count_poi()

# count No of persons with missing(Nan) values and their percentage compared to total
def no_of_missing(feature):
    count_nan = 0
    count_p = 0
    for person in data_dict:
        count_p = count_p + 1
        if data_dict[person][feature] == 'NaN':
            count_nan = count_nan + 1
    return count_nan, float(count_nan)/float(count_p)

print "\nMissing values for 'salary' feature and percentage to total: ", no_of_missing("salary")
print "\nMissing values for 'bonus' feature and percentage to total: ", no_of_missing("bonus")
print "\nMissing values for 'exercised_stock_options' feature and percentage to total: ", no_of_missing("exercised_stock_options")
print "\nMissing values for 'total_payments' feature and percentage to total: ", no_of_missing("total_payments")

###################################################################################
print "\n### Task 2: Remove outliers"

# bonus outliers
bonus_outliers = []
for name in data_dict:
    value = data_dict[name]['bonus']
    if value == 'NaN':
        continue
    bonus_outliers.append((name,int(value)))

# salary outliers
salary_outliers = []
for name in data_dict:
    value = data_dict[name]['salary']
    if value == 'NaN':
        continue
    salary_outliers.append((name, int(value)))

print "Above are the max values both for salary and bonus. One can see that 'TOTAL' has to be removed: "

pprint(sorted(bonus_outliers,key=lambda x:x[1],reverse=True)[:2])
pprint(sorted(salary_outliers,key=lambda x:x[1],reverse=True)[:2])

# remove the TOTAL key. It is an outlier!
data_dict.pop('TOTAL', 0)
print "'TOTAL' has been removed"
print ""

###################################################################################
print "### Task 3: create new features"
## create rescaled_to_messages
# ignore NaNs for "to_messages"
list_a = []
for name in data_dict:
    list_a.append(data_dict[name]["to_messages"])
list_a = filter(lambda a: a!='NaN' , list_a)

# transform each item of the list to float in order to input into MinMaxScaler
list_a_new = []
for item in list_a:
    list_a_new.append(float(item))

# create scaler for rescaling
scaler_a = MinMaxScaler()
rescaled_to_messages = scaler_a.fit_transform(list_a_new)
print "'rescaled_to_messages' feature has been created"


## create rescaled_from_messages
# ignore NaNs for "from_messages"
list_b = []
for name in data_dict:
    list_b.append(data_dict[name]["from_messages"])
list_b = filter(lambda a: a!='NaN' , list_b)

# transform each item of the list to float in order to input into MinMaxScaler
list_b_new = []
for item in list_b:
    list_b_new.append(float(item))

# create scaler for rescaling
scaler_b = MinMaxScaler()
rescaled_from_messages = scaler_b.fit_transform(list_b_new)
print "'rescaled_from_messages' feature has been created"

## create fraction_from_poi and fraction_to_poi and apply to data_dict
def compute_fraction(poi_messages, all_messages):
    if poi_messages == 'NaN' or all_messages == 'NaN':
        return 0.
    fraction = float(poi_messages) / all_messages
    return fraction

# apply each feature to the compute_fraction function and store to data_dict
for name in data_dict:
    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    data_point["fraction_from_poi"] = compute_fraction(from_poi_to_this_person, to_messages)

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    data_point["fraction_to_poi"] = compute_fraction(from_this_person_to_poi, from_messages)
print "'fraction_to_poi' feature has been created"
print "'fraction_from_poi' feature has been created"
# store rescaled_from_messages
count = 0
for name in data_dict:
    if data_dict[name]["to_messages"] !='NaN':
        data_dict[name]["rescaled_to_messages"] = rescaled_to_messages[count]
        count = count + 1
    else:
        data_dict[name]["rescaled_to_messages"] = 'NaN'

# store rescaled_from_messages
count = 0
for name in data_dict:
    if data_dict[name]["from_messages"] !='NaN':
        data_dict[name]["rescaled_from_messages"] = rescaled_from_messages[count]
        count = count + 1
    else:
        data_dict[name]["rescaled_from_messages"] = 'NaN'

print ""
print "New features have been stored to the dataset and a new features_list has been created"

# create new copies of feature list for grading
new_features_list_1 = features_list + ["fraction_from_poi", "fraction_to_poi"]
new_features_list_2 = new_features_list_1 + ["rescaled_to_messages", "rescaled_from_messages"]
my_dataset = data_dict
print ""


###################################################################################


print "Run K-best to determine best rating features sourced from the new features_list"
# Run k_best to determine best rating features sourced from the initial features_list
def get_k_best(features_list, k):
    data = featureFormat(my_dataset, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores) # exclude the 'poi' feature
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    print "\t k_best.scores: ", sorted_pairs
    k_best_features = dict(sorted_pairs[:k])
    return k_best_features.keys()

# get K-best features
best_features_list = get_k_best(new_features_list_2, 4)
best_features_list.insert(0, 'poi') # insert 'poi' feature
print "\nBest features list: ", best_features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, best_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# split train and testing data
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)


###################################################################################
print "### Task 4,5: Try a varity of classifiers and tune them"
print ""

# Compute a PCA on the dataset
pca = RandomizedPCA(n_components=50, whiten=True).fit(features_train)
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)

#print sorted variances in order to find the first's and second's pca variances
print "First two PC's variances: ", pca.explained_variance_ratio_[:2]

# Tune KNN classifier and find best estimator to apply on the tester
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
metrics = ['chebyshev'] # before tuning all the rest metrics are tested ['minkowski', 'euclidean', 'manhattan']
weights = ['uniform', 'distance']
n_neighbors = [1,2,3,4]
parameters_knn = dict(metric=metrics, weights=weights, n_neighbors=n_neighbors)
clf_knn = GridSearchCV(KNeighborsClassifier(), param_grid=parameters_knn, cv=cv)
clf_knn.fit(features, labels)
print "\n Tuned KNeighborsClassifier estimator is: \n", clf_knn.best_estimator_
print "Best score: ", clf_knn.best_score_
print ""
clf_knn = clf_knn.best_estimator_

# Test precission and recall with the best_features_list
print "Precission and Recall for KNN: "
from tester import test_classifier
test_classifier(clf_knn, my_dataset, best_features_list)

print "## Pipelines ##"

# Tune SVC classifier
svc = SVC()
pca = PCA()
scaler_c = MinMaxScaler()

# create estimators
estimators = [('reduce_dim', pca), ('scale_features', scaler_c), ('svc', svc)]
parameters_pipe = dict(reduce_dim__n_components=[2, 3, 4], svc__C=[0.1, 10, 100], svc__kernel=['linear', 'rbf', 'sigmoid'])

# create GridSearchCV includind pipeline
pipe = Pipeline(estimators)
clf_pipe = GridSearchCV(pipe, param_grid=parameters_pipe, cv=cv)
clf_pipe.fit(features, labels)
print "\n Tuned Pipeline estimator is: \n", clf_pipe.best_estimator_
print "Best score: ", clf_pipe.best_score_
print ""
clf_pipe = clf_pipe.best_estimator_


# Test precission and recall with the best_features_list on the Pipeline
print "Precission and Recall for Pipeline along with GridSearch: "
from tester import test_classifier
test_classifier(clf_pipe, my_dataset, best_features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_knn, my_dataset, best_features_list)