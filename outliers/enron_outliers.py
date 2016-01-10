#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
import operator
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

#marks down the names with the top salary and bonus
key_list = [k for k in data_dict.keys() if data_dict[k]["salary"] != 'NaN' and (data_dict[k]["salary"] > 1000000 and data_dict[k]["bonus"] > 5000000)]
print "before ", key_list

#remove the TOTAL key. It is an outlier!
data_dict.pop('TOTAL', 0)

#mark down the names wuth the top salary and bonus after removing the TOTAL
key_list = [k for k in data_dict.keys() if data_dict[k]["salary"] != 'NaN' and (data_dict[k]["salary"] > 1000000 and data_dict[k]["bonus"] > 5000000)]
print "after ", key_list


data = featureFormat(data_dict, features)


### your code below

for point in data:
#	print point
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
    

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()