#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

keys = enron_data.keys()
features_dict = enron_data.values()

def count_poi():
	count = 0
	for person in enron_data:
		if enron_data[person]['poi'] == True:
				count = count + 1
	return count

def count_poi_from_text():
	poi_names = open('../final_project/poi_names.txt', 'r')
	a = poi_names.read()

	for count, line in enumerate(a):
		return enumerate(a)

print "No. of persons: %d" % len(keys)
print "No. of variables for each person: %d" % len(features_dict[0])
#print "No. of pois in the dataset: %d" % sum( [ enron_data[p]['poi'] for p in enron_data ] )
print "No. of pois in the dataset: ", count_poi()
print "No. of pois in the text: ", count_poi_from_text()


