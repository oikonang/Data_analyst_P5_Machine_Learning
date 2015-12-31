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
		return count

def max_total_payments_person():
	sjk = enron_data['SKILLING JEFFREY K']['total_payments']
	lkl = enron_data['LAY KENNETH L']['total_payments']
	fas = enron_data['FASTOW ANDREW S']['total_payments']
	max_of_three = max([sjk, lkl, fas])
	for p in enron_data:
		if enron_data[p]['total_payments'] == max_of_three:
			return p

def no_of_salaries():
	count = 0
	for p in enron_data:
		if enron_data[p]['salary'] != 'NaN':
			count = count + 1
	return count

def no_of_emails():
	count = 0
	for p in enron_data:
		if enron_data[p]['email_address'] != 'NaN':
			count = count + 1
	return count


print "No. of persons: %d" % len(keys)
print "No. of variables for each person: %d" % len(features_dict[0])
#print "No. of pois in the dataset: %d" % sum( [ enron_data[p]['poi'] for p in enron_data ] )
print "No. of pois in the dataset: ", count_poi()
print "No. of pois in the text: ", count_poi_from_text()
print "Total value of stock belonging to James Prentice: ", enron_data['PRENTICE JAMES']['total_stock_value']
print "No. of emails to poi by Wesley Colwell: ", enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print "Stock options exercised by Jeffrey Skilling: ", enron_data['SKILLING JEFFREY K']['exercised_stock_options']
print "Total payments of Jeffrey Skilling: ", enron_data['SKILLING JEFFREY K']['total_payments']
print "Total payments of Kenneth Lay: ", enron_data['LAY KENNETH L']['total_payments']
print "Total payments of Andrew Fastow: ", enron_data['FASTOW ANDREW S']['total_payments']
print "Max total payments person between the top 3 executives: ", max_total_payments_person()
print "No. of persons that have a quantified salary:", no_of_salaries()
print "No. of persons that have known emails:", no_of_emails()