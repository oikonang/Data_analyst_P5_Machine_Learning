#!/usr/bin/python


import numpy

def getKey(item):
    return item[2]

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []


    errors = predictions - net_worths
    #print ages[0][0], net_worths[0][0], errors[0][0]
    for i in range(0, len(predictions)):
        cleaned_data.append((ages[i][0], net_worths[i][0], errors[i][0]**2))
    
    #print "before cleaning: ", cleaned_data[:10]
    cleaned_data = sorted(cleaned_data, key=getKey)
    #print "after cleaning: ", cleaned_data[:10]

    return cleaned_data[:81]
