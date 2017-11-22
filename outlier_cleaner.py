#!/usr/bin/python
'''Code Orignally from udacity Machine learning UD120 Class
'''

def outlierCleaner(predictions, base_data, target_data, cut_per):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    from collections import deque, OrderedDict
    
    cleaned_data = []

    cut_threshold = int(round((len(predictions)* cut_per), 0))

    print 'Data Points Cut: ', cut_threshold
    fit = abs(target_data.item(0)-predictions.item(0) ) 
    cleaned_data.append([base_data.item(0), target_data.item(0), fit])

    ### your code goes here
    for i in range(1, len(predictions)):
        fit = abs(target_data.item(i)-predictions.item(i) ) 
    
        if predictions.item(i) > cleaned_data[0][2]:
            cleaned_data.append([base_data.item(i), target_data.item(i), fit])
            cleaned_data =  sorted( cleaned_data, key=lambda t: t[2])


    #print cleaned_data
    return cleaned_data[0:(len(cleaned_data)-cut_threshold)]

