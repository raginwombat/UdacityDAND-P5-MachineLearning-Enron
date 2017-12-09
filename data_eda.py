#!/usr/bin/python
'''
	The project requires the EDA steps taken to actually generate the model used inthe project. This
	work is messy and dupicates some of the steps taken in the actual project. 	I've broken out that
	work in this file to prevent the main file from getting more clutterd.
'''
import sys
import pickle
sys.path.append("../../tools/")

from feature_format import featureFormat, targetFeatureSplit
from helper import *
from sklearn.cross_validation  import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from outlier_cleaner import outlierCleaner
import operator
from collections import OrderedDict


def edaPreWork():
	### Task 1: Select what features you'll use.
	### features_list is a list of strings, each of which is a feature name.
	### The first feature must be "poi".
	features_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus',
		'restricted_stock_deferred', 'deferred_income', 'from_poi_to_this_person', 
		'exercised_stock_options', 'long_term_incentive', 'from_this_person_to_poi' ] # You will need to use more features

	### Load the dictionary containing the dataset
	with open("final_project_dataset.pkl", "r") as data_file:
		data_dict = pickle.load(data_file)
		printDataStats(data_dict)

		printNonZeroPoints(data_dict)

		printNumPOI(data_dict)

	### Task 2: Remove outliers
	#plot outliers

	#build data dict
		my_data={}

		my_data['label_name'] = [per[0] for per in data_dict.items()]
		for key in  features_list:
			if key != 'poi':
				my_data[key] = [0 if per[1][key]=='NaN' else per[1][key] for per in data_dict.items()]		
		
		#Test plot of some data	
		plotDataPoints(my_data, 'salary' ,  'deferred_income')

		for i,rs in enumerate(my_data['restricted_stock_deferred'] ):
			if rs<0:
				#print i
				#print 'person: '+ label_name[i]
				break

		plotDataPoints(my_data, 'salary' ,  'total_payments')




	#Salary seems like a pretty straight forward and stable metric to use. we'll use it as the predictive key for the  rest of the features
		#check regression for salary
		print "Outlier Testing:"
		#first convert data to the correct numpyarray for linear regression aalysis
		for key in my_data.iterkeys():
			if key != 'label_name':
				my_data[key] = np.reshape( np.array(my_data[key]).astype(np.int), (len(my_data[key]), 1))

		#crunch regression for all of the data against salary
		print "Regression crunching held against salary"
		cleaned_training_data={} #stage dict for cleaned data, using loop for regression crunch is faster than breakind out
		feature_scores ={}
		for target in my_data.iterkeys():
			if target != 'label_name' and target != 'salary':
				print "Reg for: "+target

				salary_train, salary_test, target_train, target_test = train_test_split(my_data['salary'], my_data[target] , test_size=0.5, random_state=42)
				reg = LinearRegression()
				reg.fit(salary_train, target_train)
				
				#store feature data so we can  pull out the best performing
				#feature_scores[target] ={'coef': reg.coef, 'intercept': reg.intercept, 'score':reg.score(salary_test, target_test) }
				feature_scores[target] =reg.score(salary_test, target_test) 

				#write out cleaned data for checking
				predictions = reg.predict(target_train)
				#print predictions
				cleaned_training_data[target] = outlierCleaner(predictions, my_data['salary'], my_data[target] , .1)
				#print len(cleaned_training_data[target])
				
		plotDataPoints(my_data, 'salary' ,  'deferred_income', 'cleaned data')
		
		#store features by score rank
		feature_scores = OrderedDict(sorted(feature_scores.items(), key=operator.itemgetter(1), reverse=True ) )
		print "top 3 features"
		print feature_scores.items()[:6]


