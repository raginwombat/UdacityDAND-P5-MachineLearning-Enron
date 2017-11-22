#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#my adds:
import matplotlib.pyplot
from sklearn.cross_validation  import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from outlier_cleaner import outlierCleaner
import operator
from collections import OrderedDict
#Dump out 



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'loan_advances', 'bonus',
	'restricted_stock_deferred', 'deferred_income', 'from_poi_to_this_person', 
	'exercised_stock_options', 'long_term_incentive', 'from_this_person_to_poi' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)
	''' All data avail 
	print data_dict[data_dict.keys()[0]]
		{'salary': 365788,
		'to_messages': 807,
		'deferral_payments': 'NaN',
		'total_payments': 1061827,
		'exercised_stock_options': 'NaN',
		'bonus': 600000,
		'restricted_stock': 585062,
		'shared_receipt_with_poi': 702,
		'restricted_stock_deferred': 'NaN',
		'total_stock_value': 585062,
		'expenses': 94299,
		'loan_advances': 'NaN',
		'from_messages': 29,
		'other': 1740,
		'from_this_person_to_poi': 1,
		'poi': False,
		'director_fees': 'NaN',
		'deferred_income': 'NaN',
		'long_term_incentive': 'NaN',
		'email_address': 'mark.metts@enron.com',
		'from_poi_to_this_person': 38}
		'''
	

### Task 2: Remove outliers
#plot outliers

#build data dict

	my_data={}

	my_data['label_name'] = [per[0] for per in data_dict.items()]
	for key in  features_list:
		if key != 'poi':
			my_data[key] = [0 if per[1][key]=='NaN' else per[1][key] for per in data_dict.items()]		
	
	#Test plot of some data	
	matplotlib.pyplot.scatter( my_data['salary'], my_data['deferred_income'])
	matplotlib.pyplot.xlabel("salary")
	matplotlib.pyplot.ylabel("deferred_income")
	#matplotlib.pyplot.show()

	for i,rs in enumerate(my_data['restricted_stock_deferred'] ):
		if rs<0:
			#print i
			#print 'person: '+ label_name[i]
			break
	
	matplotlib.pyplot.scatter( my_data['salary'], my_data['total_payments'])
	matplotlib.pyplot.title("orig")
	matplotlib.pyplot.xlabel("salary")
	matplotlib.pyplot.ylabel("total_payments")
	#matplotlib.pyplot.show()



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
			
	#matplotlib.pyplot.scatter( my_data['salary'], cleaned_training_data['deferred_income'])
	matplotlib.pyplot.title("cleaned")
	matplotlib.pyplot.xlabel("salary")
	matplotlib.pyplot.ylabel("deferred_income")
	#matplotlib.pyplot.show()	
	#store features by score rank
	feature_scores = OrderedDict(sorted(feature_scores.items(), key=operator.itemgetter(1)))
	#print top 3 features 
	#print feature_scores.items()[:3]


### Task 3: Create new feature(s)

#new feature will be % of messages to POI and From POI

from_poi/from_messages
to_poi/to_messages

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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


'''
Cite:
https://www.wordnik.com/lists/star-wars-words

'''