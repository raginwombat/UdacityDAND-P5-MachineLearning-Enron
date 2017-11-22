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




### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'restricted_stock_deferred',  'from_poi_frac', 'to_poi_frac',
	'exercised_stock_options', 'long_term_incentive' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)

### Task 2: Remove outliers
#plot outliers

#build data dict

	my_data={}
	my_data['name'] = [per[0] for per in data_dict.items()]

	for key in  data_dict[data_dict.keys()[0]].keys():
		my_data[key] = [0 if per[1][key]=='NaN' else per[1][key] for per in data_dict.items()]		
	
	
### Task 3: Create new feature(s)
#new feature will be % of messages to POI and From POI
#Using list comprehension to avoid having to initailize array  of proper length for new term
my_data['from_poi_frac'] = [0 if my_data['from_messages'][i] ==0  else (float(my_data['from_poi_to_this_person'][i]) / float(my_data['from_messages'][i])) for i in range(0, len(my_data['to_messages'])-1)]
my_data['to_poi_frac'] = [0 if my_data['to_messages'][i]==0 else (float(my_data['from_this_person_to_poi'][i])/ float(my_data['to_messages'][i]))  for i in range(0, len(my_data['to_messages'])-1)]


### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing


'''due to the previous converstion of dict -> array  I can skip feature format func, 
 I still need to  convert my data to numpy arrays and make data[0]=poi for comapitlity later
reduce data set to only the ones in the freature list and set data[0] = poi for compatiblity'''
	
data = [np.reshape( np.array(my_data['poi']).astype(np.int), (len(my_data['poi']), 1))]

#extract only features in list
for key in features_list:
	if key!='poi':
		data.append( np.reshape( np.array(my_data[key]).astype(np.int), (len(my_data[key]), 1)))
data = np.array(data)



labels, features = targetFeatureSplit(data)
#print labels
#print my_data['poi']
#print features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)


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