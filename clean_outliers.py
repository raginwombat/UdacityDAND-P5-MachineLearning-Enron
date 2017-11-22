#!/usr/bin/python
# working out features.
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#my adds:
import matplotlib.pyplot

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

	for k,v in data_dict.items():
		print k, v


for k,v in data_dict.items():
	#rint max_out
	if v['bonus'] > max_out['bonus'] and v['bonus']!= 'NaN':
		#print k, 'makes ', v['bonus'], 'in bonus'
		max_out =v
		max_out['poi'] = k

		
### Task 2: Remove outliers

#plot outliers
'''
	for k in data_dict.keys():
	poi = [datpt[0] for datpt in data_dict]
	salary= [datpt[1] for datpt in data_dict]
	total_payments= [datpt[2] for datpt in data_dict]
	loan_advances=[datpt[3] for datpt in data_dict]
	bonus= [datpt[4] for datpt in data_dict]
	restricted_stock_deferred = [datpt[5] for datpt in data_dict]
	deferred_income= [datpt[6] for datpt in data_dict]
	from_poi_to_this_person= [datpt[7] for datpt in data_dict]
	exercised_stock_options= [datpt[8] for datpt in data_dict]
	long_term_incentive= [datpt[9] for datpt in data_dict]
	from_this_person_to_poi = [datpt[10] for datpt in data_dict]
		


	for per in data_dict:
		poi = per[0], 
		salary= per[1], 
		total_payments= per[2], 
		loan_advances= per[3], 
		bonus= per[4], 
		restricted_stock_deferred= per[5], 
		deferred_income= per[6], 
		from_poi_to_this_person= per[7], 
		exercised_stock_options= per[8], 
		long_term_incentive= per[9], 
		from_this_person_to_poi = per[10]
		matplotlib.pyplot.scatter(salary, total_payments)
	'''
	#matplotlib.pyplot.show()


### Task 3: Create new feature(s)
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