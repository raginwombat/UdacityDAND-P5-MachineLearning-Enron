#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

#my adds:
import matplotlib.pyplot
from sklearn.cross_validation  import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import operator
from collections import OrderedDict
from sklearn.ensemble import AdaBoostClassifier
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'restricted_stock_deferred',  'from_poi_frac', 'to_poi_frac',	'exercised_stock_options', 'long_term_incentive' ] # You will need to use more features

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

	#identify outliers and remove them by forcing to 0. can't pop them without screwin up the index:
	for key in my_data.iterkeys():
		cutoff = int(round(len(my_data[key]) *.05))
		print  "removing top ", cutoff,"outliers for ", key
		if key!='poi':
			for v in  sorted(my_data[key], reverse=True)[:cutoff]:
				#print my_data[key].index(v)
				my_data[key][my_data[key].index(v) ] = 0
		
		
### Task 3: Create new feature(s)
#new feature will be % of messages to POI and From POI
#Using list comprehension to avoid having to initailize array  of proper length for new term
my_data['from_poi_frac'] = [0 if my_data['from_messages'][i] ==0  else (float(my_data['from_poi_to_this_person'][i]) / float(my_data['from_messages'][i])) for i in range(0, len(my_data['to_messages']))]
my_data['to_poi_frac'] = [0 if my_data['to_messages'][i]==0 else (float(my_data['from_this_person_to_poi'][i])/ float(my_data['to_messages'][i]))  for i in range(0, len(my_data['to_messages']))]


### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing


'''due to the previous converstion of dict -> array  I can skip feature format func, 
 I still need to  convert my data to numpy arrays and make data[0]=poi for comapitlity later
reduce data set to only the ones in the freature list and set data[0] = poi for compatiblity'''

data = [ np.array(my_data['poi']) ]

#extract only features in list
for key in features_list:
	if key!='poi':
		data.append( np.array(my_data[key]).astype(np.int))
	if key == 'to_poi_frac' or key == 'from_poi_frac':
		data.append( np.array(my_data[key]).astype(np.float))



labels, features = targetFeatureSplit(data)


labels = np.array(my_data['poi']) 
features = []
for key in features_list:
	if key!='poi':
		features.append( np.array(my_data[key]).astype(np.int))
	if key == 'to_poi_frac' or key == 'from_poi_frac':
		features.append( np.array(my_data[key]).astype(np.float))


#Currentl shape of features is teatures == rows and people == columns need to transpose for compatiblity
features = np.transpose(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)

#create dict for diff classifiers
clfs = {'classifier':{}, 'result':{}, 'params':{}}
clfs['classifier']['Niaeve Bayes'] = GaussianNB()
clfs['classifier']['Ada Boost'] = AdaBoostClassifier(n_estimators =13, learning_rate =1)
clfs['classifier']['Decision Tree'] = DecisionTreeClassifier(criterion='entropy',min_samples_split=40)
clfs['classifier']['SVC'] = SVC(C= 10065 , gamma = 'auto', kernel='rbf')
max_score = 0.0
'''

clfs['params']['Niaeve Bayes'] = {}
clfs['params']['Ada Boost'] = 
clfs['params']['Decision Tree'] = 
'''
for clf_key in clfs['classifier'].iterkeys():
	t0 = time()
	clfs['classifier'][clf_key].fit(features_train, labels_train)
	print clf_key
	print 'Fit took: ', round(time()-t0, 3), 's'

	t0 = time()
	pred= clfs['classifier'][clf_key].predict(features_test)
	print 'Prediction took: ', round(time()-t0, 3), 's'

	t0 = time()
	score = clfs['classifier'][clf_key].score(features_test, labels_test)

	print 'Scoring took: ', round(time()-t0, 3), 's'
	print 'Accuracy of : ', score
	clfs['result'][clf_key]=score

print "Max classifier:"
max_classifier_key = max(clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key
print type(clfs['classifier'][max_classifier_key])

#test_classifier(clfs['classifier'][max_classifier_key], data, features_list, folds = 1000)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.3, random_state=42)


params={ 'kernel': ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'), 'C':range(1,20000, 1000)	}

#clf= SVC(C= 10065 , gamma = 'auto', kernel='rbf')	
clf = GridSearchCV(SVC(), param_grid=params, scoring='f1', cv=10)
t0 = time()
clf.fit(features_train, labels_train)
print 'Grid search took: ', round(time()-t0, 3), 's'
print clf.best_params_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


'''
Cite:
https://www.wordnik.com/lists/star-wars-words

'''
