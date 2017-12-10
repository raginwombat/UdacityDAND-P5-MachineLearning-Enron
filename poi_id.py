#!/usr/bin/python

import sys
import pickle
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

#my adds:
import matplotlib.pyplot
from sklearn.cross_validation  import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import operator
from collections import OrderedDict
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
from data_eda import edaPreWork
from helper import *

#Un comment beofre grading
#edaPreWork()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', 'bonus', 'exercised_stock_options',  'from_poi_frac', 'to_poi_frac',	'exercised_stock_options', 'long_term_incentive' ] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)


### Task 2: Remove outliers
#plot outliers

#build data dict

	my_dict = cleanDict(data_dict)
	'''
	my_data['name'] = [per[0] for per in data_dict.items()]

	for key in  data_dict[data_dict.keys()[0]].keys():
		my_data[key] = [0 if per[1][key]=='NaN' else per[1][key] for per in data_dict.items()]		

	#identify outliers and remove them by forcing to 0. can't pop them without screwin up the index:
	for key in my_data.iterkeys():
		cutoff = int(round(len(my_data[key]) *.05))
		#print  "removing top ", cutoff,"outliers for ", key
		if key!='poi':if val>sd then data_dict[val.key()][feat] =0
			for v in  sorted(my_data[key], reverse=True)[:cutoff]:
				#print my_data[key].index(v)
				my_data[key][my_data[key].index(v) ] = 0
	'''	
	my_data = removeOutlier(my_dict)
		
### Task 3: Create new feature(s)
#new feature will be % of messages to POI and From POI
#Using list comprehension to avoid having to initailize array  of proper length for new term
for per in my_data.keys():

	if 	my_data[per]['from_messages'] !=0:
		my_data[per]['from_poi_frac'] = float(my_data[per]['from_poi_to_this_person']) / float(my_data[per]['from_messages'])

	else:
		my_data[per]['from_poi_frac'] = 0

for per in my_data.keys():

	if 	my_data[per]['to_messages'] !=0:
		my_data[per]['to_poi_frac'] = float(my_data[per]['from_this_person_to_poi']) / float(my_data[per]['to_messages'])

	else:
		my_data[per]['to_poi_frac'] = 0

### Store to my_dataset for easy export below.
my_dataset = my_data

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size=0.4, random_state=42)


#create dict for diff classifiers
clfs = {'classifier':{}, 'result':{}, 'params':{}}
clfs['classifier']['Niaeve Bayes'] = GaussianNB()
clfs['classifier']['Decision Tree'] = DecisionTreeClassifier()
clfs['classifier']['SVC'] = SVC(kernel='rbf', C=10000)
clfs['classifier']['Adaboost'] = AdaBoostClassifier(n_estimators=13, learning_rate=1)
clfs['classifier']['RandomForest'] = RandomForestClassifier()
clfs['classifier']['KNeighbors'] = KNeighborsClassifier()
max_score = 0.0

for clf_key in clfs['classifier'].iterkeys():
	print clf_key
	t0 = time()
	clfs['classifier'][clf_key].fit(features_train, labels_train)
	
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
clf = clfs['classifier'][max_classifier_key]
test_classifier(clf, my_dataset, features_list, 1000)

test_classifier(clfs['classifier']['Decision Tree'], my_dataset, features_list,1000)
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


params = {'min_samples_leaf':range(1, 20,1), 'max_depth':range(1, 20,1) }#,  'class_weight':['balanced', 'none'], 'criterion':['gini', 'entropy']}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid=params, scoring='accuracy', cv=10, n_jobs=4)
t0 = time()
clf.fit(features_train, labels_train)
print 'Grid  Decision Tree search took: ', round(time()-t0, 3), 's'
print clf.best_params_

params={}
params={ 'kernel': ['rbf', 'sigmoid'], 'C':range(5,20000, 10),	}
features = preprocessing.scale(features)
clf= GridSearchCV(SVC(), param_grid=params, scoring='f1', n_jobs=4)
t0 = time()
clf.fit(features_train, labels_train)
print 'Grid search SVM took: ', round(time()-t0, 3), 's'
print clf.best_params_

params={}
param_grid = { 'n_estimators': range(100,800,50), 'max_features': ['sqrt', 'log2'], 'min_samples_leaf':range(1, 20,1), 'max_depth':range(1, 20,1) }
clf= GridSearchCV(RandomForestClassifier(), param_grid=params, scoring='accuracy', n_jobs=4)
t0 = time()
clf.fit(features_train, labels_train)
print 'Grid search RFC took: ', round(time()-t0, 3), 's'
print clf.best_params_



print "Parameter tuning comparision round 2"
clfs = {'classifier':{}, 'result':{}, 'params':{}}
clfs['classifier']['Decision Tree'] = DecisionTreeClassifier(min_samples_leaf= 1, criterion='gini', max_depth=3, class_weight='balanced')
clfs['classifier']['SVC'] = SVC(kernel='sigmoid', C=5)


#Run through classfiers
for clf_key in clfs['classifier'].iterkeys():
	print 'classifer ', clf_key
	clfs['result'][clf_key] = classifierrun(clfs['classifier'][clf_key],  features_train, features_test, labels_train, labels_test)

print "Max classifier:"
max_classifier_key = max(clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key

print 'Grid Search for AAdaboosted Decision Tree  calssifier'
#params={ 'n_estimators': range(1,600, 1), 'learning_rate':np.arange(.1,1, .1)	}
params={ 'n_estimators': range(100,200, 10), 'learning_rate':np.arange(.1,.3, .1)	}
#clf= GridSearchCV( AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf= 17, criterion='gini', max_depth=2, class_weight='balanced')), param_grid=params, scoring='f1', n_jobs=4)
clf= GridSearchCV( AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf= 4,  max_depth=9, class_weight='balanced')), param_grid=params, scoring='accuracy', n_jobs=4)
t0 = time()
clf.fit(features_train, labels_train)
print 'Optimized  Adaboost took: ', round(time()-t0, 3), 's'
print clf.best_params_

clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf= 17, criterion='gini', max_depth=2, class_weight='balanced'), n_estimators=142, learning_rate= 0.20000000000000001)
classifierrun(clfs['classifier'][clf_key],  features_train, features_test, labels_train, labels_test)


print "test fuctions for top 2 classifiers"
print 'Adaboosed Decision Tree'
test_classifier(clf, my_dataset, features_list,1000)

print 'SVM'
clf = SVC(kernel='sigmoid', C=5)
test_classifier(clf, my_dataset, features_list, 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

