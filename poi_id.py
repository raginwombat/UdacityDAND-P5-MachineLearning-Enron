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
from time import time
from collections import OrderedDict
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from data_eda import edaPreWork
from helper import *
from sklearn.feature_selection import SelectKBest, f_classif 
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation  import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score

#Un comment beofre grading
#edaPreWork()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 
					'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 
					'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 
					'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive', 
					'from_poi_to_this_person']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
	data_dict = pickle.load(data_file)



	

### Task 2: Remove outliers
#plot outliers

#build data dict

	my_dict = cleanDict(data_dict)
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

#figure out what the top k features are that match the labels
k=18
skb = SelectKBest(f_classif, k=k)
skb.fit(features, labels)
scores = skb.scores_

#new if then statement syntax
cutoff= (k, len(features_list)-1 )[k=='all'] 
print '\n\n##Top features ',cutoff, ' are: ', zip(*sorted(zip(features_list, scores), reverse=True, key=lambda x: x[1]))[0][:cutoff]
features_list =  ['poi'] + list( zip(*sorted(zip(features_list, scores), reverse=True, key=lambda x: x[1]))[0][:cutoff])

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
clfs['classifier']['LinearSVC'] = LinearSVC()
clfs['classifier']['Adaboost'] = AdaBoostClassifier()
clfs['classifier']['RandomForest'] = RandomForestClassifier()
clfs['classifier']['KNeighbors'] = KNeighborsClassifier()
clfs['classifier']['LogisticRegression'] = LogisticRegression()

max_score = 0.0

#Run through classfiers
for clf_key in clfs['classifier'].iterkeys():
	print '\nclassifer ', clf_key 
	scores = classifierrun(clfs['classifier'][clf_key],  features_train, features_test, labels_train, labels_test)
	#Scoring is recall+precision
	clfs['result'][clf_key] = scores[0]+scores[1]

print "\n##Classifier ranking:"
print sorted(clfs['result'].iteritems(), key=lambda (k, v): (v,k), reverse=True)

print "\n\n##Max classifier:"
max_classifier_key = max(clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key


test_classifier(clfs['classifier'][max_classifier_key], my_dataset, features_list, 1000)



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


print "Hyperparameter Tuning for use in pipeline"

tune_clfs = {'classifier':{}, 'result':{}, 'params':{}, 'tuned_params':{}}

tune_clfs['classifier']['DecisionTreeClassifier'] = ( DecisionTreeClassifier(), \
							{'classifier__min_samples_leaf':range(1, 20,1), 'classifier__max_depth':range(1, 20,1) })

#tune_clfs['classifier']['LinearSVC'] = ( LinearSVC(), {'classifier__C':range(5,20000, 100),	})

tune_clfs['classifier']['RandomForest'] = (RandomForestClassifier(), \
						 { 'classifier__n_estimators':range(10,400,100), 'classifier__min_samples_leaf':range(1, 5,1), 'classifier__max_depth':range(1, 5,1) })

					
#tune_clfs['classifier']['LogisticRegression'] = (LogisticRegression(), {'classifier__tol':[.1, 10**-5, 10**-10, 10**-20], 'classifier__C': [0.05, 0.1, 0.5, 0.55, 1, 10, 10**2,10**5,10**10, 10**20] })

tune_clfs['classifier']['Adaboost'] =( AdaBoostClassifier(n_estimators=13, learning_rate=1), \
						{ 'classifier__n_estimators': range(100,200, 10), 'classifier__learning_rate':np.arange(.1,.3, .1)	})

#tune_clfs['classifier']['KNeighborsClassifier'] = (KNeighborsClassifier(), {'classifier__n_neighbors':range(1,10,1), 'classifier__leaf_size': range(1,100,10)})


cv = StratifiedShuffleSplit(labels, n_iter=10, test_size=0.33, random_state=42)
#cv = StratifiedShuffleSplit(labels, 1000, random_state=42)
scaler = StandardScaler()
pca = PCA()
#clf = AdaBoostClassifier(RandomForestClassifier(), n_estimators=150, learning_rate=0.20000000000000001)


for clf_name in tune_clfs['classifier'].keys():
	print '\n #Tuning params for: ', clf_name
	clf_base, params = tune_clfs['classifier'][clf_name]
	#clf_base =  LinearSVC() #tune_clfs['classifier'][clf_name]
	pipe = Pipeline(steps=[('scale',scaler), ('pca',pca), ('classifier', clf_base)])
	params['pca__n_components']= range(1,len(features_list), 1)
	#print "params: ", params
	grid = GridSearchCV( pipe, param_grid=params,  n_jobs=4, cv=cv, scoring = 'recall', refit=True)
	
	#print "param keys:"
	#print grid.get_params().keys()
	print " --#Fitting"
	grid.fit(features, labels)
	clf = grid.best_estimator_
	pred= clf.predict(features_test)
	score  =  precision_recall_fscore_support(labels_test, pred, average='binary')# classifierrun(clf,  features_train, features_test, labels_train, labels_test)
	tune_clfs['result'][clf_name] = score[0] +score[1]
	tune_clfs['tuned_params'][clf_name] = grid.best_params_
	print "--#Best params: ", grid.best_params_	
	#test_classifier(clf, my_dataset, features_list, 1000)
	tune_clfs['classifier'][clf_name] = clf
	
	print " --#Precsision recall fscore support"
	print precision_recall_fscore_support(labels_test, pred, average='binary')
	

print "\n##Classifier ranking:"
print sorted(clfs['result'].iteritems(), key=lambda (k, v): (v,k), reverse=True)


print "\n\n##Max classifier:"
max_classifier_key = max(tune_clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key
print '#Best parms:', tune_clfs['tuned_params'][max_classifier_key]

test_classifier(tune_clfs['classifier'][max_classifier_key], my_dataset, features_list, 1000)

print "\n\n##2nd Max Classifier"
clf_name = sorted(tune_clfs['result'].iteritems(), key=lambda (k, v): (v,k), reverse=True)[1][0]

test_classifier(tune_clfs['classifier'][clf_name], my_dataset, features_list, 1000)

clf = Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), \
				('pca', PCA(copy=True, n_components=18, whiten=False)), \
				('classifier', RandomForestClassifier(bootstrap=True, class_weight=None, \
					criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=None,
          			n_jobs=4, oob_score=False, random_state=None, verbose=0, warm_start=False))])

''' 
max calssifer = Pipeline(steps=[('scale', StandardScaler(copy=True, with_mean=True, with_std=True)), \
				('pca', PCA(copy=True, n_components=14, whiten=False)),  \
				('classifier', LinearSVC(C=5, class_weight=None, dual=True, fit_intercept=True, \
     				intercept_scaling=1, loss='squared_hinge', max_iter=1000, \
     				multi_class='ovr', penalty='l2', random_state=None, tol=0.0001, \
     				verbose=0))])
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

