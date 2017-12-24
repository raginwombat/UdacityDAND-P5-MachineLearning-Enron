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
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from data_eda import edaPreWork
from helper import *

from sklearn.feature_selection import SelectKBest 
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics.precision_score import precision
#from sklearn.metrics.recall_score import recall

#feature seelction
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

#Un comment beofre grading
#edaPreWork()

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi', 'salary', 'total_payments', 'bonus',  'from_poi_frac', 'to_poi_frac',	'exercised_stock_options', 'long_term_incentive' ] # You will need to use more features
#features_list = ['poi', 'salary', 'bonus',   'from_poi_frac', 'to_poi_frac', 'long_term_incentive' ]
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
k=10
skb = SelectKBest(f_classif, k=k)
skb.fit(features, labels)
scores = skb.scores_

print 'Top features are: ', zip(*sorted(zip(features_list, scores), reverse=True, key=lambda x: x[1]))[0][:k]
print type (zip(*sorted(zip(features_list, scores), reverse=True, key=lambda x: x[1]))[0][:k])
features_list =  ['poi'] + list( zip(*sorted(zip(features_list, scores), reverse=True, key=lambda x: x[1]))[0][:k])

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
clfs['classifier']['Adaboost'] = AdaBoostClassifier(n_estimators=13, learning_rate=1)
clfs['classifier']['RandomForest'] = RandomForestClassifier()
clfs['classifier']['KNeighbors'] = KNeighborsClassifier()
clfs['classifier']['LogisticRegression'] = LogisticRegression()

max_score = 0.0

#Run through classfiers
for clf_key in clfs['classifier'].iterkeys():
	print '\nclassifer ', clf_key 
	clfs['result'][clf_key] = classifierrun(clfs['classifier'][clf_key],  features_train, features_test, labels_train, labels_test)

print "Max classifier:"
max_classifier_key = max(clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key

clf = clfs['classifier'][max_classifier_key]
test_classifier(clf, my_dataset, features_list, 1000)


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


scorer=  {'precision': 'precision', 'recall': 'recall'}

tune_clfs = {'classifier':{}, 'result':{}, 'params':{}}
tune_clfs['classifier']['Decision Tree'] = ( DecisionTreeClassifier(), \
							{'min_samples_leaf':range(1, 20,1), 'max_depth':range(1, 20,1) })

tune_clfs['classifier']['LinearSVC'] = ( LinearSVC(), \
						{'C':range(5,20000, 10),	})

tune_clfs['classifier']['RandomForest'] = (RandomForestClassifier(), \
						 { 'n_estimators': range(10,400,50), 'max_features': ['sqrt', 'log2'], 'min_samples_leaf':range(1, 20,1), 'max_depth':range(1, 20,1) })

tune_clfs['classifier']['AdaboostRFC'] =( AdaBoostClassifier(RandomForestClassifier(), n_estimators=13, learning_rate=1), \
						{ 'n_estimators': range(100,200, 10), 'learning_rate':np.arange(.1,.3, .1)	})

tune_clfs['classifier']['AdaboostDTC']=(AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=13, learning_rate=1), \
						{ 'n_estimators': range(100,200, 10), 'learning_rate':np.arange(.1,.3, .1)	})

tune_clfs['classifier']['Adaboost'] = (AdaBoostClassifier(n_estimators=13, learning_rate=1), \
						{ 'n_estimators': range(10,200, 10), 'learning_rate':np.arange(.1,1, .1)})

tune_clfs['classifier']['LogisticRegression'] = (LogisticRegression(), \
						{'tol':[10**-10, 10**-20], 'C': [0.05, 0.5, 1, 10, 10**2,10**5,10**10, 10**20], \
								'class_weight':['balanced']})
        

for clf_name in tune_clfs['classifier'].keys():
	print "#Tuning params for: ", clf_name
	clf_base, params = tune_clfs['classifier'][clf_name]

	clf = GridSearchCV( clf_base, param_grid=params, scoring='recall', n_jobs=4)
	
	tune_clfs['result'][clf_name] = classifierrun(clf,  features_train, features_test, labels_train, labels_test)
	print "--Best params: ", clf.best_params_	
	test_classifier(clf, my_dataset, features_list, 1000)

print "Max classifier:"
max_classifier_key = max(tune_clfs['result'].iteritems(), key=operator.itemgetter(1))[0]
print max_classifier_key
print 'Best parms:'
tune_clfs['classifier'][max_classifier_key].best_params_


t0 = time()
tuning_parameters = {'n_estimators': [20,50,100], 'min_samples_split': [1,2,4], 'max_features': [1,2,3]}
print("# Tuning hyper-parameters for recall")
RF = GridSearchCV(RandomForestClassifier(), tuning_parameters,  scoring = scorer, n_jobs=4)
RF.fit(features, labels)
print("Best parameters are:")
print(RF.best_params_)
print "tunning time: {0}".format(round(time()-t0, 3))
d



print "Parameter tuning comparision round 2"
clfs = {'classifier':{}, 'result':{}, 'params':{}}
clfs['classifier']['Decision Tree'] = DecisionTreeClassifier(min_samples_leaf= 1, criterion='gini', max_depth=3, class_weight='balanced')
clfs['classifier']['LinearSVC'] = LinearSVC( C=17385)


print 'Grid Search for AAdaboosted Decision Tree  calssifier'

params={ 'n_estimators': range(100,200, 10), 'learning_rate':np.arange(.1,.3, .1)	}

clf= GridSearchCV( AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf= 4,  max_depth=9, class_weight='balanced')), param_grid=params, scoring='average_precision', n_jobs=4)
t0 = time()
clf.fit(features_train, labels_train)
print 'Optimized  Adaboost took: ', round(time()-t0, 3), 's'
print clf.best_params_

clf = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf= 4,  max_depth=9, criterion='gini',  class_weight='balanced'), n_estimators=142, learning_rate= 0.20000000000000001)
classifierrun(clfs['classifier'][clf_key],  features_train, features_test, labels_train, labels_test)


print "test fuctions for top 2 classifiers"
print 'Adaboosed Decision Tree'
test_classifier(clf, my_dataset, features_list,1000)

#print 'SVM'
#clf = LinearSVC(kernel='sigmoid', C=5)
print 'RandomForestClassifier'
clf = RandomForestClassifier(max_features = 3, min_samples_split = 4, n_estimators= 100)
test_classifier(clf, my_dataset, features_list, 1000)


print 'SVM'
clf = LinearSVC( C=17385)
print 'RandomForestClassifier'
test_classifier(clf, my_dataset, features_list, 1000)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

