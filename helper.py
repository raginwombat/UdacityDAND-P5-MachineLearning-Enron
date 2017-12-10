
#!/usr/bin/python

'''
	Helper functions for final project to make the EDA less messy
'''
import matplotlib.pyplot
from collections import defaultdict
import numpy as np
from time import time

def printDataStats(data_dict):
	print "Check how many people are in the dataset: ", len(data_dict)
	print "check what data is available for each person: "
	print data_dict[data_dict.keys()[0]].keys()
	'''
		The data set has 146 people in total and the follwoing data poiints
		'salary'
		'to_messages'
		'deferral_payments'
		'total_payments'
		'exercised_stock_options'
		'bonus'
		'restricted_stock'
		'shared_receipt_with_poi'
		'restricted_stock_deferred'
		'total_stock_value'
		'expenses'
		'loan_advances'
		'from_messages'
		'other'
		'from_this_person_to_poi'
		'poi'
		'director_fees'
		'deferred_income'
		'long_term_incentive'
		'email_address'
		'from_poi_to_this_person'
	'''
def printNonZeroPoints(data_dict):
	featureCount =0
	nanCount =0
	for key in data_dict.keys():
		for k,v in data_dict[key].items():
			if v!='NaN':
				featureCount+=1
			else:
				nanCount+=1

	print "Total number of valid data points: ", featureCount
	print "Total number of invalid datapoints: ", nanCount

def printNumPOI(data_dict):
	poiCount=0

	for key in data_dict.keys():
		for k,v in data_dict[key].items():
			if k =='poi':
				if v==1:
					poiCount+=1

	print "Total number of POIs in the data points: ", poiCount
	
def plotDataPoints(data_dict, x_key, y_key, title=None):
	print x_key
	print y_key
	matplotlib.pyplot.scatter( data_dict[x_key], data_dict[y_key])
	matplotlib.pyplot.xlabel(x_key)
	matplotlib.pyplot.ylabel(y_key)
	if title==None:
		title = x_key+' vs '+y_key
	matplotlib.pyplot.title(title)
	#matplotlib.pyplot.show()

def cleanDict(data_dict):

	cleaned_dict=data_dict
	#iterate over people
	for k_per, v_per in cleaned_dict.items():
		#iterate over features
		for k in v_per.keys():
			if k != 'poi' and k!='email_addres':
				if v_per[k] == 'NaN':
					v_per[k] =0
		if k_per == 'TOTAL':
			del cleaned_dict['TOTAL']
	return cleaned_dict

def removeOutlier(data_dict):
	''' This method assumes all of the data is complete or will throw and index error
		This  method writes 0's to the top 
	'''

	feats = data_dict.items()[0][1].keys()
	feat_dic= {feat: {} for feat in feats}

	#Extract values from data dict
	for feat in feats:
		if feat != 'email_address' and feat!='poi':
			for per in data_dict.keys():		
				feat_dic[feat][per] =data_dict[per][feat]

	for feat in feats:
		if feat !='email_address' and feat != 'name':
			#print feat
			avg = np.mean(feat_dic[feat].values() )
			sd = np.std(feat_dic[feat].values() )

			for per in data_dict.keys():		
				if abs(data_dict[per][feat]-avg) > sd:
					#print 'removing val. Per: ', per, ' feat',feat, ' val: ',data_dict[per][feat], ' sd: ', sd
					data_dict[per][feat]= 0 

			#for k,v in sorted(feat_dic[feat].iteritems(), reverse=True, key=lambda (k,v): (v, k))[:cutoff]:
				#data_dict[k][feat]=0

	return data_dict


def classifierrun(clf, features_train, features_test, labels_train, labels_test):

	t0 = time()
	clf.fit(features_train, labels_train)
	print 'Fit took: ', round(time()-t0, 3), 's'

	t0 = time()
	pred= clf.predict(features_test)
	print 'Prediction took: ', round(time()-t0, 3), 's'

	t0 = time()
	score =clf.score(features_test, labels_test)

	print 'Scoring took: ', round(time()-t0, 3), 's'
	print 'Accuracy of : ', score
	return score
