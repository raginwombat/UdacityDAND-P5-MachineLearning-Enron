
#!/usr/bin/python

'''
	Helper functions for final project to make the EDA less messy
'''
import matplotlib.pyplot

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

