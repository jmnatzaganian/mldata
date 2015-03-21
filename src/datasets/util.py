# util.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/15/15
#	
# Description    : Utility module.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Utility module. This module handles any sort of accessory items.

G{packagetree datasets}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, csv

def load_csv(path, lbl_type=int, data_type=int, has_header=True):
	"""
	Get the data from a dumped CSV.
	
	@param path: The full path to the CSV.
	
	@param lbl_type: Function representing what datatype the labels should be
	converted into.
	
	@param data_type: Function representing what datatype the data should be
	converted into.
	
	@param has_header: Set to True if the CSV file has a header.
	
	@return: A tuple containing the data and the labels.
	"""
	
	with open(path, 'rb') as f:
		reader = csv.reader(f)
		if has_header: reader.next()
		x = []; y = []
		for row in reader:
			x.append(lbl_type(row[0]))
			y.append([data_type(d) for d in row[1:]])
	
	return (x, y)

def load_pkl(path):
	"""
	Get the data from a dumped pickled data file.
	
	@param path: The full path to the pickled file.
	
	@return: A tuple containing the data and the labels for the training and
	test sets, i.e. (x_train, y_train), (x_test, y_test).
	"""
	
	with open(path, 'rb') as f:
		(x_train, y_train), (x_test, y_test) = cPickle.load(f)
	
	return (x_train, y_train), (x_test, y_test)