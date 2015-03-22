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

G{packagetree mldata}
"""

__docformat__ = 'epytext'

# Native imports
import cPickle, csv

# Third party imports
import numpy as np

def load_csv(path, x_dtype=np.dtype('uint8'), y_dtype=np.dtype('uint8'),
	has_header=True):
	"""
	Get the data from a dumped CSV.
	
	@param path: The full path to the CSV.
	
	@param x_dtype: The NumPy data type that corresponds to the x data.
	
	@param y_dtype: The NumPy data type that corresponds to the y data.
	
	@param has_header: Set to True if the CSV file has a header.
	
	@return: A tuple containing the data and the labels.
	"""
	
	with open(path, 'rb') as f:
		reader = csv.reader(f)
		if has_header: reader.next()
		x = []; y = []
		for row in reader:
			x.append(np.array(row[1:], dtype=x_dtype))
			y.append(row[0])
	
	return (np.array(x), np.array(y, dtype=y_dtype))

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