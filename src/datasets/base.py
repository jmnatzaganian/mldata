# base.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/17/15
#	
# Description    : Base class for datasets.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Base class for datasets.

G{packagetree datasets}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle, csv
from abc import ABCMeta, abstractmethod

# Third party imports
import numpy as np

###############################################################################
########## Class Implementation
###############################################################################

class BaseDataset(object):
	"""
	Base class description for a dataset.
	"""
	__metaclass__ = ABCMeta
	
	@abstractmethod
	def __init__(self, in_dir=None, out_dir=os.getcwd(), seed=None):
		"""
		Initializes the class instance. This must at the very least create a
		seed parameter. The seed will be used for all random numbers. This
		method should call the "load" function.
		
		@param in_dir: Base directory for all input data.
		
		@param out_dir: Base directory for all output data.
		
		@param seed: The seed used for all random numbers.
		"""		
	
	def _csv_dump(self, path, header, x, y=None, iters=1):
		"""
		Output the data to a CSV, with or without labels.
		
		@param path: The path to create the file.
		
		@param header: A sequence containing the headers.
		
		@param x: The x data.
		
		@param y: The y data.
		
		@param iters: The number of times to duplicate the data. If this value
		is larger than one, using this data set will simulate running multiple
		epochs.
		"""
		
		with open(path, 'wb') as f:
			writer = csv.writer(f)
			if len(header) > 0:
				writer.writerow(header)
			for iter in xrange(iters):
				for i, item in enumerate(x):
					if y is not None:
						writer.writerow([y[i]] + list(item))
					else:
						writer.writerow(item)
	
	def _get_unique_labels(self):
		"""
		Creates a set of unique labels as well as a distribution of the count
		of each label type. It assumes that all labels are shared between
		training and test sets, i.e. there is at least one occurrence of each
		label in each dataset.
		"""
		
		# Find unique labels
		self.unique_labels = set(self.y_train)
		self.num_labels    = len(self.unique_labels)
		
		# Find label distribution
		self.label_train_count = {lbl:0 for lbl in self.unique_labels}
		self.label_test_count  = {lbl:0 for lbl in self.unique_labels}
		for lbl in self.y_train:
			self.label_train_count[lbl] += 1
		for lbl in self.y_test:
			self.label_test_count[lbl] += 1
		self.min_train_count = min(self.label_train_count.values())
		self.min_test_count  = min(self.label_test_count.values())
	
	@abstractmethod
	def dump_csv(self, train_file_name=None, test_file_name=None,
		make_header=True):
		"""
		Output the data to a CSV file.
		
		@param train_file_name: The file name to use for the training data.
		
		@param test_file_name: The file name to use for the testing data.
		
		@param make_header: Boolean denoting whether a header should be made or
		not.
		"""
	
	def dump_pkl(self, file_name=None):
		"""
		Output the data to a pickled data file. The data will be in the format
		of a list of two lists. The inner lists will contain the images and the
		labels, respectively, for the training and testing data, respectively.
		
		@param file_name: The file name to use.
		"""
		
		# Initialize the path
		if file_name is None:
			path = os.path.join(self.out_dir, 'dataset.pkl')
		else:
			path = os.path.join(self.out_dir, file_name)
		
		with open(path, 'wb') as f:
			cPickle.dump([[self.x_train, self.y_train], [self.x_test,
				self.y_test]], f, cPickle.HIGHEST_PROTOCOL)
	
	@abstractmethod
	def load(self):
		"""
		Loads the data into memory. This must create four class variables:
		
		x_train - The training data.
		y_train  - The training labels.
		x_test  - The testing data.
		y_test  - The testing labels.
		
		This class should additionally call the "_get_unique_labels" function.
		"""
	
	def shuffle(self):
		"""
		Randomly shuffles the training and test data.
		"""
		
		# Shuffle the training sets
		np.random.seed(self.seed)
		np.random.shuffle(self.x_train)
		np.random.seed(self.seed)
		np.random.shuffle(self.y_train)
		
		# Shuffle the test sets
		seed = np.random.get_state()[1][0]
		np.random.shuffle(self.x_test)
		np.random.seed(seed)
		np.random.shuffle(self.y_test)
	
	def reduce_dataset(self, n_train, n_test, normalize=True):
		"""
		Reduce the dataset down to specified amount. If normalize is True an
		even distribution of all labels in the dataset will be maintained. If
		the number of elements is set to 0, the dataset will be deleted.
		
		@param n_train: The number of training samples to use.
		
		@param n_test: The number of test samples to use.
		
		@param normalize: If True, the number of items represented by the input
		will be the count of the number of each occurrence of the label. In
		other words, there will be n_train * self.num_labels datapoints for the
		training set and n_test * self.num_labels datapoints for the testing
		set. In this case, there will be an equal number of occurrences of each
		label. If False, the number of items will be used to represent the
		total number of items to extract. In this case, there may be an unequal
		distribution of the labels.
		
		@raise InvalidSelectionAmount: Raised if the number of desired items to
		select is larger than the total number of available items.
		"""
		
		# Ensure that the inputs are integers
		n_train = int(n_train)
		n_test  = int(n_test)
		
		# Test to see if valid parameters were provided
		if normalize:
			if n_train > self.min_train_count or n_train < 0:
				raise InvalidSelectionAmount(n_train, self.min_train_count,
					'train')
			if n_test > self.min_test_count or n_test < 0:
				raise InvalidSelectionAmount(n_test, self.min_test_count,
					'test')
		else:
			if n_train > len(self.x_train) or n_train < 0:
				raise InvalidSelectionAmount(n_train, len(self.x_train),
					'train')
			if n_test > len(self.x_test) or n_test < 0:
				raise InvalidSelectionAmount(n_test, len(self.x_test),
					'test')
		
		def reduce_set(x, y, limit):
			"""
			Reduces the provided data set, maintaining an even distribution of
			each label.
			
			@param x: The data.
			
			@param y: The label data.
			
			@param limit: The number of items of each type to select.
			
			@returns: A tuple of the new data and its corresponding labels.
			"""
			
			# Keep track of label occurrences
			count  = {lbl:0 for lbl in self.unique_labels}
			s_x    = list(x.shape); s_y = list(y.shape)
			s_x[0] = limit * self.num_labels; s_y[0] = limit * self.num_labels
			o_x    = np.zeros(s_x, x.dtype); o_y = np.zeros(s_y, y.dtype)
			
			# Pull out data
			tot = 0
			for i, lbl in enumerate(y):
				c = count[lbl]
				if c < limit:
					o_x[tot]   = x[i]
					o_y[tot]   = lbl
					count[lbl] += 1
					tot        += 1
			
			return o_x, o_y
		
		# Only add data where we are selecting at least one point
		if n_train > 0:
			if normalize:
				self.x_train, self.y_train = reduce_set(self.x_train,
					self.y_train, n_train)
			else:
				self.x_train = self.x_train[:n_train]
				self.y_train = self.y_train[:n_train]
		else:
			self.x_train = np.array([]); self.y_train = np.array([])
		
		if n_test > 0:
			if normalize:
				self.x_test, self.y_test = reduce_set(self.x_test, self.y_test,
					n_test)
			else:
				self.x_train = self.x_train[:n_train]
				self.y_train = self.y_train[:n_train]
		else:
			self.x_test = np.array([]); self.y_test = np.array([])