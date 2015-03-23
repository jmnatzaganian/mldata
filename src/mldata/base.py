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

G{packagetree mldata}
"""

__docformat__ = 'epytext'

# Native imports
import os, cPickle, csv, shutil
from abc import ABCMeta, abstractmethod

# Third party imports
import numpy as np

# Program imports
from mldata.util import downloader, extractor, load_pkl
from mldata.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class InvalidSavedDataset(BaseException):
	"""
	Exception if the saved dataset does not exist.
	"""
	
	def __init__(self, name, datasets):
		"""
		Initialize this class.
		
		@param name: The name that was attempted to be loaded.
		
		@param datasets: A list of valid datasets.
		"""
		
		self.msg = wrap_error('The saved dataset, {0}, is invalid. The '
			'current saved datasets are {1}'.format(name, ', '.join(map(str,
				datasets))))

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
	
	@abstractmethod
	def dump_csv(self, out_dir, make_header=True):
		"""
		Output the data to CSV files. This is only supported if the data is
		1D.
		
		@param out_dir: The destination of where to save the CSVs.
		
		@param make_header: Boolean denoting whether a header should be made or
		not.
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
	
	def _get_user_saves(self):
		"""
		Returns a list of the valid user saves.
		
		@return: A list of saved datasets.
		"""
		
		return [x[:-4] for x in os.listdir(self.user_dir)]
	
	def fetch(self, refetch=False, extract=True, keep_archive=False,
		verbose=True):
		"""
		Downloads and extracts the data.
		
		@param refetch: If True, the dataset will be downloaded even if it
		already exists.
		
		@param extract: If True, the dataset will also be extracted.
		
		@param keep_archive: If True, the archive will be kept in addition to
		the extracted data. This parameter is only used if extract is True.
		
		@param verbose: If True a status bar will be created to show the
		download progress.
		"""
		
		# Delete any unwanted data
		if refetch:
			try:
				shutil.rmtree(self.raw_dir)
			except OSError:
				pass
		
		# Make the download directory, if necessary
		try:
			os.makedirs(self.raw_dir)
		except OSError:
			pass
		
		for url in self.urls:
			dl_path = os.path.join(self.raw_dir, url.split('/')[-1])
			if not os.path.exists(dl_path):
				# Download
				downloader(url, dl_path, verbose=verbose)
				
				# Extract and delete the original archive
				if extract:
					extractor(dl_path, self.raw_dir)
					if not keep_archive:
						os.remove(dl_path)
	
	def save(self, name):
		"""
		Saves the data as a custom dataset.
		
		@param name: The name to save the data as.
		"""
		
		path = os.path.join(self.user_dir, name + '.pkl')
		try:
			os.makedirs(self.user_dir)
		except OSError:
			pass
		
		self.dump_pkl(path)
	
	def load(self, name=None):
		"""
		Loads the saved data.
		
		@param name: The name of the saved data to load. If None, the default
		base set will be used.
		
		@raise InvalidSavedDataset: Raised if the saved dataset does not exist.
		"""
		
		if name is None:
			path = self.default_set
		else:
			path = os.path.join(self.user_dir, name + '.pkl')
			if not os.path.isfile(path):
				raise InvalidSavedDataset(path, self._get_user_saves())
		
		(self.x_train, self.y_train), (self.x_test, self.y_test) =            \
			load_pkl(path)
		
		# Extract properties about the data
		self._get_unique_labels()
		
	def dump_pkl(self, out_path):
		"""
		Output the data to a pickled data file. The data will be in the format
		of a list of two lists. The inner lists will contain the images and the
		labels, respectively, for the training and testing data, respectively.
		
		@param out_path: The destination of where to write the file.
		"""
		
		try:
			os.makedirs(os.path.dirname(out_path))
		except OSError:
			pass
		
		with open(out_path, 'wb') as f:
			cPickle.dump([[self.x_train, self.y_train], [self.x_test,
				self.y_test]], f, cPickle.HIGHEST_PROTOCOL)
	
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