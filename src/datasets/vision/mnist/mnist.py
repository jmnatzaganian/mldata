# mnist.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 12/20/14
#	
# Description    : Module for working with MNIST data.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Module for working with MNIST data.

G{packagetree datasets}
"""

__docformat__ = 'epytext'

# Native imports
import os, struct, gzip, random, pkgutil
from array import array

# Program imports
from datasets.base              import BaseDataset
from datasets.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class WrongMagicNumber(BaseException):
	"""
	Exception if the magic number of the file does not equal the expected
	magic number.
	"""
	
	def __init__(self, path, expected, actual):
		"""
		Initialize this class.
		
		@param path: The path to the file.
		
		@param expected: The expected magic number.
		
		@param actual: The actual magic number.
		"""
		
		self.msg = wrap_error('The file {0} had an expected magic number of '
			'{1}, but the actual magic number was {2}'.format(path, expected,
			actual))

class InvalidSelectionAmount(BaseException):
	"""
	Exception if the number of items to reduce the dataset by is too small or
	too large.
	"""
	
	def __init__(self, desired, limit, type):
		"""
		Initialize this class.
		
		@param desired: The user-desired limit amount.
		
		@param limit: The actual limit.
		
		@param type: The category, i.e. 'train' or 'test'
		"""
		
		self.msg = wrap_error('The requested number of {0} items, {1}, is '
			'invalid. The amount must be greater than 0 and less than {2}'    \
			.format(type, desired, limit))

class InvalidDimensions(BaseException):
	"""
	Exception if the number of dimensions is invalid.
	"""
	
	def __init__(self, ndims):
		"""
		Initialize this class.
		
		@param ndims: The number of dimensions
		"""
		
		self.msg = wrap_error('{0} dimensions were requested. MNIST data must '
			'be 1D or 2D only.'.format(ndims))

class InvalidCSVDimensions(BaseException):
	"""
	Exception if the number of dimensions is invalid.
	"""
	
	def __init__(self):
		"""
		Initialize this class.
		"""
		
		self.msg = wrap_error('The "dump_csv" method only supports 1D data.'
			' Use 1D data instead and try again. Alternatively, you may use '
			'the "dump_pickled" method.')

###############################################################################
########## Class Implementation
###############################################################################

class MNIST(BaseDataset):
	"""
	Class for working with MNIST data.
	"""
	
	def __init__(self, ndims=1, in_dir=None, out_dir=os.getcwd(), seed=None):
		"""
		Initializes this MNIST object.
		
		@param ndims: The number of dimensions (1 or 2)
		
		@param in_dir: Base directory for all input data.
		
		@param out_dir: Base directory for all output data.
		
		@param seed: The seed used for all random numbers.
		
		@raise InvalidDimensions: Raised if the number of dimensions is
		invalid.
		"""
		
		# Store and check the number of dimensions
		self.ndims = ndims
		if self.ndims > 2 or self.ndims < 1:
			raise InvalidDimensions(self.ndims)
		
		# Initialize the paths
		if in_dir is None:
			self.in_dir = os.path.join(pkgutil.get_loader(
				'datasets.mnist').filename, 'data')
		else:
			self.in_dir = in_dir
		self.out_dir        = out_dir
		self.train_x_path = os.path.join(self.in_dir,
			'train-images-idx3-ubyte.gz')
		self.train_y_path = os.path.join(self.in_dir,
			'train-labels-idx1-ubyte.gz')
		self.test_x_path  = os.path.join(self.in_dir,
			't10k-images-idx3-ubyte.gz')
		self.test_y_path  = os.path.join(self.in_dir,
			't10k-labels-idx1-ubyte.gz')
		
		# Objects to store data
		self.x_train = []; self.y_train = []
		self.x_test  = []; self.y_test  = []
		
		# Set the seed for all future random numbers
		random.seed(seed)
		
		# Load the data
		self.load()
		
		# Extract properties about the data
		self._get_unique_labels()
		
	def _load(self, x_path, y_path):
		"""
		Load the data into memory.
		
		@param x_path: The path to the x data (the images).
		
		@param y_path: The path to the y data (the labels).
		
		@returns: A tuple containing the x data and its corresponding labels.
		
		@raise WrongMagicNumber: Raised if a magic number mismatch occurs.
		"""
		
		# Read the image into memory
		with gzip.open(x_path, 'rb') as f:
			magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
			if magic != 2051:
				raise WrongMagicNumber(x_path, 2051, magic)
			data = array("B", f.read())
		
		if self.ndims == 1: # 1D representation
			img = [[0] * rows * cols for x in xrange(size)]
			for i in xrange(size):
				img[i] = data[i * rows * cols:(i + 1) * rows * cols]
		else:               # 2D representation
			img = [[[0] * cols] * rows for x in xrange(size)]
			for i in xrange(size):
				for r in xrange(rows):
					img[i][r] = data[i * rows * cols + r * cols:i * rows * cols
						+ r * cols + cols]
		
		# Read the labels into memory
		with gzip.open(y_path, 'rb') as f:
			magic, size = struct.unpack(">II", f.read(8))
			if magic != 2049:
				raise WrongMagicNumber(x_path, 2049, magic)
			lbl = array("B", f.read())
		
		return img, lbl
	
	def load(self):
		"""
		Loads the training and testing images and labels into memory.
		
		This method is automatically called upon creating the MNIST object.
		Any future calls to this method will reset the data to the full
		dataset.
		"""
		
		self.x_train, self.y_train = \
			self._load(self.train_x_path, self.train_y_path)
		self.x_test, self.y_test = \
			self._load(self.test_x_path, self.test_y_path)
	
	def dump_csv(self, train_file_name=None, test_file_name=None,
		make_header=True):
		"""
		Output the data to a CSV file. This is only supported if the data is
		1D.
		
		@param train_file_name: The file name to use for the training data.
		
		@param test_file_name: The file name to use for the testing data.
		
		@param make_header: Boolean denoting whether a header should be made or
		not.
		
		@raise InvalidCSVDimensions: Raised if data was attempted to be dumped
		for 2D.
		"""
		
		# Check to see if 2D data was used
		if self.ndims == 2:
			raise InvalidCSVDimensions()
		
		# Build the header
		if make_header:
			header = ['label'] + ['pixel_{0}'.format(x) for x in
					xrange(len(self.x_train[0]))]
		else:
			header = []
		
		# Initialize path names
		if train_file_name is None:
			train_path = os.path.join(self.out_dir, 'train.csv')
		else:
			train_path = os.path.join(self.out_dir, train_file_name)
		if test_file_name is None:
			test_path = os.path.join(self.out_dir, 'test.csv')
		else:
			test_path = os.path.join(self.out_dir, test_file_name)
		
		# Create the CSV files
		self._csv_dump(train_path, header, self.x_train, self.y_train)
		self._csv_dump(test_path, header, self.x_test, self.y_test)

###############################################################################
########## Example Usage
###############################################################################

def run_parse_example(out_dir):
	"""
	Run a simple example showing how to create test data.
	
	@param out_dir: The directory you wish to write the data to.
	"""
	
	in_dir = os.path.join(pkgutil.get_loader(
		'datasets.mnist').filename, 'data')
		
	# How many samples per number to use
	train_samples = 100
	
	# 1D example
	mnist = MNIST(1, in_dir, out_dir)
	mnist.reduce_dataset(train_samples, train_samples * 0.2)
	mnist.dump_csv('1d_train.csv', '1d_test.csv')
	mnist.dump_pkl('1d_mnist.pkl')
	
	# 2D example
	mnist = MNIST(2, in_dir, out_dir)
	mnist.reduce_dataset(train_samples, train_samples * 0.2)
	mnist.dump_pkl('2d_mnist.pkl')

if __name__ == "__main__":
	out_dir = os.path.join(os.getcwd(), 'mnist_data')
	try:
		os.makedirs(out_dir)
	except:
		pass
	run_parse_example(out_dir)