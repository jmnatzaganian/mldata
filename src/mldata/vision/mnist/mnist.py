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

G{packagetree mldata}
"""

__docformat__ = 'epytext'

# Native imports
import os, struct, gzip, pkgutil, shutil

# Third party imports
import numpy as np

# Program imports
from ..                       import BASE_DIR
from mldata.base              import BaseDataset
from mldata.exception_handler import BaseException, wrap_error

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
	
	def __init__(self, ndims=1, seed=None):
		"""
		Initializes this MNIST object. The data is automatically fetched and
		loaded.
		
		@param ndims: The number of dimensions (1 or 2).
		
		@param seed: The seed used for all random numbers.
		
		@raise InvalidDimensions: Raised if the number of dimensions is
		invalid.
		"""
		
		# Store and check the number of dimensions
		self.ndims = ndims
		if self.ndims > 2 or self.ndims < 1:
			raise InvalidDimensions(self.ndims)
		
		# Set the base paths
		self.raw_dir      = os.path.join(BASE_DIR, 'mnist', 'raw')
		self.base_dir     = os.path.join(BASE_DIR, 'mnist', 'base')
		self.user_dir     = os.path.join(BASE_DIR, 'mnist', 'user')
		self.default_set  = os.path.join(self.base_dir,
			'{0}d_base.pkl'.format(ndims))
		self.train_x_path = os.path.join(self.raw_dir,
			'train-images-idx3-ubyte.gz')
		self.train_y_path = os.path.join(self.raw_dir,
			'train-labels-idx1-ubyte.gz')
		self.test_x_path  = os.path.join(self.raw_dir,
			't10k-images-idx3-ubyte.gz')
		self.test_y_path  = os.path.join(self.raw_dir,
			't10k-labels-idx1-ubyte.gz')
		
		# Set the URLs
		self.urls = (
			'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
			'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
			'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
			'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
		)
		
		# Set the seed for all future random numbers
		self.seed = seed
	
	def fetch(self, refetch=False, verbose=True):
		"""
		Downloads and loads the data.
		
		@param refetch: If True, the dataset will be downloaded even if it
		already exists.
		"""
		
		super(MNIST, self).fetch(refetch=refetch, extract=False,
			verbose=verbose)
		self._save_base(refetch)
		self._get_unique_labels()
		
	def _load(self, x_path, y_path, ndims=1):
		"""
		Load the data into memory.
		
		@param x_path: The path to the x data (the images).
		
		@param y_path: The path to the y data (the labels).
		
		@param ndims: The number of dimensions to use (1D or 2D only!).
		
		@returns: A tuple containing the x data and its corresponding labels.
		
		@raise WrongMagicNumber: Raised if a magic number mismatch occurs.
		"""
		
		# Read the image into memory
		dt = np.dtype('uint8')
		with gzip.open(x_path, 'rb') as f:
			magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
			if magic != 2051:
				raise WrongMagicNumber(x_path, 2051, magic)
			data = np.frombuffer(f.read(), dtype=dt)
		
		if ndims == 1:
			# 1D representation
			img = np.zeros((size, rows * cols), dtype=dt)
			for i in xrange(size):
				img[i] = data[i * rows * cols:(i + 1) * rows * cols]
		else:
			# 2D representation
			img = np.zeros((size, rows, cols), dtype=dt)
			for i in xrange(size):
				for r in xrange(rows):
					img[i][r] = data[i * rows * cols + r * cols:i * rows * cols
						+ r * cols + cols]
		
		# Read the labels into memory
		with gzip.open(y_path, 'rb') as f:
			magic, size = struct.unpack(">II", f.read(8))
			if magic != 2049:
				raise WrongMagicNumber(x_path, 2049, magic)
			lbl = np.frombuffer(f.read(), dtype=dt)
		
		return img, lbl
	
	def _save_base(self, refetch=False):
		"""
		Saves the training and testing images and labels into memory.
		
		@param refetch: If True, the dataset will be downloaded even if it
		already exists.
		"""
		
		# Delete any unwanted data
		if refetch:
			try:
				shutil.rmtree(self.base_dir)
			except OSError:
				pass
		
		path_2d = os.path.join(self.base_dir, '2d_base.pkl')
		path_1d = os.path.join(self.base_dir, '1d_base.pkl')
		
		if self.ndims == 1:
			p = (path_2d, 2, path_1d, 1)
		else:
			p = (path_1d, 1, path_2d, 2)
		
		if not os.path.exists(p[0]):			
			self.x_train, self.y_train =                                      \
				self._load(self.train_x_path, self.train_y_path, p[1])
			self.x_test, self.y_test =                                        \
				self._load(self.test_x_path, self.test_y_path, p[1])
			self.dump_pkl(p[0])
			
		if not os.path.exists(p[2]):			
			self.x_train, self.y_train =                                      \
				self._load(self.train_x_path, self.train_y_path, p[3])
			self.x_test, self.y_test =                                        \
				self._load(self.test_x_path, self.test_y_path, p[3])
			self.dump_pkl(p[2])
		else:
			self.load()

	def dump_csv(self, out_dir, make_header=True):
		"""
		Output the data to CSV files. This is only supported if the data is
		1D.
		
		@param out_dir: The destination of where to save the CSVs.
		
		@param make_header: Boolean denoting whether a header should be made or
		not.
		
		@raise InvalidCSVDimensions: Raised if data was attempted to be dumped
		for 2D.
		"""
		
		try:
			os.makedirs(out_dir)
		except OSError:
			pass
		
		# Check to see if 2D data was used
		if self.ndims == 2:
			raise InvalidCSVDimensions()
		
		# Build the header
		if make_header:
			header = ['label'] + ['pixel_{0}'.format(x) for x in
					xrange(self.x_train.shape[1])]
		else:
			header = []
		
		# Initialize path names
		train_path = os.path.join(out_dir, 'train.csv')
		test_path  = os.path.join(out_dir, 'test.csv')
		
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
		
	# How many samples per number to use
	train_samples = 100
	
	# 1D example
	mnist = MNIST(1)
	mnist.fetch()
	mnist.reduce_dataset(train_samples, train_samples * 0.2)
	mnist.dump_csv(out_dir)
	mnist.dump_pkl(os.path.join(out_dir, '1d_mnist.pkl'))
	mnist.save('1d_100')
	
	# 2D example
	mnist = MNIST(2)
	mnist.load()
	mnist.reduce_dataset(train_samples, train_samples * 0.2)
	mnist.dump_pkl(os.path.join(out_dir, '2d_mnist.pkl'))
	mnist.save('2d_100')

if __name__ == "__main__":
	out_dir = os.path.join(os.getcwd(), 'mnist_data')
	try:
		os.makedirs(out_dir)
	except:
		pass
	run_parse_example(out_dir)