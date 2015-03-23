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
import cPickle, csv, zipfile, tarfile, os
from ConfigParser import SafeConfigParser, NoSectionError

# Third party imports
import numpy as np
import requests

# Program imports
from mldata                   import BASE_DIR, USER_CFG
from mldata.status_bar        import StatusBar
from mldata.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class UnsupportedArchive(BaseException):
	"""
	Exception if the archive type is unsupported.
	"""
	
	def __init__(self, path):
		"""
		Initialize this class.
		
		@param path: The full path to the archive.
		"""
		
		self.msg = wrap_error('The archvie, {0}, is unsupported. The archive'
			'must be a zip or tar file.'.format(path))

###############################################################################
########## Primary Functions
###############################################################################

def get_base_dir():
	"""
	Get the base directory for the datasets.
	
	@return: The base direcotry.
	"""
	
	config = SafeConfigParser()
	config.read(USER_CFG)
	
	try:
		return config.get('global', 'base_dir')
	except NoSectionError:
		set_base_dir(BASE_DIR)
		return BASE_DIR

def set_base_dir(base_dir):
	"""
	Set the base directory for the datasets.
	
	@param base_dir: The new base directory.
	"""
	
	config = SafeConfigParser()
	config.add_section('global')
	config.set('global', 'base_dir', base_dir)
	
	with open(USER_CFG, 'wb') as f:
		config.write(f)

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

def downloader(url, out_path, chunk_size=65535, verbose=True):
	"""
	Download the specified item.
	
	@param url: The URL to fetch the item from.
	
	@param out_path: The full path to where the file should be saved.
	
	@param chunk_size: The number of bytes to download in a single transaction.
	
	@param verbose: If True a status bar will be created to show the download
	progress.
	"""
	
	# Create a new request
	r = requests.get(url, stream=True)
	
	# Make the destination directory (if aplicable)
	try:
		os.makedirs(os.path.dirname(out_path))
	except OSError:
		pass
	
	# Create a status bar
	if verbose:
		print '\nDownloading...'
		sb = StatusBar(int(r.headers['content-length']))
	
	# Download the file
	with open(out_path, 'wb') as f:
		for chunk in r.iter_content(chunk_size):
			if chunk:
				f.write(chunk)
				f.flush()
			if verbose: sb.increment(chunk_size)
	if verbose: sb.finish()

def extractor(source_path, destination_path):
	"""
	Extract an archive.
	
	@param source_path: The full path to the source file.
	
	@param destination_path: The full path to where the archive should be
	extracted.
	
	@raise UnsupportedArchive: Raised if the archive is an unsupported type.
	"""
	
	# Check to see if the archive type is supported
	try:
		archive = zipfile.ZipFile(source_path)
	except zipfile.BadZipfile:
		try:
			archive = tarfile.open(source_path)
		except tarfile.ReadError:
			raise UnsupportedArchive(source_path)
	
	# Make the destination directory (if aplicable)
	try:
		os.makedirs(destination_path)
	except OSError:
		pass
	
	# Extract the archive
	archive.extractall(destination_path)