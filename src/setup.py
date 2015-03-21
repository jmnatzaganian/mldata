# setup.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 10/28/14
#	
# Description    : Installs the datasets project
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

# Native imports
from distutils.core import setup
import shutil

# Install the program
setup(
	name='datasets',
	version='1.2.0',
	description="Datasets for use in machine learning",
	author='James Mnatzaganian',
	author_email='jamesmnatzaganian@outlook.com',
	url='http://techtorials.me',
	packages=['datasets', 'datasets.vision', 'datasets.vision.mnist'],
	package_data={'datasets.vision.mnist':['data/*.gz']}
	)

# Remove the unnecessary build folder
try:
	shutil.rmtree('build')
except:
	pass