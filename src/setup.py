# setup.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 10/28/14
#	
# Description    : Installs the mldata project
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

# Native imports
from distutils.core import setup
import shutil

# Install the program
setup(
	name='mldata',
	version='2.0.0',
	description="Machine Learning Datasets",
	author='James Mnatzaganian',
	author_email='jamesmnatzaganian@outlook.com',
	url='http://techtorials.me',
	packages=['mldata', 'mldata.vision', 'mldata.vision.mnist']
	)

# Remove the unnecessary build folder
try:
	shutil.rmtree('build')
except:
	pass