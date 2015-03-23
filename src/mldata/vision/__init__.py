# __init__.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 03/20/15
#
# Description    : Defines the vision package.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
This is a collection of various datasets geared towards use in Computer Vision
applications.
"""

__docformat__ = 'epytext'

# Native imports
import os

# Program imports
from mldata.util import get_base_dir

# Constant - Denotes where the base directory is
BASE_DIR = os.path.join(get_base_dir(), 'vision')