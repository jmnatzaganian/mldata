# exception_handler.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 09/27/14
#	
# Description    : Module for dealing with all errors.
# Python Version : 2.7.8
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Exception handler for this project. This module is used for every single
exception.

G{packagetree mldata}
"""

__docformat__ = 'epytext'

# Native imports
import textwrap

def wrap_error(msg):
	"""
	Wraps an error message such that it will get displayed properly.
	
	@param msg: The error to display.
	
	@return: A string containing the formatted message.
	"""
	
	return '\n  ' + '\n  '.join(textwrap.wrap(msg, 77))
	
class BaseException(Exception):
	"""
	Base class for exception handling in this program.
	"""
	
	def __str__(self):
		"""
		Allows for the exception to throw the message even if it wasn't caught.
		
		@return: The error message.
		"""
		
		return self.msg

class UnsupportedFunction(BaseException):
	"""
	Exception raised if an unsupported function is called, i.e. a user tried to
	do something he / she shouldn't be doing with that specific object.
	"""
	
	def __init__(self, class_name, function_name):
		"""
		Initialize this class.
		
		@param class_name: The class name of the caller.
		
		@param function_name: The function name of the caller.
		"""
		
		self.msg = wrap_error('The object, {0}, does not support the function '
			'{1}. Please check your usage and try again.'.format(class_name,
			function_name))