# status_bar.py
#	
# Author         : James Mnatzaganian
# Contact        : http://techtorials.me
# Date Created   : 09/17/14
#	
# Description    : Fancy console-based status bar.
# Usage          : See "run_example()".
# Python Version : 2.7.8
#
# Adapted From   : https://github.com/tehtechguy/py_StatusBar
#
# License        : MIT License http://opensource.org/licenses/mit-license.php
# Copyright      : (c) 2015 James Mnatzaganian

"""
Fancy console-based status bar.

G{packagetree mldata}
"""

__docformat__ = 'epytext'

# Native imports
import sys

# Program imports
from mldata.exception_handler import BaseException, wrap_error

###############################################################################
########## Exception Handling
###############################################################################

class StatusBarLengthTooSmallError(BaseException):
	"""
	Exception if the bar length drops below a length of one.
	"""
	
	def __init__(self, sb):
		"""
		Initialize this class.
		
		@param sb: The status bar object to finish.
		"""
		
		sb.finish()
		self.msg = wrap_error('The bar length became too small to represent. '
			'This occurs when the percent complete reaches a very large '
			'number. Make sure that your total_length value is accurate.')

###############################################################################
########## Primary Classes
###############################################################################

class StatusBar(object):
	"""
	Class for a status bar.
	"""
		
	def __init__(self, total_length, bar_length=72, max_bar_length=72,
		min_bar_length=1, style=('[','=',']')):
		"""
		Initializes this StatusBar instance.
		
		@param total_length: The total number of steps.
		
		@param bar_length: How many characters the bar should be on the screen.
		
		@param max_bar_length: The maximum length of the bar. Set to be 79 - 7.
		(screen width for Windows compatibility, *NIX can use 80) -
		(2 closing braces + 1 space + 3 digits + 1 percent sign). Make sure the
		max_bar_length is always 7 less than the total window size.
		
		@param style: The status bar style format to use. This needs to be a
		tuple of three elements: start of the bar, the bar progress notation,
		and end of the bar.
		"""
		
		# Initializations
		self.total_length   = total_length
		self.bar_length     = bar_length
		self.max_bar_length = max_bar_length
		self.min_bar_length = min_bar_length
		self.style          = style
		self.position       = 0
		self.percent_length = 3
		
		# Ensure that the minimum bar length isn't too small
		if self.min_bar_length < 0:
			self.min_bar_length = 1
		
		# Ensure that everything can fit in a normal window
		if self.bar_length > self.max_bar_length:
			self.bar_length = max_bar_length
		
		# Ensure that the bar_length isn't too small
		if self.bar_length < self.min_bar_length:
			self.bar_length = self.min_bar_length
		
		# Ensure that the provided style is valid
		if (len(style) != 3) or (sum([len(x) for x in style]) != 3):
			self.style = ('[','=',']')
		
	def increment(self, step_size=1):
		"""
		Increments the bars position by the specified amount.
		
		@param step_size: The number to increment the bar's position by.
		
		@raise StatusBarLengthTooSmallError: Raised if the length of the status
		bar becomes too small.
		"""
		
		# Update position
		self.position += step_size
		
		# Calculate the progress
		progress         = self.position / float(self.total_length)
		percent_progress = int(progress * 100)
		percent_length   = len(str(percent_progress))
		
		# Calculate the current bar length, limiting it to the max size
		current_bar_length = min(int(progress * self.bar_length),
			self.bar_length)
		
		# Shrink bar to account for overflow scenarios
		if (current_bar_length == self.bar_length) and \
			(percent_length > self.percent_length):
			
			# If the bar length has room to grow, give it to it
			if (self.bar_length + percent_length - 3) < self.max_bar_length:
				self.bar_length     += 1
				current_bar_length  = self.bar_length
				self.percent_length = percent_length
			else:			
				self.bar_length     -= (percent_length - self.percent_length)
				current_bar_length  = self.bar_length
				self.percent_length = percent_length
				
				# Check for bar being too small
				if self.bar_length < self.min_bar_length:
					raise StatusBarLengthTooSmallError(self)
		
		# Update the status bar
		bars       = self.style[1] * current_bar_length
		bar_spaces = ' ' * (self.bar_length - current_bar_length)
		sys.stdout.write('\r{0}{1}{2}{3} {4}%'.format(self.style[0], bars,
			bar_spaces, self.style[2], percent_progress))
		sys.stdout.flush()
	
	def reset(self):
		"""
		Resets the bar's current position. This should method should be used as
		a way to reuse the bar. Do not use it unless you have first called the
		"finish" method.
		"""
		
		self.position = 0
	
	def finish(self):
		"""
		Ends the status bar, resetting the terminal to normal usage.
		"""
		
		sys.stdout.write('\n')

###############################################################################
########## Primary Functions
###############################################################################

def run_example():
	"""
	Example of various usage cases for this status bar.
	
	Note - This example will not work with Python 3, due to the print
	statements.
	"""
	
	# Native imports
	import time
	
	# Initializations
	total_length = 100                     # Total number of steps
	sb           = StatusBar(total_length) # Create the class instance
	
	print '\nThis example shows how the status bar will work under regular ' \
		'conditions'
	
	# Increment the status bar by 1 for every item
	for i in xrange(total_length):
		time.sleep(0.05)
		sb.increment()
	
	# Disable the status bar
	sb.finish()
	
	print '\nThis example shows how the status bar handles overflow ' \
		'(completeness > 100%)'
	
	# Create the class instance
	sb = StatusBar(total_length, bar_length=70)
	
	# Increment the status bar by 100 for every item
	for i in xrange(total_length):
		time.sleep(0.05)
		sb.increment(100)
	
	# Disable the status bar
	sb.finish()
	
	print '\nThis example shows what happens if the status bar gets to be ' \
		'small'
	
	# Create the class instance
	sb = StatusBar(total_length, bar_length=3)
	
	# Increment the status bar
	for i in xrange(total_length):
		time.sleep(0.05)
		sb.increment(100*i)
	
	# Disable the status bar
	sb.finish()
	
	print "\nThis example shows what happens if the status bar can't grow " \
		"anymore."
	
	# Create the class instance
	sb = StatusBar(total_length, bar_length=3, max_bar_length=5,
		min_bar_length=2)
	
	# Increment the status bar to cause fatal termination
	try:
		for i in xrange(total_length):
			time.sleep(0.05)
			sb.increment(500*i)
	except StatusBarLengthTooSmallError, e:
		print 'StatusBarLengthTooSmallError Exception Caught', e
	
	print '\nThis example shows how styling the status bar works.' \
		'\nNote that invalid styles will be ignored.'
	
	# Create the class instance
	sb = StatusBar(total_length, style=('{','-','}'))
	
	# Increment the status bar
	for i in xrange(total_length):
		time.sleep(0.05)
		sb.increment(i)
	
	# Disable the status bar
	sb.finish()

if __name__ == "__main__":
	run_example()