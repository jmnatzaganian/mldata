# mldata (Machine Learning Datasets)
## Intro
This is a collection of various datasets that can be used for machine learning
applications. More details may be found
[here](http://techtorials.me/python-machine-learning-datasets/).
## Prerequisites
[Python 2.7.X](https://www.python.org/downloads/release/python-279/) (all other
versions are untested)

[Numpy](http://www.numpy.org/)

[Requests](http://docs.python-requests.org/en/latest/)

If you are new to Python, it is recommended that you use the following
procedure to obtain the dependencies:

1) If you don't have pip (check in your Python27/Scripts folder) obtain it
from [here](https://pip.pypa.io/en/latest/installing.html).

2) Install the dependencies using pip:
pip install numpy requests

## Usage
Download this repo and inside the "src" folder execute `python setup.py
install`.

This package is platform independent, and should work on any system running
Python 2.7.X.

Click [here](http://techtorials.me/mldata/index.html) to access the API.

## Configuration
This package will save the datasets the folder ".mldata" in your home
directory. To change this, call the function "mldata.util.set_base_dir" with
your new desired path.

Note that all configuration settings are stored in the file ".mldata.cfg" in
your home directory. These are user settings that will override the global
defaults.

## Author
The original author of this code was James Mnatzaganian. For contact info, as
well as other details, see his corresponding [website](http://techtorials.me).

## Legal
This code is licensed under the [MIT license](http://opensource.org/licenses/
mit-license.php). Any included datasets may be licensed differently. Refer to
the individual dataset for more details.