@echo off
cd src
python setup.py install
cd ..
python C:\Python27\Scripts\epydoc.py --config=epydoc_config.txt datasets