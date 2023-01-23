from setuptools import setup, find_packages

setup(name='projutils', 
      version='0.1', 
      packages = find_packages(),
      test_suite='my_test_suite',
      )

import unittest
def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('projutils', pattern='test_*.py')
    return test_suite