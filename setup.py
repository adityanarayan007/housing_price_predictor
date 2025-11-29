# setup.py

from setuptools import setup, find_packages

setup(
    name='houding_price_predictor', # Uses the slug from your config
    packages=find_packages(),
    version='0.1.0',
    description='A modular ML pipeline template.',
    author='Aditya',
    license='MIT',
)