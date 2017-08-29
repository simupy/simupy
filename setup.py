from setuptools import setup, find_packages
from codecs import open
from os import path
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

long_description = re.sub(r':doc:`([^<]+)[^`]+`', r'\1', long_description)

setup(
    name='simupy',
    version='0.1.0.dev7',
    description='A framework for modeling and simulating dynamical systems.',
    long_description=long_description,
    packages=find_packages(),
    author='Benjamin Margolis',
    author_email='ben@sixpearls.com',
    url='https://github.com/sixpearls/simupy',
    install_requires=['numpy>=1.11.3','scipy>=0.18.1'],
    license="BSD 2-clause \"Simplified\" License",
)
