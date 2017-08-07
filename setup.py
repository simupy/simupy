from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='simupy',
    version='0.1.0',
    description='A framework for modeling and simulating dynamical systems.',
    long_description=long_description,
    author='Benjamin Margolis',
    author_email='ben@sixpearls.com',
    url='https://github.com/sixpearls/simupy',
    install_requires=['numpy','scipy'],
    license="BSD 2-clause \"Simplified\" License",
)
