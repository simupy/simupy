#!/usr/bin/env python

from distutils.core import setup

setup(
    name='SimuPy',
    version='0.1',
    description='A framework for modeling and simulating dynamical systems.',
    author='Benjamin Margolis',
    author_email='ben@sixpearls.com',
    url='https://github.com/sixpearls/SimuPy',
    packages=['simupy'],
    install_requires=[
            'numpy',
            'scipy',
            'sympy',
        ],
    license="BSD 2-clause \"Simplified\" License",
)
