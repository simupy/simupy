from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# get the version
exec(open('simupy/version.py').read())

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

long_description = long_description.replace(
    "https://simupy.readthedocs.io/en/latest/",
    "https://simupy.readthedocs.io/en/simupy-{}/".format(
        '.'.join(__version__.split('.')[:3])
    )
)

setup(
    name='simupy',
    version=__version__,
    description='A framework for modeling and simulating dynamical systems.',
    long_description=long_description,
    packages=find_packages(),
    author='Benjamin Margolis',
    author_email='ben@sixpearls.com',
    url='https://github.com/simupy/simupy',
    license="BSD 2-clause \"Simplified\" License",
    python_requires='>=3',
    install_requires=['numpy>=1.11.3', 'scipy>=0.18.1'],
    extras_require={
        'symbolic': ['sympy>=1.0'],
        'doc': ['sphinx>=1.6.3', 'sympy>=1.0'],
        'examples': ['matplotlib>=2.0', 'sympy>=1.0'],
    },

    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
)
