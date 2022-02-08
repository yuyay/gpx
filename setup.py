#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os

from setuptools import setup, find_packages

try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''

def _requires_from_file(filename):
    return open(filename).read().splitlines()

# version
here = os.path.dirname(os.path.abspath(__file__))
version = '0.1'

setup(
    name="pytorch-gpx",
    version=version,
    url='https://github.com/yuyay/gpx',
    author='Yuya Yoshikawa',
    author_email='yoshikawa@stair.center',
    maintainer='Yuya Yoshikawa',
    maintainer_email='yoshikawa@stair.center',
    description='An official implementation of "Gaussian Process Regression With Interpretable Sample-Wise Feature Weights"',
    # long_description=readme,  #TODO: convert to reST format
    packages=find_packages(
        exclude=('examples', 'docs', '.vscode', 'docker', 'scripts', 'experiments', 'data', 'notebooks')),
    python_requires='>=3.7',
    install_requires=_requires_from_file('requirements.txt'),
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # entry_points="""
    #   # -*- Entry points: -*-
    #   [console_scripts]
    #   pkgdep = pypipkg.scripts.command:main
    # """,
)

# from setuptools import setup, find_packages

# setup(
#     name="gpx",
#     version="0.9",
#     install_requires=[
#         "numpy", "scipy", "torch", "matplotlib", "scikit-learn",
#         "jupyter", "ipython", "gpytorch"],
#     packages=find_packages(
#         exclude=('examples', 'docs', '.vscode', 'docker', 'scripts', 'experiments', 'data')),
# )
