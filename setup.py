# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='qupy',
    version='0.1.0',
    description='QuPy: A quantum circuit simulator for both CPU and GPU',
    long_description=readme,
    author='Ken Nakanishi',
    author_email='ikyhn1.ken.n@gmail.com',
    install_requires=['numpy'],
    url='https://github.com/ken-nakanishi/qupy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
