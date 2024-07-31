# -*- coding: utf-8 -*-
# Learn more: https://github.com/kennethreitz/setup.py
from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    readme = f.read()

setup(
    name="repsim",
    version="0.1.0",
    description="Benchmark for representational similarity measures",
    long_description=readme,
    packages=find_packages(exclude=("tests", "docs")),
)
