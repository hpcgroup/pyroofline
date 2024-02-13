# Copyright (c) 2023, Parallel Software and Systems Group, University of
# Maryland. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: MIT

from setuptools import setup
from setuptools import Extension
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Get the version in a safe way which does not refrence roofline `__init__` file
# per python docs: https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open("./roofline/version.py") as fp:
    exec(fp.read(), version)


setup(
    name="roofline",
    version=version["__version__"],
    description="A Python library for roofline analysis",
    url="https://github.com/hpcgroup/roofline",
    author="Onur Cankur",
    author_email="ocankur@umd.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="",
    packages=[
        "roofline",
    ],
    install_requires=[
        "pydot",
        "PyYAML",
        "matplotlib",
        "numpy",
        "pandas",
        "textX",
    ],
)
