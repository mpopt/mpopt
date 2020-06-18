#
# Copyright (c) 2020 LA EPFL.
#
# This file is part of MPOPT 
# (see http://github.com/mpopt).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pip>=20.1.1", "wheel>=0.34.2", "numpy>=1.18.2", "typing>=3.7.4.1", "casadi>=3.5.1", "pytest>=5.4.1", "matplotlib>=3.2.1", "scipy>=1.4.1", "jupyterlab>=2.1.3"]

setup(
    name="mpopt",
    version="0.0.1",
    author="Devakumar THAMMISETTY",
    author_email="deva.aerospace@gmail.com",
    description="Multi-phase Optimal control problem solver",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mpopt/homepage/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU Lesser General Public License v3.0 (LGPLv3)",
    ],
)
