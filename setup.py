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

requirements = [
    "wheel>=0.34",
    "numpy>=1.18",
    "typing>=3.7",
    "casadi>=3.5",
    "matplotlib>=3.2",
    "scipy>=1.4",
]

setup(
    name="mpopt",
    version="0.1.5",
    author="Devakumar THAMMISETTY, Colin Jones",
    author_email="deva.aerospace@gmail.com",
    description="A Multi-phase nonlinear Optimal control problem solver using Pseudo-spectral collocation",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mpopt",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        # "License :: OSI Approved :: GNU Lesser General Public License v3.0 (LGPLv3)",
        #'Intended Audience :: Developers, Research community across Academic and Public institutions',
        #'Topic :: Software Development :: Optimal Control',
    ],
    keywords="optimal control, multi-phase OCP, collocation, adaptive grid refinement, nonlinear optimization",
    project_urls={
        "Docmentation": "https://mpopt.readthedocs.io/",
        #'Funding': 'https://donate.pypi.org',
        #'Say Thanks!': 'http://saythanks.io/to/example',
        "Source": "https://github.com/mpopt/mpopt/",
        "Tracker": "https://github.com/mpopt/mpopt/issues",
    },
    python_requires=">=3.6",
)
