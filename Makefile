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

# Colors for echos
ccend = $(shell tput sgr0)
ccbold = $(shell tput bold)
ccgreen = $(shell tput setaf 2)
ccso = $(shell tput smso)

VIRTUAL_ENV=env
PYTHON=${VIRTUAL_ENV}/bin/python3

venv: $(VIRTUAL_ENV) ## >> install virtualenv and setup the virtual environment

$(VIRTUAL_ENV):
	@echo "$(ccso)--> Install and setup virtualenv $(ccend)"
	python3 -m pip install --upgrade pip
	python3 -m pip install virtualenv
	virtualenv --python python3 $(VIRTUAL_ENV)

build: ##@main >> build the virtual environment with an ipykernel for jupyter and install requirements
		@echo ""
		@echo "$(ccso)--> Build $(ccend)"
		$(MAKE) clean
		$(MAKE) install

install: venv requirements.txt ##@main >> update requirements.txt inside the virtual environment
		@echo "$(ccso)--> Updating packages $(ccend)"
		$(PYTHON) -m pip install -r requirements.txt

test: venv
	. $(VIRTUAL_ENV)/bin/activate; py.test -v tests

clean: ## >> remove all environment and build files
		@echo ""
		@echo "$(ccso)--> Removing virtual environment $(ccend)"
		rm -rf $(VIRTUAL_ENV)

run: examples/singlephase/moon_lander.py
	. $(VIRTUAL_ENV)/bin/activate; python3 examples/singlephase/moon_lander.py
