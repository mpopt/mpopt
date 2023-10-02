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
install: requirements.txt
	test -d venv || virtualenv --python=python3.6 venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate


test: venv
	. venv/bin/activate; py.test tests


clean:
		rm -rf venv
		find -iname "*.pyc" -delete


run: examples/singlephase/moon_lander.py
	. venv/bin/activate; python examples/singlephase/moon_lander.py
