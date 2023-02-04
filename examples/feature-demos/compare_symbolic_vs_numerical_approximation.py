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
"""Van der Pol oscilator OCP from https://web.casadi.org/docs/
"""
# try:
#     from mpopt import mp
# except ModuleNotFoundErro6r:
from context import mpopt
from mpopt import mp

if __name__ == "__main__":
    p = 5
    colloc = mp.Collocation([p], "LGR")
    compDn = colloc.get_composite_differentiation_matrix()
    compWn = colloc.get_composite_quadrature_weights()
    # print(compDn, compWn)

    mp.Collocation.D_MATRIX_METHOD = "symbolic"
    colloc = mp.Collocation([p], "LGR")
    compD = colloc.get_composite_differentiation_matrix()
    compW = colloc.get_composite_quadrature_weights()
    # print(compD, compW)

    # assert compD.full().all() == compDn.full().all()
    # assert compW.full().all() == compWn.full().all()

    print(
        "D error, Quad error",
        abs(compD.full() - compDn.full()).max(),
        abs(compW.full() - compWn.full()).max(),
    )

    assert abs(compD.full() - compDn.full()).max() < 1e-5
    assert abs(compW.full() - compWn.full()).max() < 1e-5
