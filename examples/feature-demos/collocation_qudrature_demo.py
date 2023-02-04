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
# try:
#     from mpopt import mp
# except ModuleNotFoundError:
from context import mpopt
from mpopt import mp

# Create a collocation object with polynomial of order degree
degree = 2
colloc = mp.Collocation([degree], "LGR")

# Get the roots of the polys and polynomials as casadi SX objects
taus = colloc.roots[degree]
polys = colloc.polys[degree]

# Compute Diff matrix corresponding to the roots (obtained with degree of the polynominal as key)
D = colloc.get_diff_matrix(degree)
w = colloc.get_quadrature_weights(degree)

# Compute D' to get D'X for derivative at tau' which are different from original roots
Dp = colloc.get_diff_matrix(degree, taus=[0, 1])

# Compute quad matrix between two specified limits of the integral (0, 1 is full domain integral. We compute now between (0.5 and 0.6))
w = colloc.get_quadrature_weights(degree)
w1 = colloc.get_quadrature_weights(degree, tau0=0.5, tau1=0.6)
assert sum((w.full()) - 1) < 1e-6
assert sum((w1.full()) - 0.1) < 1e-6

if __name__ == "__main__":
    print(taus, w)
    print(sum(w.full()), sum(w1.full()))

    print(D, Dp)
