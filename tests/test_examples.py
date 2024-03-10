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

import pytest

try:
    from context import mpopt
    from mpopt import mp
except ModuleNotFoundError:
    from mpopt import mp

from examples.singlephase.dae_vdp import vdp
from examples.singlephase.hyper_sensitive import hysens
from examples.singlephase.mine_opt_wiki import mineopt
from examples.singlephase.moon_lander import moon_lander
from examples.singlephase.ocp_with_solution import ocpwithsolution
from examples.singlephase.robot_arm import robot_arm
from examples.singlephase.Betts.alpr01_alp_rider import alpr01


@pytest.mark.parametrize(
    "problem",
    [vdp, hysens, moon_lander, ocpwithsolution, robot_arm, alpr01],
)
def test_singlephase(problem):
    solution = problem.solve()
    post = problem.process_results(solution, plot=False)
    print(solution.keys())

    for key in ["f", "g", "lam_g", "lam_p", "lam_x", "x"]:
        assert key in solution

    assert post is not None
