# numerateNathan, a toy number counting neural net
# Copyright (C) 2019  Ian Gore
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from nate_network import NateBrain
from time import time
from utility import demo_network, hidden_neuron_test, quarter_training_set_test, half_training_set_test


print("\nHello, my name is Nathan, I like eggs and am learning to read numbers.")
print("I also like cats and talking about myself.\n")


start_time = time()

# Hidden neuron test
# nate20 = NateBrain(20, .1, 0)
# hidden_neuron_test(nate20)
# nate50 = NateBrain(50, .1, 0)
# hidden_neuron_test(nate50)
# nate100 = NateBrain(100, .1, 0)
# hidden_neuron_test(nate100)

# Quarter and Half size training set test
# quarter_nate = NateBrain(100, .1, 0)
# quarter_training_set_test(quarter_nate)
# half_nate = NateBrain(100, .1, 0)
# half_training_set_test(half_nate)

nate = NateBrain(100, .1, 0)
demo_network(nate, 1)
end_time = time()

print(f"I completed all my tests in a total time of {end_time - start_time:5.0f} seconds.")

