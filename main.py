# numerateNathan, a toy number counting neural net
# Copyright (C) 2019  Ian L. Gore
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

from numpy import random
# exp, array, dot,

print("Hello, my name is Nathan, I like eggs and am learning to read numbers.")
print("I also like cats and talking about myself.")

# Nathan needs to take number files as input and output via one of ten neurons what number he thinks it is
# Here we define the neural net functions nate needs to read


class NateBrain(object):
    seed = 12
    input_num = 784
    hidden_neurons = 10
    output_neurons = 10

    def talk(self):
        print("I am NateBrain, my seed number is ", self.seed)
        print("I am NateBrain, my number of inputs is ", self.input_num)
        print("I am NateBrain, my number of hidden neurons is ", self.hidden_neurons)

    def __init__(self):
        random.seed(self.seed)  # User can change the seed that a nate is made with
        self.synapse1 = .1*random.random((self.input_num + 1, self.hidden_neurons)) - .05       # See 'Notes - Deriving the weights'
        self.synapse2 = .1*random.random((self.hidden_neurons + 1, self.output_neurons)) - .05  # and 'Notes - Array Size' for explanation

# NateBrain ends here


# Testing Suite
nate = NateBrain()
print("Synapse 1 data")
print(nate.synapse1)
print("Synapse 2 data")
print(nate.synapse2)

# Notes

# Deriving the weights
# We want [-.05, .05) as the initial random weight range so we use the below formula
# (b-a) * random.random((array)) + a
# ((.05)-(-.05)) * random.random((array)) + (-.05)
# .1 * random.random((array)) - .05

# Array Size
# A synapse layer needs to be an array of the total inputs from the previous layer (including bias), x
# crossed by the number of neurons in the next layer, y
#
# Layer 1 = 784 inputs + 1 bias and n hidden layer neurons -> (input_num + 1, hidden_neurons)
# Layer 2 = n hidden layer neurons + 1 bias and the number of outputs -> (hidden_neurons + 1, output_neurons)
#
