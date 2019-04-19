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

# import numpy as np

print("Hello, my name is Nathan, I like eggs and am learning to read numbers.")
print("I also like cats.")

# Nathan needs to take number files as input and output via one of ten neurons what number he thinks it is
# Here we define the neural net functions nate needs to read


class NateBrain:
    num = 12

    def talk(self):
        print("I am talking, my num is ", self.num)

# NateBrain ends here


nate = NateBrain()
nate.talk()
NateBrain.num = 20  # Set class var instance to 20
nate.talk()
nate.num = 25  # set instance var to 20
nate.talk()

print("The numbers printed are 12, 20, and 25 in that order")
