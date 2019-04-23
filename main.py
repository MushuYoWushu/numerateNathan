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

from numpy import random, exp, fromstring, dot, array, mean, arange, zeros_like, insert
from time import time


print("\nHello, my name is Nathan, I like eggs and am learning to read numbers.")
print("I also like cats and talking about myself.\n")

# Nathan needs to take number files as input and output via one of ten neurons what number he thinks it is
# Here we define the neural net functions nate needs to read


class NateBrain(object):
    seed = 12
    input_num = 784
    hidden_neurons = 10
    output_neurons = 10
    learning_rate = .1

    def talk(self):
        print("I am NateBrain, my seed number is", self.seed)
        print("I am NateBrain, my number of input neurons is", self.input_num)
        print("I am NateBrain, my number of hidden neurons is", self.hidden_neurons)
        print("I am NateBrain, my number of output neurons is", self.output_neurons)
        print(f'I am NateBrain, my learning rate is {self.learning_rate}\n')

    def __init__(self):
        random.seed(self.seed)  # User can change the seed that a nate is made with
        self.synapse1 = array(.1*random.random((self.input_num + 1, self.hidden_neurons)) - .05)   # See 'Notes'
        self.synapse2 = array(.1*random.random((self.hidden_neurons + 1, self.output_neurons)) - .05)  # for explanation

    def sigmoid(self, x):  # We do not change the class contents so the IDE may suggest making static
        return 1 / (1 + exp(-x))

    def sigmoid_deriv(self, x):  # We do not change the class contents so the IDE may suggest making static
        return x * (1-x)    # Note this is equivalent to pow((1 + exp(-x)), 2)

    def label2output(self, num_label):  # This translates an int to an array with a 1 in the correct index
        output = zeros_like(arange(self.output_neurons))  # makes a 0'd array of output_neurons size
        output[num_label] = 1  # sets the entry of the corresponding digit to 1
        return output

    def think(self, csv_file_path, learn=False):  # Trains based on passed data location
        with open(csv_file_path) as imgs:
            counter = 0  # iteration counter for error
            for number_img in imgs:  # Here we are going to process each image piece by piece
                image = fromstring(number_img, dtype=int, sep=',')
                output = self.label2output(image[0])  # image[0] is the data label, this is an array translation.
                number_data = image  # bias neuron and image data
                number_data[0] = 1  # set bias to 1, we already have an array of 785 so no sweat there

                l0 = number_data  # this series of dots and sigmoids represent the transfer from start to end
                l1 = self.sigmoid(dot(l0, self.synapse1))  # between neurons, through synapse1
                l1 = insert(l1, 0, 1)  # we need to add in the bias unit set to 1 at the beginning!
                l2 = self.sigmoid(dot(l1, self.synapse2))

                l2_err = output - l2

                if (counter % 10000) == 0:  # Periodically output error average
                    accuracy = 100. * (1 - mean(abs(l2_err)))
                    print(f'Accuracy: {100. * (1 - mean(abs(l2_err))):5.2f} %')

                l2_delta = self.sigmoid_deriv(l2) * l2_err  # this tells us the amount weight gotta change on l2
                l1_err = dot(l2_delta, self.synapse2.T)  # backprop step that shows how much l1 contributed to error
                l1_delta = self.sigmoid_deriv(l1) * l1_err  # this tells us the amount weight gotta change on l1

                if learn:  # If we were instructed to learn from the data rather than just evaluate it
                    self.train(l1_delta, l2_delta)

                counter += 1  # Keep track of iteration number for error printing
        imgs.close()  # Done with training examples
        return counter, accuracy  # Let the caller know how many samples were trained on and final accuracy

    def train(self, l1_delta, l2_delta):
        self.synapse2 += self.learning_rate * l2_delta
        self.synapse1 += self.learning_rate * l1_delta[:1]  # The bias goes in 1 direction -> cut out 0th element here

    def think_epoch(self, csv_file_path, epoch_num, learn=False):
        samples = 0
        accuracy = 0

        for count in range(0, epoch_num):
            samples, accuracy = self.think(csv_file_path, learn)  # The number examined through epochs will always be the same

        return samples, accuracy

    def demo_network(self, epochs):  # Demo's the networks capabilities
        self.talk()
        start = time()
        examples, accuracy = self.think_epoch("mnist_train.csv", epochs, learn=True)
        end = time()
        print(f'\nI studied a total of {examples} sample(s) in {end - start:2.0f} seconds through {epochs} epoch(s) with a(n) {accuracy:5.2f} % accuracy.')

        print(f'\nI will now take the test set without learning.\n')
        start = time()
        examples, accuracy = self.think("mnist_test.csv")
        end = time()
        print(f'\nI examined a total of {examples} test sample(s) in{end - start:2.0f} seconds with a(n) {accuracy:5.2f} % accuracy.')

    def hidden_neuron_test(self):  # Tests the effects of different numbers of hidden neurons with 50 epochs + logs
        print("Beginning 20 hidden neuron test...\n")
        self.hidden_neurons = 20
        log = open("hidden_neuron_results_20.txt", "w")
        log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")
        for epoch_num in range(1, 51):
            sample_train, accuracy_train = self.think("mnist_train.csv", learn=True)
            sample_test, accuracy_test = self.think("mnist_test.csv")
            log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}')
            print(f'Epoch {epoch_num} of 50 completed')
        log.close()
        print('...20 hidden neuron test complete.\n')

        print("Beginning 50 hidden neuron test...\n")
        self.hidden_neurons = 50
        log = open("hidden_neuron_results_50.txt", "w")
        log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")
        for epoch_num in range(1, 51):
            sample_train, accuracy_train = self.think("mnist_train.csv", learn=True)
            sample_test, accuracy_test = self.think("mnist_test.csv")
            log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}')
            print(f'Epoch {epoch_num} of 50 completed')
        log.close()
        print('...50 hidden neuron test complete.\n')

        print("Beginning 100 hidden neuron test...\n")
        self.hidden_neurons = 100
        log = open("hidden_neuron_results_100.txt", "w")
        log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")
        for epoch_num in range(1, 51):
            sample_train, accuracy_train = self.think("mnist_train.csv", learn=True)
            sample_test, accuracy_test = self.think("mnist_test.csv")
            log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}')
            print(f'Epoch {epoch_num} of 50 completed')
        log.close()
        print('...100 hidden neuron test complete.\n')

        print('All tests successfully completed and results logged.')


# NateBrain ends here


nate = NateBrain()
nate.hidden_neuron_test()


# Notes

# Deriving the weights
# We want [-.05, .05) as the initial random weight range so we use the below formula
# (b-a) * random.random((array)) + a
# ((.05)-(-.05)) * random.random((array)) + (-.05)
# .1 * random.random((array)) - .05

# Array Size
# A synapse layer needs to be an array of the total inputs from the previous layer (including bias), x
# crossed by the number of neurons in the next layer, y. In this case we need to make space for the bias in two places,
# on the input layer and on the hidden layer so we add one space to the array
#
# Layer 1 = 784 inputs + 1 bias and n hidden layer neurons -> (input_num + 1, hidden_neurons + 1)
# Layer 2 = n hidden layer neurons + 1 bias and the number of outputs -> (hidden_neurons + 1, output_neurons)

# Sigmoid Functions
# The sigmoid function normalizes the sum of weighted inputs into a value between 0 and 1
# The sigmoid derivative measures the gradient of the curve leading to the optimum

# MNIST Training Set
# Each line is the label of the digit followed by the 28x28 (784) pixel image

# Backpropagation Notes
# Basically calculate the error starting at the back and working your way back through the network.
