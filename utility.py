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

from time import time
from nate_network import NateBrain
import texttable as ttable


def print_results_matrix(matrix, total_acc):

    table1 = ttable.Texttable().set_cols_align(["c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"])\
                               .set_cols_valign(["c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c", "c"])\
                               .set_cols_width(["9", "9", "9", "9", "9", "9", "9", "9", "9", "9", "14", "12"])\
                               .add_row(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Predicted x Actual", "Accuracy"])\
                               .add_rows(matrix, False)\
                               .add_row(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "Total Avg Acc", total_acc])

    table_str = table1.draw() + "\n"
    print(table_str)
    return table_str


def demo_network(nate: NateBrain, epochs):  # Demo's the networks capabilities
    nate.talk()

    start = time()
    if epochs == 0:  # We will run at least one epoch
        epochs = 1
    examples, accuracy, results = nate.think("mnist_train.csv", learn=True)
    for _ in range(1, epochs):  # this guarantees that it will run at least once but no more than 'epochs'
        examples, accuracy, results = nate.think("mnist_train.csv", learn=True)
    end = time()
    print(f'\nI studied a total of {examples} sample(s) in {end - start:5.0f} seconds through {epochs}'
          f' epoch(s) with a(n) {accuracy:5.2f} % accuracy.')
    print_results_matrix(results, accuracy)

    print(f'\nI will now take the test set without learning.\n')
    start = time()
    examples, accuracy, results = nate.think("mnist_test.csv")  # ...But we do here
    end = time()
    print(f'\nI examined a total of {examples} test sample(s) in{end - start:5.0f} seconds with a(n)'
          f' {accuracy:5.2f} % accuracy.\n')
    print_results_matrix(results, accuracy)


def hidden_neuron_test(nate: NateBrain):  # Tests effect of different numbers of hidden neurons with 50 epochs + logs
    print(f"Beginning {nate.hidden_neurons} hidden neuron test...\n")
    start_t = time()
    log = open(f"hidden_neuron_results_{nate.hidden_neurons}.txt", "w")
    log.write(f"{nate.hidden_neurons} hidden neurons, {nate.learning_rate} learning rate,"
              f" {nate.momentum} momentum rate.\n")
    log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")

    accuracy_test = 0.0  # These are here to silence warnings about potential non-assignment
    results_test = []
    for epoch_num in range(1, 51):
        sample_train, accuracy_train, results_train = nate.think("mnist_train.csv", learn=True)
        sample_test, accuracy_test, results_test = nate.think("mnist_test.csv")
        log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}\n')
        print(f'Epoch {epoch_num} of 50 completed')
    end_t = time()
    log.write(f'Total execution time is {end_t - start_t:5.0f} seconds\n')
    log.write(f'Confusion Matrix for tests\n')
    log.write(print_results_matrix(results_test, accuracy_test))
    log.close()
    print(f'...{nate.hidden_neurons} hidden neuron test complete in {end_t - start_t:5.0f} seconds.\n')


def quarter_training_set_test(nate: NateBrain):  # Tests the neural net on 1/4th of the training set
    print("Beginning quartered training set test...\n")
    start_t = time()
    log = open("quarter_training_set_test_result.txt", "w")
    log.write(f"{nate.hidden_neurons} hidden neurons, {nate.learning_rate} learning rate,"
              f" {nate.momentum} momentum rate.\n")
    log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")

    accuracy_test = 0.0  # These are here to silence warnings about potential non-assignment
    results_test = []
    for epoch_num in range(1, 51):
        sample_train, accuracy_train, results_train = nate.think("quarter_mnist_train.csv", learn=True)
        sample_test, accuracy_test, results_test = nate.think("mnist_test.csv")
        log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}\n')
        print(f'Epoch {epoch_num} of 50 completed')
    end_t = time()
    log.write(f'Total execution time is {end_t - start_t:5.0f} seconds\n')
    log.write(f'Confusion Matrix for tests\n')
    log.write(print_results_matrix(results_test, accuracy_test))
    log.close()
    print(f'... quartered training set test complete in {end_t - start_t:5.0f} seconds.\n')


def half_training_set_test(nate: NateBrain):  # Tests the neural net on 1/2th of the training set
    print("Beginning halved training set test...\n")
    start_t = time()
    log = open("half_training_set_test_result.txt", "w")
    log.write(f"{nate.hidden_neurons} hidden neurons, {nate.learning_rate} learning rate,"
              f" {nate.momentum} momentum rate.\n")
    log.write("Format is as follows {epoch_num, accuracy_train, accuracy_test}\n")

    accuracy_test = 0.0  # These are here to silence warnings about potential non-assignment
    results_test = []
    for epoch_num in range(1, 51):
        sample_train, accuracy_train, results_train = nate.think("half_mnist_train.csv", learn=True)
        sample_test, accuracy_test, results_test = nate.think("mnist_test.csv")
        log.write(f'{epoch_num}, {accuracy_train:5.2f}, {accuracy_test:5.2f}\n')
        print(f'Epoch {epoch_num} of 50 completed')
    end_t = time()
    log.write(f'Total execution time is {end_t - start_t:5.0f} seconds\n')
    log.write(f'Confusion Matrix for tests\n')
    log.write(print_results_matrix(results_test, accuracy_test))
    log.close()
    print(f'... halved training set test complete in {end_t - start_t:5.0f} seconds.\n')