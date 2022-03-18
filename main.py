import sys
from neuron import Neuron
from network import Network
import numpy as np


def read_inputs(fname):
    data = []
    input_file = open(fname, "r")
    for line in input_file:
        data += [line.split(',')]  # Transforming data to list of lists
    return data


def main(argv):
    if not len(argv) == 4:
        print("python main.py <file> <learning_rate> <epochs> <activation>")
        sys.exit(0)
    f = open("output.txt", "w")
    f.write("")
    neuron1 = Neuron(0, [], [])
    neuron2 = Neuron(0, [], [])

    neuron1.set_activation(int(argv[3]))
    neuron2.set_activation(int(argv[3]))

    inputs = read_inputs(argv[0])
    # First case
    zeros = []  # All zero labeled data
    ones = []  # All one labeled data
    for i in inputs:
        temp = list(map(float, i[:-1]))  # Get inputs without classes
        if len(temp) <= 1:  # Skip empty data
            continue
        if i[-1] == 'Iris-setosa\n':
            temp.append(0)  # Append class label
            zeros.append(temp)  # Add to list
        else:
            temp.append(1) # Append class label
            ones.append(temp)  # Add to list
    # Get split index in 80:20
    split0 = int(len(zeros) * 0.2)
    split1 = int(len(ones) * 0.2)
    # Split them
    test1 = zeros[:split0] + ones[:split1]
    inputs1 = zeros[split0:] + ones[split1:]
    # Create and train neuron
    train1 = Network(neuron1, inputs1, test1, float(argv[1]), int(argv[2]))
    train1.train_gradient()
    # Get results
    train1.res()
    # Second case
    zeros = []  # All zero labeled data
    ones = []  # All one labeled data
    for i in inputs:
        temp = list(map(float, i[:-1]))  # Get inputs without classes
        if len(temp) <= 1:  # Skip empty data
            continue
        if i[-1] == 'Iris-versicolor\n':
            temp.append(0)  # Append class label
            zeros.append(temp)  # Add to list
        elif i[-1] == 'Iris-virginica\n':
            temp.append(1)  # Append class label
            ones.append(temp)  # Add to list
    # Get split index in 80:20
    split0 = int(len(zeros) * 0.2)
    split1 = int(len(ones) * 0.2)
    # Split them
    test2 = zeros[:split0] + ones[:split1]
    inputs2 = zeros[split0:] + ones[split1:]
    # Create and train neuron
    train2 = Network(neuron2, inputs2, test2, float(argv[1]), int(argv[2]))
    train2.train_gradient()
    # Ger results
    train2.res()

if __name__ == '__main__':
    main(sys.argv[1:])
