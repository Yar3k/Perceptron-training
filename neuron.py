import math


class Neuron:
    # Constructor
    def __init__(self, activation_type, inputs, weights):
        self._activation_type = activation_type  # 0 - Heaviside, 1 - Logistic
        self._inputs = inputs  # First element always 1
        self._weights = weights  # First element is bias

    # Sets weights
    def set_weights(self, weights):
        self._weights = weights

    # Sets inputs
    def set_inputs(self, inputs):
        self._inputs = inputs

    # Sets activation type
    def set_activation(self, activation_type):
        self._activation_type = activation_type

    def get_weights(self):
        return self._weights

    def get_inputs(self):
        return self._inputs

    def get_activation(self):
        return self._activation_type

    # Sums all inputs with weights and bias
    def __activation(self):
        output = 0.0
        for (i, v) in zip(self._inputs, self._weights):
            output += float(i) * v
        return output

    # Output, depends on chosen activation type
    def activation_func(self):
        if self._activation_type == 0:
            if self.__activation() >= 0:
                return 1
            else:
                return 0

        if self._activation_type == 1:
            return 1 / (1 + math.e ** (-1 * self.__activation()))
