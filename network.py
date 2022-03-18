import numpy as np


class Network:

    def __init__(self, neuron, inputs, test, learning_rate, epochs):
        self._weights = []  # Weights (first one is bias)
        self._neuron = neuron  # Neuron
        self._inputs = inputs  # Training list
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._test = test  # Testing list
        for _ in self._inputs[0]:
            self._weights.append(np.random.randn())  # choose random weights

    def set_lr_ep(self, learning_rate, epochs):
        self._learning_rate = learning_rate
        self._epochs = epochs

    def train_ADALINE(self):
        for ep in range(self._epochs):
            for inp in self._inputs:
                self._neuron.set_inputs([1.0] + inp[:-1])
                self._neuron.set_weights(self._weights)
                output = self._neuron.activation_func()
                loss = (output - inp[-1])  # Difference between real and expected output
                gradients = []
                for ins in ([1.0] + inp[:-1]):
                    gradients.append(loss * ins)  # Calculate value, that would be multiplied to weight
                # Update weights
                for idx, _ in enumerate(self._weights):
                    self._weights[idx] -= self._learning_rate * gradients[idx]

    def res(self, mess="Final: "):
        loss = 0.0
        guess = 0.0
        for inp in self._test:
            self._neuron.set_inputs([1.0] + inp[:-1])
            self._neuron.set_weights(self._weights)
            output = self._neuron.activation_func()

            loss += (output - inp[-1]) ** 2
            if round(output) == inp[-1]:
                guess += 1.0

        loss /= len(self._test)
        guess /= len(self._test)
        # f = open("output.txt", "a")
        # f.write(mess + str(loss) + "," + str(guess) + "\n")
        print(mess + " Loss: " + str(loss) + " Guessed: " + str(guess))
        return loss

    def train_gradient(self):
        for t in range(self._epochs):  # Running epochs
            if t % 100 == 99:
                self.res(str(t) + ",")
            y_pred = []  # Will be stored predicted outputs
            y = []  # Will be stored actual outputs
            for inp in self._inputs:  # Get ouput and modify some values
                self._neuron.set_inputs([1.0] + inp[:-1])
                self._neuron.set_weights(self._weights)
                output = self._neuron.activation_func()
                y_pred.append(output)
                y.append(inp[-1])
            y_pred = np.array(y_pred)
            y = np.array(y)
            # Count (loss) (MSE - Mean Squared Error)
            loss = np.square(y_pred - y).mean()
            # Count gradients
            grad_y_pred = 2.0 * (y_pred - y)
            gradients = []
            transposed = np.array(self._inputs).transpose()  # Transpose inputs, so there would be array of weights
            gradients.append(grad_y_pred.mean())
            for x in transposed[:-1]:
                gradients.append((grad_y_pred * x).mean())
            # Update the weights
            for idx, _ in enumerate(self._weights):
                self._weights[idx] -= self._learning_rate * gradients[idx]
