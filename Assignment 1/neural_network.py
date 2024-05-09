# Nihal Ranchod -> 2427378

import numpy as np

class NeuralNetwork:
    def __init__(self):
        # Initialize weights and biases
        self.weights_input_hidden = np.ones((4, 8))
        self.bias_hidden = np.ones(8)
        self.weights_hidden_output = np.ones((8, 3))
        self.bias_output = np.ones(3)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sum_of_squares_loss(self, predicted, target):
        return 0.5 * np.sum((predicted - target) ** 2)

    def feedforward(self, inputs):
        # Input to hidden layer
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)

        # Hidden to output layer
        output_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        output_outputs = self.sigmoid(output_inputs)

        return output_outputs

    def backpropagation(self, inputs, target, learning_rate=0.1):
        # Feedforward
        hidden_inputs = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_outputs = self.sigmoid(hidden_inputs)
        output_inputs = np.dot(hidden_outputs, self.weights_hidden_output) + self.bias_output
        output_outputs = self.sigmoid(output_inputs)

        # Output layer error
        output_errors = (output_outputs - target) * output_outputs * (1 - output_outputs)

        # Backpropagated error
        hidden_errors = np.dot(output_errors, self.weights_hidden_output.T) * hidden_outputs * (1 - hidden_outputs)

        # Update weights and biases
        self.weights_hidden_output -= learning_rate * np.dot(hidden_outputs.T, output_errors)
        self.bias_output -= learning_rate * output_errors.sum(axis=0)
        self.weights_input_hidden -= learning_rate * np.dot(inputs.T, hidden_errors)
        self.bias_hidden -= learning_rate * hidden_errors.sum(axis=0)

if __name__ == "__main__":

    # Read input and target datat
    data = []
    for _ in range(7):
        data.append(float(input()))

    inputs = np.array([data[:4]])
    target = np.array([data[4:]])

    # Instantiate neural network
    nn = NeuralNetwork()

    # Feedforward to obtain output and compute initial loss
    initial_output = nn.feedforward(inputs)
    initial_loss = nn.sum_of_squares_loss(initial_output, target)

    # Backpropagation
    nn.backpropagation(inputs, target)

    # Feedforward after one iteration of backpropagation
    updated_output = nn.feedforward(inputs)
    updated_loss = nn.sum_of_squares_loss(updated_output, target)

    # Output results
    print(f'{round(initial_loss, 4)}')
    print(f'{round(updated_loss, 4)}')
    # print("Loss before training:", round(initial_loss, 4))
    # print("Loss after training:", round(updated_loss, 4))
