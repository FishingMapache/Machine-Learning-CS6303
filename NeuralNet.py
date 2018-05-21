#####################################################################################################################
#   CS 6375.003 - Assignment 3, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection

class NeuralNet:
    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        #raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        raw_input = pd.DataFrame(train)
        self.X, self.y, self.scaler = self.preprocess(raw_input)
        self.activation = "Relu"
        self.iterations = 500
        self.learningrate = 0.05

        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "tanh":
            self.__tanh(self, x)
        if activation == "Relu":
            self.__Relu(self, x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)
        if activation == "Relu":
            self.__Relu_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def __Relu(self, x):
        return np.maximum(x, 0)

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):
        return 1 - np.power(2, (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))

    def __Relu_derivative(self, x):
        x[x<= 0] = 0
        x[x>0] = 1
        return x

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        # Missing Values
        X = X.dropna()
        # Encoding
        X = X.apply(preprocessing.LabelEncoder().fit_transform)
        # Scaling
        nrows = len(X.index)
        ncols = len(X.columns)
        p = X.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols - 1)
        q = X.iloc[:, (ncols - 1)].values.reshape(nrows, 1)
        scaler = preprocessing.Normalizer().fit(p)
        p = scaler.transform(p)
        # Standardizing
        scaler = preprocessing.StandardScaler().fit(p)
        p = scaler.transform(p)
        return p, q, scaler

    # Below is the training function

    def train(self):
        max_iterations = self.iterations
        learning_rate = self.learningrate
        for iteration in range(max_iterations):
            out = self.forward_pass(self.activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, self.activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self, activation):
        # pass our inputs through our neural network
        if activation == "sigmoid":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
        if activation == "tanh":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
        if activation == "Relu":
            in1 = np.dot(self.X, self.w01)
            self.X12 = self.__Relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__Relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__Relu(in3)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))

        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        if activation == "Relu":
            delta_output = (self.y - out) * (self.__Relu_derivative(out))

        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))

        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))

        if activation == "Relu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__Relu_derivative(self.X23))

        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))

        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))

        if activation == "Relu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__Relu_derivative(self.X12))

        self.delta12 = delta_hidden_layer1

    # TODO: Implement other activation functions

    def compute_input_layer_delta(self, activation):
        if activation == "sigmoid":
            delta_input_layer = np.multiply(self.__sigmoid_derivative(self.X01), self.delta01.dot(self.w01.T))

        if activation == "tanh":
            delta_input_layer = np.multiply(self.__tanh_derivative(self.X01), self.delta01.dot(self.w01.T))

        if activation == "Relu":
            delta_input_layer = np.multiply(self.__Relu_derivative(self.X01), self.delta01.dot(self.w01.T))

            self.delta01 = delta_input_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True):

        test_input = pd.DataFrame(test)
        #test_input = pd.read_csv(test)
        test_dataset = test_input.dropna()
        test_dataset = test_dataset.apply(preprocessing.LabelEncoder().fit_transform)
        nrows = len(test_dataset.index)
        ncols = len(test_dataset.columns)
        self.X = test_dataset.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        self.y = test_dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)
        scaler = preprocessing.Normalizer().fit(self.X)
        self.X = scaler.transform(self.X)
        self.X = self.scaler.transform(self.X)

        out = self.forward_pass(self.activation)
        error = 0.5 * np.power((out - self.y), 2)

        print("Testing error is " + str(np.sum(error)))
        return out

if __name__ == "__main__":
    # NOTICE: use http instead of https
    #

    # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    # url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

    data = pd.read_csv(url, header = None)
    train_set, test_set = model_selection.train_test_split(data, test_size = 0.3)

    neural_network = NeuralNet(train_set)
    neural_network.train()
    testError = neural_network.predict(test_set)

