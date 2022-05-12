import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


user_option = int(
    input(("Select an option: \n 1 : training percentage \n 2 : 'k' partions \n")))


def getTrainingPercentage():
    return float(
        input("Enter the training percentage value like '0.15': "))


def getKPartitions():
    return int(input("enter the k value: "))


def invalid_op():
    raise Exception("Invalid operation")


def getInputMethod(chosen_training_method=1):
    ops = {
        1: getTrainingPercentage,
        2: getKPartitions
    }
    chosen_function = ops.get(chosen_training_method, invalid_op)
    return chosen_function()


user_value = getInputMethod(user_option)


iris = pd.read_csv("input/Iris.csv")
# randomize data
iris = iris.sample(frac=1).reset_index(drop=True)

X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)


# change string variables to numbers call one hot variables
one_hot_encoder = OneHotEncoder(sparse=False)
Y = iris.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))


def getTrainingTestValues(X, Y, test_size=0.15):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1)
    return X_train, X_test, X_val, Y_train, Y_test, Y_val


def getPartitionCrossValidation(X, Y, k):
    index_partition = math.ceil((len(X))/k)

    x_partitions = []
    y_partitions = []
    for index in range(0, len(X), index_partition):
        if index == 0:
            x, y = X[index:index_partition], Y[index:index_partition]
        elif index == len(X)-1:
            x, y = X[index:], Y[index:]
        else:
            x, y = X[index:index_partition +
                     index], Y[index:index_partition+index]

        x_partitions.append(x)
        y_partitions.append(y)

    return x_partitions, y_partitions


def NeuralNetwork(X_train, Y_train, X_val=None, Y_val=None, epochs=10, nodes=[], lr=0.15):
    hidden_layers = len(nodes) - 1
    weights = InitializeWeights(nodes)

    for epoch in range(1, epochs+1):
        weights = Train(X_train, Y_train, lr, weights)

        if(epoch % 20 == 0):
            print("Epoch {}".format(epoch))
            print("Training Accuracy:{}".format(
                Accuracy(X_train, Y_train, weights)))
            if X_val.any():
                print("Validation Accuracy:{}".format(
                    Accuracy(X_val, Y_val, weights)))

    return weights


def InitializeWeights(nodes):
    """Initialize weights with random values in [-1, 1] (including bias)"""
    layers, weights = len(nodes), []

    for i in range(1, layers):
        w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)]
             for j in range(nodes[i])]
        weights.append(np.matrix(w))

    return weights


def ForwardPropagation(x, weights, layers):
    activations, layer_input = [x], x
    for j in range(layers):
        activation = Sigmoid(np.dot(layer_input, weights[j].T))
        activations.append(activation)
        layer_input = np.append(1, activation)  # Augment with bias

    return activations


def BackPropagation(y, activations, weights, layers):
    outputFinal = activations[-1]
    error = np.matrix(y - outputFinal)  # Error at output

    for j in range(layers, 0, -1):
        current_activation = activations[j]

        if(j > 1):
            # Augment previous activation
            prevActivation = np.append(1, activations[j-1])
        else:
            # First hidden layer, prevActivation is input (without bias)
            prevActivation = activations[0]

        delta = np.multiply(error, SigmoidDerivative(current_activation))
        weights[j-1] += lr * np.multiply(delta.T, prevActivation)

        w = np.delete(weights[j-1], [0], axis=1)  # Remove bias from weights
        error = np.dot(delta, w)  # Calculate error for current layer

    return weights


def Train(X, Y, lr, weights):
    layers = len(weights)
    for i in range(len(X)):
        x, y = X[i], Y[i]
        x = np.matrix(np.append(1, x))  # Augment feature vector

        activations = ForwardPropagation(x, weights, layers)
        weights = BackPropagation(y, activations, weights, layers)

    return weights


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))


def SigmoidDerivative(x):
    return np.multiply(x, 1-x)


def Predict(item, weights):
    layers = len(weights)
    item = np.append(1, item)  # Augment feature vector

    ##_Forward Propagation_##
    activations = ForwardPropagation(item, weights, layers)

    outputFinal = activations[-1].A1  # flatten this matrix
    index = FindMaxActivation(outputFinal)

    # Initialize prediction vector to zeros
    y = [0 for i in range(len(outputFinal))]
    y[index] = 1  # Set guessed class to 1

    return y  # Return prediction vector


def FindMaxActivation(output):
    """Find max activation in output"""
    m, index = output[0], 0
    for i in range(1, len(output)):
        if(output[i] > m):
            m, index = output[i], i

    return index


def Accuracy(X, Y, weights):
    """Run set through network, find overall accuracy"""
    correct = 0

    for i in range(len(X)):
        x, y = X[i], list(Y[i])
        guess = Predict(x, weights)

        if(y == guess):
            # Guessed correctly
            correct += 1

    return correct / len(X)


features = len(X[0])  # Number of features
outputs = len(Y[0])  # Number of outputs / classes

layers = [features, 5, 10, outputs]  # Number of nodes in layers
lr, epochs = 0.15, 80


def getBasicValidation(X, Y, user_value, epochs=epochs, nodes=layers, lr=lr):
    X_train, X_test, X_val, Y_train, Y_test, Y_val = getTrainingTestValues(
        X, Y, user_value)
    weights = NeuralNetwork(X_train, Y_train, X_val, Y_val,
                            epochs=epochs, nodes=layers, lr=lr)
    print("Testing Accuracy: {}".format(Accuracy(X_test, Y_test, weights)))


def getCrossValidation(X, Y, k, epochs=epochs, nodes=layers, lr=lr):

    x_partitions, y_partitions = getPartitionCrossValidation(X, Y, k)

    accuracies = []
    for index in range(len(x_partitions)):
        x_temporal = x_partitions.copy()
        y_temporal = y_partitions.copy()

        X_test = x_temporal.pop(index)
        X_train = mergeArrays(x_temporal)

        Y_test = y_temporal.pop(index)
        Y_train = mergeArrays(y_temporal)

        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=0.1)

        weights = NeuralNetwork(X_train, Y_train, X_val, Y_val,
                                epochs=epochs, nodes=layers, lr=lr)

        accuracies.append(Accuracy(X_test, Y_test, weights))

    return print("Testing with cross validation: {}".format(average(accuracies)))


def average(lst):
    return sum(lst) / len(lst)


def mergeArrays(array):
    merged = array[0]
    for index in range(1, len(array)):
        merged = np.concatenate((merged, array[index]))
    return merged


def getValidationMethod(chosen_training_method=1):
    ops = {
        1: getBasicValidation,
        2: getCrossValidation
    }
    chosen_function = ops.get(chosen_training_method, invalid_op)
    return chosen_function(X, Y, user_value, epochs=epochs, nodes=layers, lr=lr)


getValidationMethod(user_option)
