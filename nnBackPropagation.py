import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math

option = int(
    input(("Select an option: \n 1 : training percentage \n 2 : 'k' partions \n")))


def getTrainingPercentage():
    return float(
        input("Enter the training percentage value like '0.7': "))


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


user_value = getInputMethod(option)


variables_number = 4
intermediate_neurons = 10
hidden_neurons = 8
output_neurons = 3

# iris --> dataframe
iris = pd.read_csv("data/iris.csv", sep=';')

# randomize data
iris = iris.sample(frac=1)

# change string variables to numbers call one hot variables
one_hot = pd.get_dummies(
    iris['class'])
iris = iris.drop('class', axis=1)
iris = iris.join(one_hot)

# drop unnecessary column
iris = iris.drop('ID', axis=1)


# x columns
subSetIris = iris.iloc[:, 0:variables_number]

X, Y = subSetIris, iris.iloc[:, variables_number:]

training_index = math.ceil(len(subSetIris) * (user_value))

X_train, X_test, y_train, y_test = X[:training_index], X[training_index:
                                                         ], Y[:training_index], Y[training_index:]

X_train, X_test, y_train, y_test = X_train.to_numpy(
), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

y_train.shape = ((len(y_train), 3))
y_test.shape = ((len(y_test), 3))

'''
TODO REFACTOR
# cross validation
def getPartition(iris, k, n):
    test = iris.iloc[int(n/k):int((n+1)/k), :]
    train = iris.iloc[int(n+1/k):int((n+1)/k), :]

    # x columns
    subSetIris = iris.iloc[:, 0:variables_number]

    X, Y = subSetIris, iris.iloc[:, variables_number:]

    training_index = math.ceil(len(subSetIris) * (user_value))

    X_train, X_test, y_train, y_test = X[:training_index], X[training_index:
                                                             ], Y[:training_index], Y[training_index:]

    X_train, X_test, y_train, y_test = X_train.to_numpy(
    ), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    y_train.shape = ((len(y_train), 3))
    y_test.shape = ((len(y_test), 3))

    print(par)
    return par, n
'''

'''sigmoid function'''


def nonlin(p, deriv=False):
    if(deriv == True):
        return p*(1-p)
    return 1/(1+np.exp(-p))


def getInitialWeight(variables_number, intermediate_neurons, hidden_neurons, output_neurons):

    np.random.seed(1)
    syn0 = 2*np.random.random((variables_number, intermediate_neurons)) - 1

    syn1 = 2*np.random.random((intermediate_neurons, hidden_neurons)) - 1

    syn2 = 2*np.random.random((hidden_neurons, output_neurons)) - 1

    return syn0, syn1, syn2


def getResultNN(variables_number, intermediate_neurons, hidden_neurons, output_neurons):
    eta = 0.1
    iteration_values = []
    error_values = []
    epochs = 80000

    syn0, syn1, syn2 = getInitialWeight(variables_number, intermediate_neurons,
                                        hidden_neurons, output_neurons)

    for iter in range(epochs):
        # Forward Propagation between layer 0 to layer 1 and layer 2 (end)
        iteration_values.append(iter)

        l0 = X_train
        l1 = nonlin(np.dot(l0, syn0))
        l2 = nonlin(np.dot(l1, syn1))
        l3 = nonlin(np.dot(l2, syn2))

        l3_error = y_train-l3
        l3_delta = l3_error * nonlin(l3, deriv=True)*eta

        l2_error = l3_delta.dot(syn2.T)
        l2_delta = l2_error * nonlin(l2, deriv=True)*eta

        l1_error = l2_delta.dot(syn1.T)

        l1_delta = l1_error * nonlin(l1, deriv=True)*eta

        mse = (np.square(l3_error)).mean()
        error_values.append(mse)

        # update weights
        syn2 += l2.T.dot(l3_delta)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)

    return iteration_values, error_values


iteration_values, error_values = getResultNN(variables_number, intermediate_neurons,
                                             hidden_neurons, output_neurons)

# print(iteration_values, error_values)
plt.title('MSE red neuronal')
plt.plot(iteration_values, error_values)
plt.show()
