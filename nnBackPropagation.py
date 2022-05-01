import pandas as pd
import numpy as np
import random
k = int(input("enter the k value: "))
# iris --> dataframe
iris = pd.read_csv("data/iris.csv", sep=';')

# save id and class
iris_classes = iris.iloc[:, [0] + [-1]]


# randomize data
iris = iris.sample(frac=1)

training_perc = float(
    input("Enter the training percentage value like '0.7': "))

# separate into train and test data
# random state is a seed value
train = iris.sample(frac=training_perc, random_state=200)
test = iris.drop(train.index)

'''sigmoid function'''


def nonlin(p, deriv=False):
    if(deriv == True):
        return p*(1-p)
    return 1/(1+np.exp(-p))


np.random.seed(1)


numVar = 5
CantidadNeuroInter = 10
CantidadNeuroH = 8
numSalidas = 1

# pesos
syn0 = 2*np.random.random((numVar, CantidadNeuroInter)) - 1

syn1 = 2*np.random.random((CantidadNeuroInter, CantidadNeuroH)) - 1

syn2 = 2*np.random.random((CantidadNeuroH, numSalidas)) - 1


eta = 0.1
iteraVec = []
vecerror = []

for iter in range(80000):
    # Forward Propagation between layer 0 to leyer 1 and leyer 2 (end)
    iteraVec.append(iter)

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
    vecerror.append(mse)

    # update weights
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)


def get_class(id_):
    """ based on an id returns a class """
    ids = iris_classes.loc[iris_classes['ID'] == id_]
    return ids["class"].values[0]


def validate_prediction(test_class, class_prediction):
    """ validates if a predicted class and the REAL class are EQUAL or NOT"""
    return test_class == class_prediction


'''TODO refactoring'''


def get_prediction(test, trains, k, oneItem=False):
    # element --> [[[id_test, id_train, distance]], [...], [...]]
    dmatrix = get_nearest_neighbors(test, train, k)
    count = 0
    predictions = {}
    for arr in dmatrix:
        class_prediction = get_clasification(arr)
        predictions[arr[0][0]] = class_prediction
        if validate_prediction(get_class(arr[0][0]), class_prediction):
            count += 1

    if not oneItem:
        return (f"accuracy: {round(count / len(test), 5) * 100}% 😊", f"prediction: {predictions}")
    return f"prediction: {predictions}"


'''    
print ('*********************************')
print ('Output After Training:')
print('Salida red:','\n',l2)
print ('Error:' + str(np.mean(np.abs(l2_error))),'\n')
print ('pesos: ')
print (syn0,'\n')
print (syn1,'\n')
'''
# Con formateo
# print "%5d%10s" %(1,'a')
plt.title('MSE red neuronal')
plt.plot(iteraVec, vecerror)
plt.show()
