import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as f:
        content = pickle.load(f, encoding='bytes')
    return content[b'data'], content[b'label']

def loadData():
    xTrain,yTrain = unpickle("./cifar-10/data_batch_1")
    XTest,yTest = unpickle("./cifar-10/test_batch")
    for i in range(2,6):
        x,y = unpickle("./cifar-10/data_batch_{}".format(i))
        xTrain = np.concatenate((xTrain,x),axis=0)
        yTrain = np.concatenate((yTrain,y),axis=0)
    return xTrain,XTest,yTrain,yTest


if __name__ == "__main__":
    xTrain,yTrain = unpickle("./cifar-10/data_batch_1")
    XTest,yTest = unpickle("./cifar-10/test_batch")