import sys
sys.path.append('..')
import os
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from configs import TEST_SIZE
import numpy as np


def loadData():
    mnist = input_data.read_data_sets("./dataset/mnist/")
    xTrain = mnist.train.images
    xTest = mnist.test.images
    yTrain = mnist.train.labels
    yTest = mnist.test.labels
    return xTrain, xTest, yTrain, yTest


if __name__ == "__main__":
    os.chdir('..')
    xTrain, xTest, yTrain, yTest = loadData()
    print(xTrain.shape)
    print(xTest.shape)
    print(yTrain.shape)
    print(yTest.shape)
