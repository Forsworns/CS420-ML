import numpy as np
from configs import TEST_SIZE
from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('..')


def loadData():
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)
    xTrain = mnist.train.images
    xTest = mnist.test.images
    yTrain = mnist.train.labels
    yTest = mnist.test.labels
    return xTrain, xTest, yTrain, yTrain


if __name__ == "__main__":
    mnist = input_data.read_data_sets("./mnist/", one_hot=True)
    print(mnist.train.images.shape)
