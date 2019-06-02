import sys
sys.path.append('..')
from configs import TEST_SIZE
from sklearn.model_selection import train_test_split
import numpy as np
import os


def readColonFile():
    with open("./dataset/colon-cancer.txt") as f:
        row = f.readline()
        row = row.split(' ')
        x = np.asarray([float(row[i].split(':')[1]) for i in range(2, 2002)])
        y = np.asarray([float(row[0])])
        while row:
            row = f.readline()
            if row!='':
                row = row.split(' ')
                x = np.vstack((x, [float(row[i].split(':')[1])
                                for i in range(2, 2002)]))
                y = np.vstack((y, [float(row[0])]))
    return x, y


def loadData():
    x, y = readColonFile()
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=1)
    return xTrain, xTest, yTrain, yTest


if __name__ == "__main__":
    os.chdir('..')
    x1, x2, y1, y2 = loadData()
    print(x2.shape)
