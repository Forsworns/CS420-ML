import sys
sys.path.append('..')
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from configs import TEST_SIZE

def loadData():
    data = load_iris()
    x = data['data']
    y = data['target']
    xTrain, xTest, yTrain, yTest = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=1)
    return xTrain, xTest, yTrain, yTest


if __name__ == "__main__":
    data = load_iris()
    x = data['data']
    y = data['target']
