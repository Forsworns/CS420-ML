from sklearn.neural_network import MLPClassifier
from sl_rm import *
from configs import *

from dataset.iris import loadData as loadDataA1
from dataset.cifar import loadData as loadDataA2
from dataset.mnist import loadData as loadDataB

def MLP_recommend(**MLP_paras):
    if MLP_paras == {}:
        MLP_paras = A1_MLP_PARAS
    return MLPClassifier(**MLP_paras)

def MLP_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={}, **MLP_paras):
    if MLP_paras == {}:
        MLP_paras = A1_MLP_PARAS
    if paras == {}:
        paras.update(MLP_paras)
    result = load_result(model_name, paras)
    if result is None:
        clf = load_model(model_name, paras)
        if clf is None:
            print("can't find clf",model_name)
            clf = MLPClassifier(**MLP_paras)
            clf.fit(X_train, y_train)
            save_model(clf, model_name, paras)
        sc = clf.score(X_test, y_test)
        # unweighted mean of metrics for labels
        result = {'score': sc}
        save_result(result, model_name, paras)
        print("{} with {}: score is {}".format(
            model_name, paras, sc))
        return clf
    else:
        clf = load_model(model_name, paras)
        sc = result.values()
        print("{} with {}: score is {}".format(
            model_name, paras, sc))
        return clf


def MLP_base(xTrain, xTest, yTrain, yTest, datasetName):
    # model_name为采用的降维方法，X为降维后的feature数据
    for a in ACTIVATION:
        for alpha in ALPHA:
            for h in HIDDEN:
                MLP_recommend_run(datasetName, xTrain, xTest, yTrain, yTest, paras={}, activation=a, alpha=alpha, hidden_layer_sizes=h)

if __name__ == "__main__":
    xTrain,xTest,yTrain,yTest = loadDataA1()
    MLP_base(xTrain,xTest,yTrain,yTest,A1_NN)

    xTrain,xTest,yTrain,yTest = loadDataA2()
    MLP_base(xTrain,xTest,yTrain,yTest,A2_NN)