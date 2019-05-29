from sklearn.svm import SVC, LinearSVC
from sl_rm import *
from configs import *

def SVM_recommend(**SVM_paras):
    if SVM_paras == {}:
        SVM_paras = A1_SVM_PARAS
    return LinearSVC(**SVM_paras)

def SVM_recommend_run(model_name, X_train, X_test, y_train, y_test, paras={}, **SVM_paras):
    if SVM_paras == {}:
        SVM_paras = A1_SVM_PARAS
    if paras == {}:
        paras.update(SVM_paras)
    result = load_result(model_name, paras)
    if result is None:
        clf = load_model(model_name, paras)
        if clf is None:
            print("can't find clf",model_name)
            clf = SVC(**SVM_paras)
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


def SVM_base(xTrain, xTest, yTrain, yTest, datasetName):
    # model_name为采用的降维方法，X为降维后的feature数据
    for d in DECI_FUNCS:
        for k in KERNELS:
            for C in CS:
                SVM_recommend_run(datasetName, xTrain, xTest, yTrain, yTest, paras={}, C=C, kernel=k, decision_function_shape=d)


if __name__ == "__main__":
    pass
