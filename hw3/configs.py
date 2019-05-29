import numpy as np

# file type
MODEL = "models"
RESULT = "results"

# methods name/model name
A_SVM = "a_svm"
B_SVM = "b_svm"
A_NN = "nn"

# test train split
TEST_SIZE = 0.4

# paras for SVM base
DECI_FUNCS = ['ovo', 'ovr']
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
KERNELS_MAP = {'linear':0, 'poly':1, 'rbf':2, 'sigmoid':3}
CS = [0.01, 0.04, 0.07, 0.1, 0.3, 0.5, 1, 2]

# paras for baseline
A1_SVM_PARAS = {'C': 0.01, 'kernel': 'linear', 'max_iter': 2000}
A2_SVM_PARAS = {}
B1_SVM_PARAS = {}
A1_MLP_PARAS = {'solver':'lbfgs', 'alpha':1e-5, 'hidden_layer_sizes':(5, 2), 'random_state':1}
A2_MLP_PARAS = {}
B1_MLP_PARAS = {}

COLORS = np.array(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#FF7216',  # orange
                   '#4DBD33',  # green
                   '#87421F',  # brown
                   '#000000'   # black
                   ])