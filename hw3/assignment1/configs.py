import numpy as np

# file type
MODEL = "models"
RESULT = "results"

# methods name/model name
A1_SVM = "a1_svm"
A2_SVM = "a2_svm"
B_SVM = "b_svm"
A1_NN = "a1_nn"
A2_NN = "a2_nn"

# test train split
TEST_SIZE = 0.4

# paras for SVM base
DECI_FUNCS = ['ovo', 'ovr']
KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']
KERNELS_MAP = {'linear':0, 'poly':1, 'rbf':2, 'sigmoid':3}
CS = [0.01, 0.04, 0.07, 0.1, 0.3, 0.5, 1, 2]

# paras for MLP base
ACTIVATION = ['relu','logistic','tanh']
ALPHA = [1e-5,1e-4,1e-3,1e-2,1e-1]
HIDDEN = [(5,),(25,),(100,),(100,20,),(100,50,20,)]
HIDDEN_P = ['5-','25-','100-','100- 20','100- 50- 20']
HIDDEN_MAP = {'5-':0,'25-':1,'100-':2,'100- 20':3,'100- 50- 20':4}

# paras for baseline
A1_SVM_PARAS = {'C': 0.01, 'kernel': 'linear', 'max_iter': 2000}
A2_SVM_PARAS = {}
B1_SVM_PARAS = {}
A1_MLP_PARAS = {'alpha':1e-5, 'hidden_layer_sizes':(5,), 'activation':'relu', 'early_stopping':True}
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