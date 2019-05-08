import numpy as np

POINTS_NUM = 100
BOUNDARY = 10
COVARIANCE_TYPE = 'full'
MAX_ITE = 50
DELTA = 0.00001
EPSILON = 0.001
THRESHOLD = 0.01
FONTSIZE = 20
POINTSIZE = 20

MAX_K = 7
MIN_K = 2

COLORS = np.array(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#4DBD33',  # green
                   '#87421F',   # brown
                   '#000000',  # black
                   ])

CRI_FILES = ["data/aic_com.json", "data/bic_com.json", "data/aic_EM.json", "data/bic_EM.json"]
