import numpy as np
from itertools import cycle, islice

INFINITE = 4000
POINTS_NUM = 100
MAX_ITE = 1000
DELTA = 0.001
EPSILON = 0.001
FONTSIZE = 20
POINTSIZE = 20

COLORS = np.array(islice(cycle(['#FF3333',  # red
                   '#0198E1',  # blue
                   '#BF5FFF',  # purple
                   '#FCD116',  # yellow
                   '#4DBD33',  # green
                   '#87421F',   # brown
                   '#000000',  # black
                   ]),POINTS_NUM))

CRI_FILES = ["data/aic_com.json", "data/bic_com.json", "data/aic_EM.json", "data/bic_EM.json"]
