import numpy as np
import pandas as pd
import scipy.io as scio
from lingam import LiNGAM

def readData():
    GDP = scio.loadmat("../dataset/GDP.mat")
    stocks = scio.loadmat("../dataset/stocks.mat")
    return GDP, stocks

if __name__ == "__main__":
    GDP, stocks = readData()
    countries = ['Australia','Brazil','Canada','China','Germany','India','UnitedKingdom','UnitedStates']
    for cty in countries:
        print(cty)
        x = GDP['{}_GDP'.format(cty)]
        y = stocks['{}_stock'.format(cty)]
        X = pd.DataFrame(np.squeeze(np.asarray([x,y])).T,columns=["GDP","stock"])
        lingam = LiNGAM()
        lingam.fit(X)
        # lingam.fit(X,use_sklearn=True)