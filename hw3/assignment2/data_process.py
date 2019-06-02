import numpy as np
import pandas as pd 
import scipy.io as scio

def toTxt():
    countries = ['Australia','Brazil','Canada','China','Germany','India','UnitedKingdom','UnitedStates']
    GDP = scio.loadmat("dataset/GDP.mat")
    stocks = scio.loadmat("dataset/stocks.mat")
    with open("dataset/data.txt","w") as f:
        f.write("GDP stock\n")
        for cty in countries:
            for i in range(0,13):
                f.write("{} {}\n".format(GDP['{}_GDP'.format(cty)][i][0],stocks['{}_stock'.format(cty)][i][0]))


def readGDP():
    inputFile = "dataset/GDP.csv"
    outputFile = "dataset/GDP.mat"
    countries = ['Australia','Brazil','Canada','China','Germany','India','UnitedKingdom','UnitedStates']
    GDP = {}
    df = pd.read_csv(inputFile)
    df = df.iloc[0:9,:]
    df = df.iloc[:,2:18]
    df.drop(columns=['Country Code','2005 [YR2005]'],inplace=True)
    for cty in countries:
        GDP.update({"{}_GDP".format(cty):np.transpose(df[df['Country Name']==cty].iloc[:,1:].values)})
    scio.savemat(outputFile,GDP)


def readStock(cty):
    df = pd.read_csv(cty)
    df = df.loc[:,['日期','收盘']]
    df.loc[:,'日期'] = df.loc[:,'日期'].apply(lambda item:item.split('年')[0])
    df = df[~df['日期'].isin(['2004','2018'])]
    df.loc[:,'收盘'] = df.loc[:,'收盘'].apply(lambda item:float(item.replace(',','')))
    df = df.groupby('日期').mean()
    return df.values


def readStocks():
    countries = ['Australia','Brazil','Canada','China','Germany','India','UnitedKingdom','UnitedStates']
    stocks = {}
    outputFile = "dataset/stocks.mat"
    for cty in countries:
        inputFile = 'dataset/{}.csv'.format(cty)
        stocks.update({"{}_stock".format(cty):readStock(inputFile)})
    scio.savemat(outputFile,stocks)


if __name__ == "__main__":
    # readGDP()
    # readStocks()
    toTxt()