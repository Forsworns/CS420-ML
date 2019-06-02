import numpy as np
import pandas as pd 
import scipy.io as scio


def readGDP():
    inputFile = "dataset/GDP.csv"
    outputFile = "dataset/GDP.mat"
    df = pd.read_csv(inputFile)
    df = df.iloc[0:9,:]
    df = df.iloc[:,2:18]
    df.drop(columns=['Country Code','2005 [YR2005]'],inplace=True)
    scio.savemat(outputFile,{'GDP':df.values})


def readStock(cty):
    df = pd.read_csv(cty)
    df = df.loc[:,['日期','收盘']]
    df.loc[:,'日期'] = df.loc[:,'日期'].apply(lambda item:item.split('年')[0])
    df = df[~df['日期'].isin(['2004','2018'])]
    df.loc[:,'收盘'] = df.loc[:,'收盘'].apply(lambda item:float(item.replace(',','')))
    df = df.groupby('日期').mean()
    return df.values


def readStocks():
    countries = ['Austrilia','Brazil','Canada','China','Germany','India','UnitedKingdom','UnitedState']
    stocks = {}
    outputFile = "dataset/stocks.mat"
    for cty in countries:
        inputFile = 'dataset/{}.csv'.format(cty)
        stocks.update({cty:readStock(inputFile)})
    scio.savemat(outputFile,stocks)


if __name__ == "__main__":
    readGDP()
    readStocks()
