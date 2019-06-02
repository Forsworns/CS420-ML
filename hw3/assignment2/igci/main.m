clear;
clc;
load('GDP.mat')
load('stocks.mat')

x = randn(100,1); 
y = exp(x); 
igci(x,y,2,1);