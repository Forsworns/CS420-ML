clear;
clc;
load('GDP.mat')
load('stocks.mat')
%%  Australia
x = Australia_GDP;
y = Australia_stock;
CauseOrEffect([x,y]);

%% Brazil
x = Brazil_GDP;
y = Brazil_stock;
CauseOrEffect([x,y]);

%% Canada
x = Canada_GDP;
y = Canada_stock;
CauseOrEffect([x,y]);

%% China
x = China_GDP;
y = China_stock;
CauseOrEffect([x,y]);

%% Germany
x = Germany_GDP;
y = Germany_stock;
CauseOrEffect([x,y]);

%% India
x = India_GDP;
y = India_stock;
CauseOrEffect([x,y]);

%% UnitedKingdom
x = UnitedKingdom_GDP;
y = UnitedKingdom_stock;
CauseOrEffect([x,y]);

%% UnitedStates
x = UnitedStates_GDP;
y = UnitedStates_stock;
CauseOrEffect([x,y]);