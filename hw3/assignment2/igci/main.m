clear;
clc;
load('GDP.mat')
load('stocks.mat')
%%  Australia
x = [Australia_GDP;Australia_GDP];
y = [Australia_stock;Australia_stock];
austrilia = igci(x,y,2,1);

%% Brazil
x = [Brazil_GDP;Brazil_GDP];
y = [Brazil_stock;Brazil_stock];
brazil = igci(x,y,2,1);

%% Canada
x = [Canada_GDP;Canada_GDP];
y = [Canada_stock;Canada_stock];
canada = igci(x,y,2,1);

%% China
x = [China_GDP;China_GDP];
y = [China_stock;China_stock];
china = igci(x,y,2,1);

%% Germany
x = [Germany_GDP;Germany_GDP];
y = [Germany_stock;Germany_stock];
germany = igci(x,y,2,1);

%% India
x = [India_GDP;India_GDP];
y = [India_stock;India_stock];
india = igci(x,y,2,1);

%% UnitedKingdom
x = [UnitedKingdom_GDP;UnitedKingdom_GDP];
y = [UnitedKingdom_stock;UnitedKingdom_stock];
unitedKingdom = igci(x,y,2,1);

%% UnitedStates
x = [UnitedStates_GDP;UnitedStates_GDP];
y = [UnitedStates_stock;UnitedStates_stock];
unitedStates = igci(x,y,2,1);