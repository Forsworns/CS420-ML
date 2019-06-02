clear;
clc;
load('GDP.mat')
load('stocks.mat')
%%  Australia
x = [Australia_GDP;Australia_GDP];
y = [Australia_stock;Australia_stock];
austrilia = igci(x,y,2,1);
if austrilia<0
    disp("Austrilia:GDP->stock")
else
    disp("Austrilia:stock->GDP")
end

%% Brazil
x = [Brazil_GDP;Brazil_GDP];
y = [Brazil_stock;Brazil_stock];
brazil = igci(x,y,2,1);
if brazil<0
    disp("Brazil:GDP->stock")
else
    disp("Brazil:stock->GDP")
end

%% Canada
x = [Canada_GDP;Canada_GDP];
y = [Canada_stock;Canada_stock];
canada = igci(x,y,2,1);
if canada<0
    disp("Canada:GDP->stock")
else
    disp("Canada:stock->GDP")
end

%% China
x = [China_GDP;China_GDP];
y = [China_stock;China_stock];
china = igci(x,y,2,1);
if china<0
    disp("China:GDP->stock")
else
    disp("China:stock->GDP")
end

%% Germany
x = [Germany_GDP;Germany_GDP];
y = [Germany_stock;Germany_stock];
germany = igci(x,y,2,1);
if germany<0
    disp("Germany:GDP->stock")
else
    disp("Germany:stock->GDP")
end

%% India
x = [India_GDP;India_GDP];
y = [India_stock;India_stock];
india = igci(x,y,2,1);
if india<0
    disp("India:GDP->stock")
else
    disp("India:stock->GDP")
end

%% UnitedKingdom
x = [UnitedKingdom_GDP;UnitedKingdom_GDP];
y = [UnitedKingdom_stock;UnitedKingdom_stock];
unitedKingdom = igci(x,y,2,1);
if unitedKingdom<0
    disp("UnitedKingdom:GDP->stock")
else
    disp("UnitedKingdom:stock->GDP")
end

%% UnitedStates
x = [UnitedStates_GDP;UnitedStates_GDP];
y = [UnitedStates_stock;UnitedStates_stock];
unitedStates = igci(x,y,2,1);
if unitedStates<0
    disp("UnitedStates:GDP->stock")
else
    disp("UnitedStates:stock->GDP")
end
