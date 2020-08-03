# The normal imports for the data management
import matplotlib
import pandas as pd
import numpy as np

# These are the plotting modules adn libraries we'll use:
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import MachineLearningModule

from get_arguments import get_arguments

import DataAnalysisModule


# restructure the data frame with more important data
def replace_data_frame(data_frame):
    cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageCond', 'GarageQual',
            'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']
    data_frame[cols] = data_frame[cols].fillna('None')
    cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1',
            'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    data_frame[cols] = data_frame[cols].fillna(0)
    cols = ['MasVnrType', 'MSZoning', 'Utilities', 'Exterior1st',
            'Exterior2nd', 'SaleType', 'Electrical', 'KitchenQual', 'Functional']
    data_frame[cols] = data_frame.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.mode()[0]))
    cols = ['GarageArea', 'LotFrontage']
    data_frame[cols] = data_frame.groupby('Neighborhood')[cols].transform(lambda x: x.fillna(x.median()))
    cols = ['MSSubClass', 'YrSold']
    data_frame[cols] = data_frame[cols].astype('category')
    print(data_frame.describe())
    # engineer new features
    data_frame['TotalSF'] = data_frame['GrLivArea'] + data_frame['TotalBsmtSF']
    data_frame['TotalPorchSF'] = data_frame['OpenPorchSF'] + data_frame['EnclosedPorch'] + \
                                 data_frame['3SsnPorch'] + data_frame['ScreenPorch']
    data_frame['TotalBath'] = data_frame['FullBath'] + data_frame['BsmtFullBath'] + \
                              0.5 * (data_frame['BsmtHalfBath'] + data_frame['HalfBath'])
    return data_frame


automatic = True
display_plot = False
save_plots = False
machine_learning = False
input_file = ''

if __name__ == '__main__':
    automatic, display_plot, save_plots, machine_learning, input_file = get_arguments(automatic, display_plot,
                                                                                      save_plots, machine_learning,
                                                                                      input_file)

# counter represents the number of the figure in the /Images folder
counter = 1
train_df = pd.read_csv(input_file)

aux_df = train_df  # save the data frame for later use

########################################################################################
########################################################################################
# General information and data analysis
print('#' * 100)
print('#' * 100)

data_analysis_module = DataAnalysisModule.DataAnalysisClass()
plt = data_analysis_module.general_information_plotting(train_df, 'SalePrice')

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
    if display_plot == False:
        plt.clf()
        plt.close()
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()

########################################################################################
########################################################################################
# Decide what to drop by making a numeric features variable and
# making a correlation table to see what influences the data
# new features
print('#' * 100)
print('#' * 100)

plt = data_analysis_module.feature_ranking_plotting(train_df, 'SalePrice', 10)

# we see that garage cars and garage area are the same, so we drop one
# and double the other one in order to retain the importance
train_df = train_df.drop(['GarageCars'], axis=1)

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()

########################################################################################
########################################################################################
# Overall analysis of the data
print('#' * 100)
print('#' * 100)
print("The overall Analysis of the Quality and Sell price")
print(train_df[['OverallQual', 'GrLivArea', 'SalePrice']].describe())
fig = plt.figure(figsize=(20, 5))
plt.suptitle('Overall Analysis')
fig.add_subplot(121)
plt.scatter(x=train_df['OverallQual'], y=np.log(train_df.SalePrice))
plt.ylabel('Sale Price (Log 10)')
plt.xlabel('Overall Quality')

fig.add_subplot(122)
plt.scatter(x=train_df['GrLivArea'], y=np.log(train_df.SalePrice))
plt.ylabel('Sale Price (Log 10)')
plt.xlabel('Living area')

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()

########################################################################################
########################################################################################
# clean data and add new Features
# See what to change or delete
print('#' * 100)
print('#' * 100)

plt = data_analysis_module.null_distribution_plotting(train_df)

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()

########################################################################################
########################################################################################
# replace/ rename/ delete/ group the columns
train_df = replace_data_frame(train_df)

########################################################################################
########################################################################################
# Overall analysis of the data
print('#' * 100)
print('#' * 100)
cols = ['SalePrice', 'OverallQual', 'GarageArea', 'GrLivArea', 'OpenPorchSF', 'FullBath']
print('The distribution of new features made for this data frame')
print('The old one: ')
print(aux_df[cols].describe())
print('The new one: ')
cols.append('Neighborhood')
cols.append('TotalSF')
cols.append('TotalPorchSF')
cols.append('TotalBath')
print(train_df[cols].describe())

# actual plotting
fig = plt.figure(figsize=(20, 5))
plt.suptitle('New Features')
fig.add_subplot(121)
sns.distplot(train_df['TotalSF'], bins=100, color='red', )
plt.xlabel('Total Surface Area')
plt.ylabel('Distribution')

fig.add_subplot(122)
sns.distplot(train_df['TotalPorchSF'], bins=100, color='blue')
plt.xlabel('Total Porch Area')
plt.ylabel('Distribution')

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()
########################################################################################
########################################################################################
# Decide what to drop by making a numeric features variable and
# making a correlation table to see what influences the data
# new features
print('#' * 100)
print('#' * 100)

plt = data_analysis_module.feature_ranking_plotting(train_df, 'SalePrice', 10)

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()
########################################################################################
########################################################################################
# Join Plot having Total Porch and Total Surface
print('#' * 100)
print('#' * 100)
print('It continues with an analysis on the total surface area, total porch area and overall quality \n\n')
print(train_df[['TotalSF', 'TotalPorchSF', 'OverallQual']].describe())

# Total Surface Area and Total Porch Surface Area
sns.jointplot(train_df['TotalPorchSF'], train_df['TotalSF'], train_df,
              kind='kde', xlim=(-10, 400), ylim=(0, 4200))
if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1

sns.jointplot(train_df['TotalPorchSF'], train_df['TotalSF'], train_df,
              kind='hex', xlim=(-10, 400), ylim=(0, 4200))
if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1

if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    print('\n\nThe Plot 1 and 2 are related to the Total Porch Surface Area in relation to the Total Surface Area')
    plt.show()

# Overall Quality and Total Surface
sns.jointplot(train_df['OverallQual'], train_df['TotalSF'], train_df,
              kind='kde', ylim=(0, 4200))
if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
sns.jointplot(train_df['OverallQual'], train_df['TotalSF'], train_df,
              kind='hex', ylim=(0, 4200))
if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1

if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    print('\n\nThe Plots 3 and 4 are related to the Overall Quality in relation to the Total Surface Area')
    plt.show()

########################################################################################
########################################################################################
machine_module = MachineLearningModule.Machine_Learning_Class()
if machine_learning == True:
    print('#' * 100)
    print('#' * 100)
    print('Test ML algorithm\n')

    ml_data = pd.concat(
        (train_df.loc[:, 'MSSubClass':], train_df.loc[:, 'MSSubClass':]))

    ml_data = pd.get_dummies(ml_data)

    # filling NaNs with the mean of the column,
    # in order to be able to apply mathematical functions on it
    ml_data = ml_data.fillna(ml_data.mean())

    n_estimators = [1, 50, 500, 1000]
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle('Test the Algorithm with the train data and plot against itself')
    for crt, i in enumerate(n_estimators):
        subplot_number = 220 + crt + 1
        plt.subplot(subplot_number).set_title('Random Forest with ' + str(i) + ' estimators')
        model = machine_module.machine_learning(train_df, train_df, ml_data, 'SalePrice', i)
        plt.scatter(train_df['SalePrice'], model['SalePrice'])
        overlay = 'RMSE = ' + str(mean_squared_error(train_df['SalePrice'], model['SalePrice']))
        plt.xlabel(overlay)
        print('Subplot ' + str(crt + 1) + ' drawn')

    if save_plots == True:
        plt.savefig('Images/' + str(counter) + '.png')
        counter += 1
    if automatic == False:
        input('\n\nPress "Enter" to see the plot\n\n')
    if display_plot == True:
        print('\n\nThe plot of the prediction data against the train data')
        plt.show()
########################################################################################
########################################################################################
print('#' * 100)
print('#' * 100)
print('Actual ML algorithm with data\n')
test_df = pd.read_csv('test.csv')
ml_data = pd.concat(
    (train_df.loc[:, 'MSSubClass':], test_df.loc[:, 'MSSubClass':]))

ml_data = pd.get_dummies(ml_data)

# filling NaNs with the mean of the column,
# in order to be able to apply mathematical functions on it
ml_data = ml_data.fillna(ml_data.mean())
model = machine_module.machine_learning(train_df, test_df, ml_data, 'SalePrice', 500)
fig = plt.figure(figsize=(10, 10))
plt.suptitle('Predictions')
plt.hist(model['SalePrice'])
plt.xlabel('Price')
plt.ylabel('Number of Properties')
plt.yticks(rotation=90)

if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    print('\n\nThe plot of the prediction data against the train data')
    plt.show()