# imports
import getopt
import os
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib
import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics

# These are the plotting modules adn libraries we'll use:
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
import pandas as pd
import numpy as np

# These are the plotting modules adn libraries we'll use:
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

######################################################################################################################
# get console arguments
######################################################################################################################
def get_arguments():
    # default settings
    d_analysis = False
    d_analysis_option = ''

    full_cmd_arguments = sys.argv

    argument_list = full_cmd_arguments[1:]

    short_options = 'ha:'
    long_options = ['help', 'analysis=']

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        # Output error, and return with an error code
        print(str(err))
        sys.exit(2)

    return_list = []
    # parse the arguments
    for current_argument, current_value in arguments:
        if current_argument in ('-h', '--help'):  # help argument is parsed and display information
            print('\n\nStart the program in the following order, where [-c] is optional')
            print('python BigMart.py [-a opt]\n')
            print('-a / --analysis      | run the data analysis algorithm\n',
                  ' ' * 19, '| options:\n',
                  ' ' * 19, '| s -> save plots\n',
                  ' ' * 19, '| d -> display plots\n',
                  ' ' * 19, '| sd -> save and display plots')
            exit(0)
        elif current_argument in ('-a', '--analysis'):  # user argument is parsed and set the boolean variable
            d_analysis = True
            d_analysis_option = current_value
            if d_analysis_option == 's' or d_analysis_option == 'sd':
                # create the folder if it doesn't exist
                if not os.path.exists('Images'):  # if the folder 'Images' doesn't exist, create one
                    os.makedirs('Images')
            else:
                sys.exit(2)

    return_list.append(d_analysis)
    return_list.append(d_analysis_option)
    return return_list


# console arguments
d_analysis, d_analysis_option = get_arguments()
# plot saved index
nr_crt = 0

######################################################################################################################
# read the data sets
######################################################################################################################
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')
data_analysis = DataAnalysisClass()

######################################################################################################################
# see the composition of the data frame
######################################################################################################################

data_analysis.print_composition_dataframe(train_df)

# fill in the null data
train_df = data_analysis.fill_data_set(train_df, 'Item_Weight')
train_df = data_analysis.fill_data_set(train_df, 'Outlet_Size')

test_df = data_analysis.fill_data_set(test_df, 'Item_Weight')
test_df = data_analysis.fill_data_set(test_df, 'Outlet_Size')

# plot some general information about the data frame
fig = plt.figure(figsize=(20, 10))
plt.suptitle('General Information in correlation to Sale price')
fig.add_subplot(221)
plt.title('Item_MRP influence')
plt.scatter(train_df['Item_MRP'], train_df['Item_Outlet_Sales'])

fig.add_subplot(222)
plt.title('Outlet_Type influence')
plt.scatter(train_df['Outlet_Type'], train_df['Item_Outlet_Sales'])

fig.add_subplot(223)
plt.title('Item_Visibility influence')
plt.scatter(train_df['Item_Visibility'], train_df['Item_Outlet_Sales'])

fig.add_subplot(224)
plt.title('Item_Weight influence')
plt.scatter(train_df['Item_Weight'], train_df['Item_Outlet_Sales'])
if d_analysis_option == 's':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
if d_analysis_option == 'd':
    plt.show()
if d_analysis_option == 'sd':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
    plt.show()

# show the distribution of the prices
plt = data_analysis.general_information_plotting(train_df, 'Item_Outlet_Sales')
if d_analysis_option == 's':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
if d_analysis_option == 'd':
    plt.show()
if d_analysis_option == 'sd':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
    plt.show()

array_features = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type']
train_df = data_analysis.label_encoder_array(train_df, array_features)
test_df = data_analysis.label_encoder_array(test_df, array_features)

print("\n\n")
data_analysis.print_composition_dataframe(train_df)
print(train_df.info())

# see the most important features that influnce the price
plt = data_analysis.feature_ranking_plotting(train_df, 'Item_Outlet_Sales', 5)
if d_analysis_option == 's':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
if d_analysis_option == 'd':
    plt.show()
if d_analysis_option == 'sd':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
    plt.show()


######################################################################################################################
# test the ML machine on the train_df itself to make the best algorithm
######################################################################################################################
# prepare the data for ML
X = train_df.drop(columns="Item_Outlet_Sales")
X = X.drop(columns="Outlet_Identifier")
X = X.drop(columns="Item_Identifier")
y = train_df["Item_Outlet_Sales"]

X_train, X_test = X, X
y_train, y_test = y, y

# Linear Regression model:
L_R = LinearRegression()
L_R.fit(X_train, y_train)
pred_lr = L_R.predict(X_test)
print("\nLinear Regression model: ")
MAE = metrics.mean_absolute_error(y_test, pred_lr)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, pred_lr)
print('MSE score: ' + str(MSE))

# Random Forest Regressor model:
rfr_best = RandomForestRegressor(n_estimators=100)
rfr_best.fit(X_train, y_train)
pred_rf = rfr_best.predict(X_test)
print("\nRandom Forest Regressor model: ")
MAE = metrics.mean_absolute_error(y_test, pred_rf)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, pred_rf)
print('MSE score: ' + str(MSE))

final_pred = pred_lr * 0.2 + pred_rf * 0.8
print("\nCombined model: ")
MAE = metrics.mean_absolute_error(y_test, final_pred)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, final_pred)
print('MSE score: ' + str(MSE))

plt.figure(figsize=(10, 10))
plt.suptitle("Accuracy of the ML")
plt.scatter(y_test, final_pred)
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
if d_analysis_option == 's':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
if d_analysis_option == 'd':
    plt.show()
if d_analysis_option == 'sd':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
    plt.show()

######################################################################################################################
# Actual ML algorithm
######################################################################################################################
X = train_df.drop(columns="Item_Outlet_Sales")
X = X.drop(columns="Outlet_Identifier")
X = X.drop(columns="Item_Identifier")
y = train_df["Item_Outlet_Sales"]

X_train = X
X_test = test_df.drop(columns="Outlet_Identifier")
X_test = X_test.drop(columns="Item_Identifier")
y_train = y

# Linear Regression model:
L_R = LinearRegression()
L_R.fit(X_train, y_train)
pred_lr = L_R.predict(X_test)

# Random Forest Regressor model:
rfr_best = RandomForestRegressor(n_estimators=1200)
rfr_best.fit(X_train, y_train)
pred_rf = rfr_best.predict(X_test)

final_pred = pred_lr * 0.2 + pred_rf * 0.8
final_pred = abs(final_pred)
final_df = pd.DataFrame()
final_df['Item_Identifier'] = test_df['Item_Identifier']
final_df['Outlet_Identifier'] = test_df['Outlet_Identifier']
final_df['Item_Outlet_Sales']=  final_pred
final_df.to_csv('save.csv', index=0)
