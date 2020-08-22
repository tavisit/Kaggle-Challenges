# Titanic_Analysis

This repository is my first big project in Python and the first time I am around Data Analysis, so I tackled the classic [BIGMart Problem](https://www.kaggle.com/brijbhushannanda1979/bigmart-sales-data) on Kaggle using a 2 part program. This challenge was originally proposed by [AnalyticsVidhya](https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/) and I scored in the top 6% of all participants as I write this document:
<a data-flickr-embed="true" href="https://www.flickr.com/photos/189039256@N05/50253733028/in/dateposted-public/" title="HighScore"><img src="https://live.staticflickr.com/65535/50253733028_7086289f20_o.png" width="1000" height="700" alt="HighScore"></a>

### Nota Bene
This is my first attempt as a machine learning and data analysis practitioner.
If you have a question about the code or the hypotheses I made, do not hesitate to post a comment in the comment section below.
If you also have a suggestion on how this notebook could be improved, please reach out to me.

### Dependencies:
* [NumPy](https://numpy.org/)
* [Searborn](https://seaborn.pydata.org/)
* [Pandas](https://pandas.pydata.org/)
* [SciKit-Learn](https://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [Matplotlib](https://matplotlib.org/)

# Get arguments

The program is built according to some general rules and it uses the command line arguments in order to ensure a more automated usage. The arguments are as follows:

Arguments:
If we write in the cmd the following line: python BigMart.py -h, the help list will appear in the following order:
```
Start the program in the following order, where [-c] is optional
python BigMart.py [-a opt]

-a / --analysis      | run the data analysis algorithm
                     | options:
                     | s -> save plots
                     | d -> display plots
                     | sd -> save and display plots
```

The command line influences the workflow of the algorithm with the following interpretation:

```Python
if d_analysis_option == 's':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
if d_analysis_option == 'd':
    plt.show()
if d_analysis_option == 'sd':
    plt.savefig('Images/' + str(nr_crt) + '_plot.png')
    nr_crt += 1
    plt.show()
```

# Data analysis

The Data Analysis part was done using the libraries offered by Numpy, Pandas, Seaborn and Matplotlib. These are basic plots, in order to understand better the data and build a complete image around the BigMart problem. All the plots are saved in the subfolder '/Images'.

All the functions are located in the module ```DataAnalysisModule.py``` in order to ensure a clean code.

### Step 1: Importing the Relevant Libraries

```Python
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
```
### Step 2: Get the general information

Check the information from data frame, check the unique values and the columns types:

```Python
def print_composition_dataframe(data_frame):
    print(data_frame.info())
    print('\n\n')
    for i in range(11):
        out_string = '{:>30}: {:>5}, with type: {}'.format(data_frame.columns[i],
                                                           str(data_frame[data_frame.columns[i]].nunique()),
                                                           type(data_frame[data_frame.columns[i]][0]))
        print(out_string)
```
Fill the data with mean values or 0 values, in order to eliminate the null components

```Python
def fill_data_set(data_frame, feature):

    if data_frame[feature].dtype == 'object':
        data_frame[feature] = data_frame[feature].fillna(data_frame[feature].mode()[0])
    else:
        data_frame[feature] = data_frame[feature].fillna(data_frame[feature].mean())

    return data_frame
```

In the ```BigMart.py```, the workflow is the following:
```Python
data_analysis.print_composition_dataframe(train_df)

# fill in the null data
train_df = data_analysis.fill_data_set(train_df, 'Item_Weight')
train_df = data_analysis.fill_data_set(train_df, 'Outlet_Size')

test_df = data_analysis.fill_data_set(test_df, 'Item_Weight')
test_df = data_analysis.fill_data_set(test_df, 'Outlet_Size')
```
```
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype
---  ------                     --------------  -----
 0   Item_Identifier            8523 non-null   object
 1   Item_Weight                7060 non-null   float64
 2   Item_Fat_Content           8523 non-null   object
 3   Item_Visibility            8523 non-null   float64
 4   Item_Type                  8523 non-null   object
 5   Item_MRP                   8523 non-null   float64
 6   Outlet_Identifier          8523 non-null   object
 7   Outlet_Establishment_Year  8523 non-null   int64
 8   Outlet_Size                6113 non-null   object
 9   Outlet_Location_Type       8523 non-null   object
 10  Outlet_Type                8523 non-null   object
 11  Item_Outlet_Sales          8523 non-null   float64
 
               Item_Identifier:  1559, with type: <class 'str'>
                   Item_Weight:   415, with type: <class 'numpy.float64'>
              Item_Fat_Content:     5, with type: <class 'str'>
               Item_Visibility:  7880, with type: <class 'numpy.float64'>
                     Item_Type:    16, with type: <class 'str'>
                      Item_MRP:  5938, with type: <class 'numpy.float64'>
             Outlet_Identifier:    10, with type: <class 'str'>
     Outlet_Establishment_Year:     9, with type: <class 'numpy.int64'>
                   Outlet_Size:     3, with type: <class 'str'>
          Outlet_Location_Type:     3, with type: <class 'str'>
                   Outlet_Type:     4, with type: <class 'str'>
```

After that, plot the general information of the data frame and see the results:
```python
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
```
<a data-flickr-embed="true" href="https://www.flickr.com/photos/189039256@N05/50253733123/in/dateposted-public/" title="0_plot"><img src="https://live.staticflickr.com/65535/50253733123_94e1f1f99a_o.png" width="2000" height="500" alt="0_plot"></a>

For further information of the distribution of prices, the algorithm uses the ```general_information_plotting``` function from the ```DataAnalysisModule``` as following:
```python
def general_information_plotting(data_frame, feature):
    print('General information about the data \n')
    print(data_frame.describe(), '\n\n')

    try:
        print(feature + ' of the data \n')
        print(data_frame[feature].describe(), '\n\n')
    except:
        print('#' * 100, '\n\nWrong data frame format')
        exit(0)

    fig = plt.figure(figsize=(20, 5))
    plt.suptitle(feature + 'Analysis')
    fig.add_subplot(121)
    subtitle = 'Skew is ' + str(data_frame[feature].skew())
    plt.title(subtitle)
    plt.hist(feature, data=data_frame)

    fig.add_subplot(122)
    subtitle = 'Skew after logarithm is ' + str(np.log(data_frame[feature]).skew())
    plt.title(subtitle)
    plt.hist(np.log(data_frame[feature]))

    return plt
```
```python
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
```
<a data-flickr-embed="true" href="https://www.flickr.com/photos/189039256@N05/50253733078/in/dateposted-public/" title="1_plot"><img src="https://live.staticflickr.com/65535/50253733078_cfeb4d22ac_o.png" width="1000" height="250" alt="1_plot"></a>

### Step 3: Data cleaning

We've seen the general information about the data, now a cleaning of the data set is required:
```python
def label_encoder_array(data_frame, array_features):
    le = preprocessing.LabelEncoder()
    for i in array_features:
        data_frame[i] = le.fit_transform(data_frame[i])

    return data_frame
```
```python
array_features = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Type']
train_df = data_analysis.label_encoder_array(train_df, array_features)
test_df = data_analysis.label_encoder_array(test_df, array_features)
```

We need to see the most important features that influence the price, in order to see what to drop from the data frame
```python
def feature_ranking_plotting(data_frame, correlation, size_ranking):

    print('Features that we care about \n')
    data_frame_numeric_features = data_frame.select_dtypes(include=[np.number])
    data_frame_corr = data_frame_numeric_features.corr()
    
    # take the first 5 best increasing factors and
    # the best indicators of the decrease of the final feature
    data_frame_pos_features = data_frame_corr[correlation].sort_values(ascending=False)[1:size_ranking + 1]
    data_frame_neg_features = data_frame_corr[correlation].sort_values(ascending=False)[-size_ranking:]
    print(data_frame_pos_features, '\n')
    print(data_frame_neg_features, '\n\n')

    # Form the plots here
    # ...
    
    return plt
```
```python
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
```

<a data-flickr-embed="true" href="https://www.flickr.com/photos/189039256@N05/50253733053/in/dateposted-public/" title="2_plot"><img src="https://live.staticflickr.com/65535/50253733053_284ff88b80_o.png" width="1000" height="500" alt="2_plot"></a>

```
Features that we care about

Item_MRP                0.567574
Outlet_Type             0.401522
Outlet_Location_Type    0.089367
Item_Type               0.017048
Item_Weight             0.011550
Name: Item_Outlet_Sales, dtype: float64

Item_Weight                  0.011550
Item_Fat_Content             0.009800
Outlet_Establishment_Year   -0.049135
Outlet_Size                 -0.086182
Item_Visibility             -0.128625
Name: Item_Outlet_Sales, dtype: float64
```

So, only the labels of the outlet and the product have to go, because only these don't influence the prices( we are lucky, because we don't have to engineer some new features)

# Building the ML Model

I chose to use a mix of Linear Regression Model and Random Forest Model in order to ensure a good distribution of values minimize the error.

### Step 1: Test the model on the train set

To ensure that the models work, we will train and test the models with the train set as it follows:

```python
# prepare the data for ML
X = train_df.drop(columns="Item_Outlet_Sales")
X = X.drop(columns="Outlet_Identifier")
X = X.drop(columns="Item_Identifier")
y = train_df["Item_Outlet_Sales"]

X_train, X_test = X, X
y_train, y_test = y, y
```

#### Make the LR model:
```python
# Linear Regression model:
L_R = LinearRegression()
L_R.fit(X_train, y_train)
pred_lr = L_R.predict(X_test)
print("\nLinear Regression model: ")
MAE = metrics.mean_absolute_error(y_test, pred_lr)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, pred_lr)
print('MSE score: ' + str(MSE))
```
with the scores of:
```
Linear Regression model:
MAE score: 899.0848111990341
MSE score: 1432944.4712958355
```
#### Make the RFR model:
```python
# Random Forest Regressor model:
rfr_best = RandomForestRegressor(n_estimators=100)
rfr_best.fit(X_train, y_train)
pred_rf = rfr_best.predict(X_test)
print("\nRandom Forest Regressor model: ")
MAE = metrics.mean_absolute_error(y_test, pred_rf)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, pred_rf)
print('MSE score: ' + str(MSE))
```
with the scores of:
```
Random Forest Regressor model:
MAE score: 293.7720292148304
MSE score: 181048.73818653874
```
#### Make the mixed model:
```python
# Combined model
final_pred = pred_lr * 0.2 + pred_rf * 0.8
print("\nCombined model: ")
MAE = metrics.mean_absolute_error(y_test, final_pred)
print('MAE score: ' + str(MAE))
MSE = metrics.mean_squared_error(y_test, final_pred)
print('MSE score: ' + str(MSE))
```
with the scores of:
```
Combined model:
MAE score: 397.91318230724977
MSE score: 308445.54388683493
```
It was clear that the Random Forest predicts better, but we need a slight deviation in order to ensure that one model doesn't make the too many mistakes.

In order to visualize the results, the following code will help:

```python
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
```
<a data-flickr-embed="true" href="https://www.flickr.com/photos/189039256@N05/50253733043/in/dateposted-public/" title="3_plot"><img src="https://live.staticflickr.com/65535/50253733043_21d279b36d_o.png" width="1000" height="1000" alt="3_plot"></a>

### Step 2: Make the actual predictions

Now, that we know that the ML algorithms work, it is time to apply these routines on the actual test set:
```python
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

# some values are negative, so we apply the 
# absolute value function, in order to make them possible
final_pred = abs(final_pred)
```
Form the data frame and save the results:
```python
final_df = pd.DataFrame()
final_df['Item_Identifier'] = test_df['Item_Identifier']
final_df['Outlet_Identifier'] = test_df['Outlet_Identifier']
final_df['Item_Outlet_Sales']=  final_pred
final_df.to_csv('save.csv', index=0)
```
# Conclusion

The algorithms used in this solution are not the best, but show a in-depth data visualisation and analysis on the hypothesi.

This kind of solution, focused on the data analysis, rather than on machine learning, enables me to process and visualize the complicated structure of the data. However, a more ML-driven algorithms may become the norm in the near future, when the knowledge on this topic becomes more embedded in my mind and the tricks and shortcuts of these methods are clearer
