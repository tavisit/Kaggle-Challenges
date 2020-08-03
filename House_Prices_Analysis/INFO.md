# House Price Analysis

This solution comes from the problem of house prices predictions of the classic problem on [kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). Bellow There will be explained my working procedure and some insights into the data analysis and machine learning techniques that I employed.

### Nota Bene
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
the defaults are taken from the main script, usually they are:
* automatic = True         |  No input is expected from the user
* display_plot = False     |  Don't display the plots
* save_plots = False       |  Don't save the plots
* dml = False              |  Don't compute the machine learning on the train data to test
* input_file = ''          |  Input file path is set to null and will raise an exception if parsed this way

If we write in the cmd the following line: python Houses_Prices.py -h, the help list will appear in the following order:
```
Start the program in the following order, where [-c] is optional
python Houses_prices.py [-u] [-d] [-s] -i location_input

-u / --automatic | user interaction with the cmd is required
-d / --display   | display the plots
-s / --save      | save the plots in local /Images
-m / --save      | show the result of ML Algorithm performing with itself
-i / --input     | input of the data frame
```

First, we get the arguments from the command line:
```
if __name__ == '__main__':
    automatic, display_plot, save_plots, machine_learning, input_file = get_arguments(automatic, display_plot,
                                                                                      save_plots, machine_learning,
                                                                                      input_file)
```

The command line influences the workflow of the algorithm with the following interpretation:
```Python
if save_plots == True:
    plt.savefig('Images/' + str(counter) + '.png')
    counter += 1
if automatic == False:
    input('\n\nPress "Enter" to see the plot\n\n')
if display_plot == True:
    plt.show()
```

For further information, please open the get_arguments.py
# Data analysis

The Data Analysis part was done using the libraries offered by Numpy, Pandas, Seaborn and Matplotlib. These are basic plots, in order to understand better the data and build a complete image around the Titanic problem. This part is merged into a single file, named 'DataAnalysisModule.py', with a couple of basic functions and a class, 'DataAnalysisClass' which encapsulates some useful methods. Each function tries to calculate and show a certain basic functionality
The main data frame that is processed is called train_df , which is read from a .CSV from the same folder. In this case we talk about the 'train.csv' in this case, but the line actually is: ```train_df = pd.read_csv(input_file)```
### General Information
We build a plot of general information about the houses with the method from the DataAnalysisModule in the following manner:
```
plt = data_analysis_module.general_information_plotting(train_df, 'SalePrice')
```
plotting the dependency of price to the number of properties, with the normal distribution and a skewed one:

![Plot1](https://live.staticflickr.com/65535/50180042196_ae6d890987_k.jpg)


### Ranking of the features
To be able to interpret and implement a machine learning algorithm, there needs to be made a ranking of the current features to see the redundant ones( be aware, we are intersted in the first 10 positive features and top 10 negative features regarding the sell price):
```Python
plt = data_analysis_module.feature_ranking_plotting(train_df, 'SalePrice', 10)
```
This function plots the following schematic:

![Plot2](https://live.staticflickr.com/65535/50179506428_5072cd3ca0_b.jpg)


### Overall Quality, in depth analysis
After this, a more in depth analysis is required to be able to familiarize with the data flow:
```
print(train_df[['OverallQual', 'GrLivArea', 'SalePrice']].describe())
# show the relation with the logarithmic SalePrice, in order not to expand too much the plots
plt.scatter(x=train_df['OverallQual'], y=np.log(train_df.SalePrice))
plt.scatter(x=train_df['GrLivArea'], y=np.log(train_df.SalePrice))
```

![Plot3](https://live.staticflickr.com/65535/50180042131_67e9678437_b.jpg)


### Clean data and add new Features
In the data set, there might be null and '0' features, so, in order to feed the ML. A plot is drawn with the number of null elements and the feature they belong to:
```Python
plt = data_analysis_module.null_distribution_plotting(train_df)
```

![Plot4](https://live.staticflickr.com/65535/50179506343_9ef38abb46_o.png)

### Replace/ rename/ delete/ group the columns
A simple function that replaces and groups the columns, dealing with the null indicators:
```Python
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
```
Because the data frame has three new columns('TotalSF','TotalPorchSF','TotalBath'), it needs to plot the relation to be able to tell if the algorithm is more efficient that way:
```Python
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
```
This new information draws the following plots:

![Plot5](https://live.staticflickr.com/65535/50180295832_70099eb58f_o.png)

And to be able to be sure that the new features are relatible, the ranking of features is plotted again:

![Plot6](https://live.staticflickr.com/65535/50179506318_1fc67333d1_o.png)


### Overall Quality, in depth analysis of the new features

The following plots share some information about the relations between the Total Surface Area, Porch Surface Area and overall quality of the houses. It is used a ```sns.joinplot``` in order to represent the 2D relation of two elements:
```Python
sns.jointplot(train_df['TotalPorchSF'], train_df['TotalSF'], train_df,
              kind='kde', xlim=(-10, 400), ylim=(0, 4200))
```
<img src="https://live.staticflickr.com/65535/50180041951_aab0aec5ea_o.png" width="450"><img src="https://live.staticflickr.com/65535/50180295817_12fd2cd7d3_o.png" width="450">

<img src="https://live.staticflickr.com/65535/50179506243_0e248037c1_o.png" width="450"><img src="https://live.staticflickr.com/65535/50180041931_e665cac852_o.png" width="450">

# Machine Learning
The module has only one class, 'MachineLearningModule', which contains all the algorithms and information needed to perform the Machine Learning Task.
The main method needs 5 elements:
* the main data frame, train_df
* test data frame, which is read from the 'test.csv'.
* data concatenated from the two, named ml_data
* the column name that the algorithm needs to work out
* the number of estimators for the random forest

I chose to work with [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest), mainly because it is the only ML Algorithm that I know so well that I can build such prediction.

The algorithm is split between two parts, an optional data comparison random forest and the actual solution builder.

### Optional ML analysis

The optional part is activated with the command line ```-m``` from machine and will plot a series of dependencies. This works by taking the train data frame as test, predict the output and then plot the relation with the actual data set. This is done to fine tune the machine and make better approximations:
```Python
n_estimators = [1, 50, 500, 1000]
fig = plt.figure(figsize=(20, 10))
plt.suptitle('Test the Algorithm with the train data and plot against itself')
for crt, i in enumerate(n_estimators):
    subplot_number = 220 + crt + 1
    plt.subplot(subplot_number).set_title('Random Forest with ' + str(i) + ' estimators')
    
    #model is the prediction data set
    model = machine_module.machine_learning(train_df, train_df, ml_data, 'SalePrice', i)
    
    plt.scatter(train_df['SalePrice'], model['SalePrice'])
    overlay = 'RMSE = ' + str(mean_squared_error(train_df['SalePrice'], model['SalePrice']))
    plt.xlabel(overlay)
    print('Subplot ' + str(crt + 1) + ' drawn')
```
This algorithm produces the following data plot:
<img src="https://live.staticflickr.com/65535/50180295707_57f8651f6c_o.png" width="1000">

The plot seems to output only good results, so no further tunning is required( this will prove to be wrong, but the 3/4 correct predictions are a mistery to me right now)

### Solution maker of the Machine Learning

The two data frames are concatenated into one from the MSSubClass to the end of the csv, so that one may use easily the information embedded within. In this step, we also get the dummies for the *ml_data*:
```Python
ml_data = pd.concat(
    (train_df.loc[:, 'MSSubClass':], test_df.loc[:, 'MSSubClass':]))

ml_data = pd.get_dummies(ml_data)
```
ml_data will be filled with mean values in order to ensure that the data frame has no nan or 0 values. In the end, make the prediction on 2000 estimators and plot the results:
```Python
# filling NaNs with the mean of the column,
# in order to be able to apply mathematical functions on it
ml_data = ml_data.fillna(ml_data.mean())
model = machine_module.machine_learning(train_df, test_df, ml_data, 'SalePrice', 2000)
fig = plt.figure(figsize=(10, 10))
plt.suptitle('Predictions')
plt.hist(model['SalePrice'])
plt.xlabel('Price')
plt.ylabel('Number of Properties')
plt.yticks(rotation=90)
```
<img src="https://live.staticflickr.com/65535/50179506193_a3285434c6_o.png" width="1000">
#### Further explanation of the machine_module.machine_learning
Considering we have a forest, the number of estimators are from the standard input and there are used mainstream methods to build the prediction and to form a score.
```Python
random_forest = RandomForestClassifier(n_estimators=estimators)
random_forest.fit(all_data[:train_data.shape[0]], train_data[feature])
random_forest_prediction = random_forest.predict(all_data[train_data.shape[0]:])
random_forest.score(all_data[:train_data.shape[0]], train_data[feature])
```
Build a solution and create a csv with the 'Id' and 'SalePrice' columns and print a description of the solution
```Python
solution_DFrame = pd.DataFrame({"Id": test_data.Id, feature: random_forest_prediction})
solution_DFrame.to_csv("solution_ML.csv", index=False)

print("The solution data form set: \n")
print(solution_DFrame.describe())
```
The final results are displayed in the command prompt and saved as a .CSV with the name 'solution_ML.csv'.

# Conclusion
The algorithms used in this solution are not the best, but show a in-depth data visualisation and analysis on the house prices. The information can be used to predict and able someone to actually buy a property in a normalized house market.

This kind of solution, focused on the data analysis, rather than on machine learning, enables me to process and visualize the complicated structure of the data. However, a more ML-driven algorithms may become the norm in the near future, when the knowledge on this topic becomes more embedded in my mind and the tricks and shortcuts of these methods are clearer
