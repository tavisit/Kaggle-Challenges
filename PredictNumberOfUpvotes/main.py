import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import Binarizer
import matplotlib
import pandas as pd
import numpy as np

# These are the plotting modules adn libraries we'll use:
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

import DataAnalysisModule
#######################################################################################################################
#######################################################################################################################
data_analysis = DataAnalysisModule.DataAnalysisClass()
df = pd.read_csv('train_NIR5Yl1.csv')
data_analysis.print_composition_dataframe(df)
data_analysis.null_distribution_plotting(df)
data_analysis.feature_ranking_plotting(df, 'Upvotes', 5)
plt.show()
plt.scatter(df['Upvotes'],df['Views'])
plt.xlim(0,200000)
plt.show()

# feature cleaning and engineering
df = df.drop(df[df.Views > 3000000].index)
df = data_analysis.label_encoder_array(df,['Tag'])
# If the public will interact with the post according to the Answers
bn = Binarizer(threshold=7)
pd_watched = bn.transform([df['Answers']])[0]
df['Watched'] = pd_watched
#######################################################################################################################
#######################################################################################################################
# test the models
df = df.drop(columns = ['ID','Username'])
X_train = df.drop(columns = 'Upvotes')
X_test = df.drop(columns = 'Upvotes')
y_train = df['Upvotes']
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#######################################################################################################################
# Linear
lr = LinearRegression()
lr.fit(X_train,y_train)
y_lr = abs(lr.predict(X_test))
linear_rmse = mean_squared_error(y_train, y_lr, squared=False)
print('RMSE on Linear: '+str(linear_rmse))
#######################################################################################################################
# Polynomial
poly_reg = PolynomialFeatures(degree = 4,interaction_only=False, include_bias=True)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_train, y_train)
lin_reg = linear_model.LassoLars(alpha=0.021,max_iter=150)
lin_reg.fit(X_poly, y_train)
y_poly = abs(lin_reg.predict(poly_reg.fit_transform(X_test)))
poly_rmse = mean_squared_error(y_train, y_poly, squared=False)
print('RMSE on Polynomial Regression: '+str(poly_rmse))

#######################################################################################################################
# Random Forest
#rfr = RandomForestRegressor(n_estimators = 10)
#rfr.fit(X_train,y_train)
#y_rfr =abs( rfr.predict(X_test))
#rand_forest_rmse = mean_squared_error(y_train, y_lr, squared=False)
#print('RMSE on Random Forest: '+str(rand_forest_rmse))

# The best is the polynomial with the alpha = 0.019
test_df = pd.read_csv('test_8i3B3FC.csv')
print(test_df.describe())
idx = test_df['ID']
test_df = test_df.drop(columns=['ID','Username'])
test_df = data_analysis.label_encoder_array(test_df,['Tag'])
# If the public will interact with the post according to the Answers
bn = Binarizer(threshold=7)
pd_watched = bn.transform([test_df['Answers']])[0]
test_df['Watched'] = pd_watched

test_df = sc_X.fit_transform(test_df)

pred_final = lin_reg.predict(poly_reg.fit_transform(test_df))
pred_final=abs(pred_final)

final =  pd.DataFrame({'ID': idx,'Upvotes':pred_final})
final.to_csv("submission.csv",index=False)