# The normal imports for the data management
import pandas as pd
import numpy as np

# These are for the ML algorithm
from scipy.stats import skew
# Random Forest import
from sklearn.ensemble import RandomForestClassifier


#
class Machine_Learning_Class:
    # Machine Learning section of the program, which
    # uses the Random forest algorithm
    def machine_learning(self, titanic_DFrame, test_DFrame):
        # clean the data, by dropping the name which doesn't give any practical advantage over the ML
        titanic_DFrame_temp = titanic_DFrame.drop(['Name'], axis=1)
        test_DFrame_temp = test_DFrame.drop(['Name'], axis=1)

        # concatenate the DataFrames in order to have a clear image
        # concatenate all the useful data from the Class to the Embarked location
        ml_data = pd.concat(
            (titanic_DFrame_temp.loc[:, 'Pclass':'Embarked'], test_DFrame_temp.loc[:, 'Pclass':'Embarked']))

        # in order to normalize the data, transform the skewed numeric features by taking log(feature + 1)
        numeric_feats = ml_data.dtypes[ml_data.dtypes != "object"].index

        skewed_feats = titanic_DFrame_temp[numeric_feats].apply(lambda x: skew(x.dropna()))
        skewed_feats = skewed_feats[skewed_feats > 0.75]
        skewed_feats = skewed_feats.index

        ml_data[skewed_feats] = np.log1p(ml_data[skewed_feats])
        ml_data = pd.get_dummies(ml_data)

        # filling NaNs with the mean of the column,
        # in order to be able to apply mathematical functions on it
        ml_data = ml_data.fillna(ml_data.mean())

        # the actual random forest algorithm
        while True:
            try:
                estimators = int(input("How many estimators do you want to be in the Classifier: "))
                break
            except:
                print("Invalid, try again!")

        try:
            random_forest = RandomForestClassifier(n_estimators=estimators)
            random_forest.fit(ml_data[:titanic_DFrame_temp.shape[0]], titanic_DFrame_temp.Survived)
            random_forest_prediction = random_forest.predict(ml_data[titanic_DFrame_temp.shape[0]:])
            random_forest.score(ml_data[:titanic_DFrame_temp.shape[0]], titanic_DFrame_temp.Survived)

            # build a solution and create a csv with the 'PassengerID' and 'Survived' as columns As the Kaggle.com,
            # Titanic: Machine Learning from Disaster, problem requires the DF has only two columns,
            # an ID and Survived prediction, but it can be expanded to show more information
            solution_DFrame = pd.DataFrame(
                {"PassengerId": test_DFrame_temp.PassengerId, "Survived": random_forest_prediction})
            solution_DFrame.to_csv("solution_ML.csv", index=False)

            # print the description of the DFs involved in this part
            print("The titanic DF as a training set: \n")
            print(titanic_DFrame_temp.describe())
            print("The titanic DF as a test set: \n")
            print(test_DFrame_temp.describe())
            print("The solution set: \n")
            print(solution_DFrame.describe())

        except:
            print("The process wasn't completed with success, try again!")

        # free the temporary variables and objects
        del titanic_DFrame_temp
        del ml_data
        del test_DFrame_temp
        del random_forest
        del random_forest_prediction
        del solution_DFrame
