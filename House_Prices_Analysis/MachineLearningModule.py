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
    # all_data should be already ready for machine, so preprocess
    # it accordingly
    #
    # Arguments:
    # - train_data represents the data frame from the train data set
    # - test_data represents the data frame from the test data set
    # - all_data represents the concatenated data frame of the two
    # - feature is the unknown in the test data set
    # - estimators is the number of n_estimators in the RandomForestClassifier
    #
    # return a data frame corresponding to the solution
    def machine_learning(self, train_data, test_data, all_data, feature, estimators):

        # the actual random forest algorithm

        try:
            random_forest = RandomForestClassifier(n_estimators=estimators)
            random_forest.fit(all_data[:train_data.shape[0]], train_data[feature])
            random_forest_prediction = random_forest.predict(all_data[train_data.shape[0]:])
            random_forest.score(all_data[:train_data.shape[0]], train_data[feature])
            # build a solution and create a csv with the 'PassengerID' and 'feature' as columns As the Kaggle.com,
            # Titanic: Machine Learning from Disaster, problem requires the DF has only two columns,
            # an ID and feature prediction, but it can be expanded to show more information
            solution_DFrame = pd.DataFrame(
                {"Id": test_data.Id, feature: random_forest_prediction})
            solution_DFrame.to_csv("solution_ML.csv", index=False)

            print("The solution data form set: \n")
            print(solution_DFrame.describe())
            return solution_DFrame

        except:
            print("The process wasn't completed with success, try again!")

        return None
