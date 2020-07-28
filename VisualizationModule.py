# The normal imports for the data management
import numpy as np
import pandas as pd

# These are the plotting modules adn libraries we'll use:
import matplotlib.pyplot as plt
import seaborn as sns
import math


########################################################################################################################
# Functions that are useful and needed
#
# A function to sort through the sex and produces 3 types of passengers
def male_female_child(passenger):
    # Take the Age and Sex
    age, sex = passenger
    # Compare the age, otherwise leave the sex
    if age < 18:
        return 'child'
    else:
        return sex


# Take the first letter of the cabin
# From the form LNNN(Letter Number Number Number), take only the L(letter)
def cabin_finder(cabin):
    return cabin[0][0]


# Calculate how many family members there are on board for each passenger
def alone_family(passenger):
    # Take if the passenger is with parents/children and Sibling
    siblings, family = passenger
    return siblings + family

# Determine if the passenger has family on board
def alone_family_bool(passenger_family):
    # if the passenger has family on board
    # return true
    if int(passenger_family) > 0:
        return 1
    else:
        return 0


# the main class for visualization
class VisualizationClass:
    # the following functions are the question embedded plots and information displayed
    # to the user, starting with the first question, followed by the second and so on

    # Interpret and display plots concerning some general information about the passengers
    def general_plots(self, titanic_DFrame):
        # 1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
        print(' 1) Some overall information about the passengers \n')
        print(titanic_DFrame.describe())
        fig =plt.figure( figsize=(30, 7))
        plt.suptitle("General Statistics")
        # a classic bar display#############################################################
        fig.add_subplot(221)
        sns.countplot('Pclass', data=titanic_DFrame, hue='Sex')
        plt.title('The distribution of the Sexes')
        plt.legend(title='Class')
        plt.xlabel('Class')

        # The total variation of age in a histogram#############################################################
        # generations are [19, 26, 41, 56, 76]
        fig.add_subplot(222)
        generations = [0, 19, 26, 41, 56, 76, np.inf]
        names = ['0-18', '19-25', '26-41', '42-55', '56-75', '76+']
        titanic_age = titanic_DFrame
        titanic_age['AgeRange'] = pd.cut(titanic_age['Age'], generations, labels=names)
        sns.countplot('AgeRange', data=titanic_DFrame, hue='Sex')
        plt.title('The distribution of Generations per Sex')
        plt.xlabel('Age')


        # clear the memory of the temporaries
        del titanic_age

        # Split the graph into males,females and children#############################################################
        type_person_DF = titanic_DFrame
        type_person_DF['type_person'] = type_person_DF[['Age', 'Sex']].apply(male_female_child, axis=1)
        fig.add_subplot(223)
        sns.countplot('Pclass', data=type_person_DF, hue='type_person')
        plt.legend(title='Type')
        plt.title('The distribution of types of passengers')

        plt.xlabel('Class')
        plt.ylabel('Number')

        plt.savefig('Images/1_dist_general.png')

        # clear the memory of the temporaries
        del type_person_DF
        del fig

        # Let's use FacetGrid to plot multiple kedplots on one plot:###################################################

        fig = sns.FacetGrid(titanic_DFrame, hue="Pclass", aspect=4, height=4)
        fig.map(sns.kdeplot, 'Age', shade=True)
        oldest = titanic_DFrame['Age'].max()
        fig.set(xlim=(0, oldest))
        plt.subplots_adjust(top=0.9)
        plt.legend(title='Class')
        plt.title('The distribution of types of passengers')

        plt.savefig('Images/1_dist_ages_facet.png')
        plt.show()

        # clear the memory of the temporaries
        del fig

    # Interpret and display plots concerning the deck level and changes of survival
    def deck_plots(self, titanic_DFrame):
        # 2.) What deck were the passengers on and how does that relate to their class?########################
        print(' 2) In this section, we will find on what deck were the passengers located\n\n')

        # remove all Nan values into a new DF
        type_cabin = titanic_DFrame.dropna()

        fig = plt.figure(figsize=(30, 7))
        # for the Count Plot, group them in subplots
        plt.suptitle("Decks and survival rate")

        # apply a filter to create a new column and clean the results
        type_cabin['Level'] = type_cabin[['Cabin']].apply(cabin_finder, axis=1)
        type_cabin = type_cabin[type_cabin.Level != 'T']

        fig.add_subplot(121)
        plt.title('The number of passengers on each cabin')
        sns.countplot('Level', data=type_cabin, hue='Pclass')
        plt.legend(title = 'Class')

        # Did the deck have an effect on the passengers survival rate? ########################################
        fig.add_subplot(122)
        plt.title('The number of passengers that survived on each cabin')
        sns.countplot('Level', hue='Survived', data=type_cabin)

        plt.savefig('Images/2_dist_level_survival.png')
        plt.show()

        # clear the memory of the temporaries
        del type_cabin
        del fig

    # Interpret and display plots concerning the embarking location and changes of survival
    def embarking_plots(self, titanic_DFrame):
        # 3.) Where did the passengers come from?
        print(' 3.) The people embarked from: \n\n')

        # plot the data according to the Embarked/Class correlation
        sns.countplot(data=titanic_DFrame, x='Embarked', hue='Pclass')
        plt.legend(title = 'Class')
        plt.title('Embarking locations of the passengers')
        plt.xlabel('Embarked Location')
        plt.legend(title='Class')
        plt.savefig('Images/3_dist_locations.png')
        plt.show()

    # Interpret and display plots concerning the family status on board and changes of survival
    def family_alone_plots(self, titanic_DFrame):
        # 4.) Who was alone and who was with family?#############################################################
        print('4.) Who was alone and how many from the same family were on board according to class')

        # copy the data into a temporary DF with only the Class and family information
        alone_df = titanic_DFrame[['Pclass', 'SibSp', 'Parch', 'Survived']]

        # apply the necessary filters and build another column with the nr of family members
        alone_df['Alone/Fam'] = alone_df[['SibSp', 'Parch']].apply(alone_family, axis=1)

        fig= plt.figure(figsize=(15, 15))
        plt.suptitle("Alone/Family and survival rate")
        # plot the data
        fig.add_subplot(221)
        sns.countplot('Alone/Fam', data=alone_df, hue='Pclass')
        plt.legend(title = 'Class')
        plt.title('The class distribution of the family members on board')
        plt.xlabel('Number of family members')
        # Did having a family member increase the odds of surviving the crash?#####################################
        fig.add_subplot(222)
        sns.countplot(hue='Alone/Fam', data=alone_df, x='Survived')
        plt.title('Family member number associated with the odds of survival')

        #The same data, but in boolean
        alone_df['Alone/Fam_bool'] = alone_df[['Alone/Fam']].apply(alone_family_bool, axis=1)

        fig.add_subplot(223)
        sns.countplot('Alone/Fam_bool', data=alone_df, hue='Pclass')
        plt.legend(title = 'Class')
        plt.title('The class distribution passengers that have family on board')
        plt.xlabel('Number of family members')
        # Did having a family member increase the odds of surviving the crash?#####################################
        fig.add_subplot(224)
        sns.countplot(hue='Alone/Fam_bool', data=alone_df, x='Survived')
        plt.legend(title = 'Family?')
        plt.title('Family member on board associated with the odds of survival')

        plt.savefig('Images/4_dist_family_survival.png')
        plt.show()

        # clear the memory of the temporaries
        del alone_df
        del fig

    # Interpret and display plots concerning all the important factors and changes of survival
    def survival_factors_plots(self, titanic_DFrame):
        # 5.) What factors helped someone survive the sinking?
        print('5.) What Factors helped someone survive?')

        fig = plt.figure(figsize=(15, 5))
        # Class divide#############################################################
        fig.add_subplot(121)
        sns.countplot('Pclass', data=titanic_DFrame, hue='Survived')
        plt.title('Class factored survival rate')
        plt.xlabel('Class')
        plt.ylabel('Number of Passengers')

        # General trend of the Age/Death correlation######################################
        fig.add_subplot(122)
        # generations are [19, 26, 41, 56, 76]
        generations = [0, 19, 26, 41, 56, 76, np.inf]
        names = ['0-18', '19-25', '26-41', '42-55', '56-75', '76+']
        titanic_age = titanic_DFrame
        titanic_age['AgeRange'] = pd.cut(titanic_age['Age'], generations, labels=names)
        sns.countplot('AgeRange', data=titanic_DFrame, hue='Survived')
        plt.title('The distribution of Generations and their Survival Numbers')
        plt.xlabel('Generation')
        plt.ylabel('Number of Passengers')

        plt.savefig('Images/5_age_sex_survival.png')
        plt.show()

        # clear the memory of the temporaries
        del titanic_age
        del fig
