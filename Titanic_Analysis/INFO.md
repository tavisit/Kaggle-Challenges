# Titanic_Analysis

This repository is my first big project in Python and the first time I am around Data Analysis, so I tackled the classic [Titanic Problem](https://www.kaggle.com/c/titanic/overview) on Kaggle using a 2 part program. The inspiration came from the lectures of Jose Partilla, problem analysed only with plots in this particular [course](https://www.udemy.com/course/learning-python-for-data-analysis-and-visualization). The application is structured as a menu consisting of 6 options solving a particular question:
1. Who were the passengers on the Titanic? (Ages,Gender,Class,..etc) **( Data Analysis)**
2. What deck were the passengers on and how does that relate to their class? **( Data Analysis)**
3. Where did the passengers come from? **( Data Analysis)**
4. Who was alone and who was with family? **( Data Analysis)**
5. What factors helped someone survive the sinking? **( Data Analysis)**
6. Machine output of a similar input **( Machine Learning)**

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

# Data analysis

The Data Analysis part was done using the libraries offered by Numpy, Pandas, Seaborn and Matplotlib. These are basic plots, in order to understand better the data and build a complete image around the Titanic problem. This part is merged into a single file, named 'VisualizationModule.py', with a couple of basic functions and a class, 'VisualizationClass' which encapsulates the 5 methods responsable for this part. Each question has a couple of plots and the capability to save each plot into a subfolder, '/Images', of the main script with the name.
The main data frame that is processed is called titanic_DFrame, which is read from a .CSV from the same folder. In this case we talk about the 'train.csv'

## First Question

We can see in the first plot the distribution of genres, class of boarding and the type of the passenger on the ship. To share more information, the program prints the .describe() method from the data frame:
```Python
print(titanic_DFrame.describe())
```
After that, it is used a the line ```fig = plt.figure( figsize=(30, 7))``` in order to produce the plots in one GUI window. The plots are all ```sns.countplot()```, besides the last one, which was build to describe a continuous functions over the domanin with the following code:
```Python
fig = sns.FacetGrid(titanic_DFrame, hue="Pclass", aspect=4, height=4)
fig.map(sns.kdeplot, 'Age', shade=True)
oldest = titanic_DFrame['Age'].max()
fig.set(xlim=(0, oldest))
```
When the methods of savefig() and .show() are called, the plot is saved in folder and displayed on screen
```
plt.savefig('Images/path')
plt.show()
```
The following figure is drawn:
![Plot_1](https://live.staticflickr.com/65535/50163556266_7d027f2812_k.jpg)

## Second Question

In the second image, we can see the number of passengers of each class on what level they were boarded and the number of survivals per level. Be aware that in order to have valid numbers, the null elements were removed beforehand and the new data frame copied into a temporary DF ```type_cabin = titanic_DFrame.dropna()```.
For the determination of level, there was imperios to remove the numbers from the cabin number, so the following code is expected to remove and extract the level:
```Python
type_cabin['Level'] = type_cabin[['Cabin']].apply(cabin_finder, axis=1)
type_cabin = type_cabin[type_cabin.Level != 'T'] 
```
*A residual level 'T' was appearing, so it was removed*

The following figure is drawn and saved:
![Plot_2](https://live.staticflickr.com/65535/50163556241_290fb4fada_k.jpg)

## Third Question

This image shows the number of people that boarded in each of the boarding harbours( shorted to S,Q,C). The plot, also, shows the exact number of passengers boarded per class, without further modification or alteration of the data frame.
The following figure is drawn:
![Plot_3](https://live.staticflickr.com/65535/50163017408_b1a597a85c_z.jpg)

## Fourth Question

This part was a little difficult, because the input data doesn't share this information easily, so further manipulation of the data frame was required. Firstly, we bulid a new data frame from the basic frame, but only with the information we need
```Python
 alone_df = titanic_DFrame[['Pclass', 'SibSp', 'Parch', 'Survived']]
```
After that, a new column is build with the help of the function 'alone_family' for the numbers and 'alone_family_bool' for the boolean information:
```Python
alone_df['Alone/Fam'] = alone_df[['SibSp', 'Parch']].apply(alone_family, axis=1)
alone_df['Alone/Fam_bool'] = alone_df[['Alone/Fam']].apply(alone_family_bool, axis=1) 
```
The following figure is drawn:
![Plot_4](https://live.staticflickr.com/65535/50163017388_23b4354c12_c.jpg)

## Fifth Question

When considering the survival factors of the passengers, there occured the problem of plots with many bars and segments, so it needed some approximations to be made, so the generations variable was implemented and used to split the age of the people into 6 distict groups
```Python
generations = [0, 19, 26, 41, 56, 76, np.inf]
names = ['0-18', '19-25', '26-41', '42-55', '56-75', '76+']
titanic_age = titanic_DFrame
titanic_age['AgeRange'] = pd.cut(titanic_age['Age'], generations, labels=names)
```
These groups were chosen in order to ensure a somehow equal representation of the modern term of 'Generation' to determine general knowledge about the passengers.
The following figure is drawn:
![Plot_5](https://live.staticflickr.com/65535/50163556171_e67e963b03_h.jpg)

Before the Machine Learning Part is discussed, every new data structure that was introduced in the methods, it was deleted from memory after the implementation with the keyword ```del d_structure```. That was done, in order to minimize the memory used in processing.

# Machine Learning
I took the challenge of the Titanic, because the said lecture tackled the problem, but only from a general data analysis side, so I tried to implement the exact task of the site, to train an algorithm to perform a prediction about the survival of a new set of passengers. As I'm writing this post, I am ranked among the top 10% of all Kagglers, More than 22000 teams are currently competing.
![Leaderboard](https://live.staticflickr.com/65535/50164384727_5d3f2ea0ac_b.jpg)

The module has only one class, 'Machine_Learning_Class', which contains all the algorithms and information needed to perform the Machine Learning Task.
The main method needs the main data frame, titanic_DFrame, but also the test data frame, which is read from the 'test.csv'.

I chose to work with [Random Forest Algorithm](https://en.wikipedia.org/wiki/Random_forest), mainly because it is the only ML Algorithm that I know so well that I can build such prediction.
First, the data needs to be cleaned, by dropping the name which doesn't give any practical advantage over the ML
```Python
titanic_DFrame_temp = titanic_DFrame.drop(['Name'], axis=1)
test_DFrame_temp = test_DFrame.drop(['Name'], axis=1)
```
After that, the two data frames are concatenated into one from the Class to the Embarked location, so that one may use easily the information embedded within:
```Python
ml_data = pd.concat((titanic_DFrame_temp.loc[:, 'Pclass':'Embarked'], test_DFrame_temp.loc[:, 'Pclass':'Embarked']))
```
In order to normalize the data, transform the skewed numeric features by taking log(feature + 1):
```Python
numeric_feats = ml_data.dtypes[ml_data.dtypes != "object"].index

skewed_feats = titanic_DFrame_temp[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

ml_data[skewed_feats] = np.log1p(ml_data[skewed_feats])
ml_data = pd.get_dummies(ml_data)
```
Considering we have a forest, the number of estimators are from the standard input and there are used mainstream methods to build the prediction and to form a score.
```Python
random_forest = RandomForestClassifier(n_estimators=estimators)
random_forest.fit(ml_data[:titanic_DFrame_temp.shape[0]], titanic_DFrame_temp.Survived)
random_forest_prediction = random_forest.predict(ml_data[titanic_DFrame_temp.shape[0]:])
random_forest.score(ml_data[:titanic_DFrame_temp.shape[0]], titanic_DFrame_temp.Survived)
```
Build a solution and create a csv with the 'PassengerID' and 'Survived' as columns As the Kaggle.com, Titanic: Machine Learning from Disaster, problem requires the DF has only two columns, an ID and Survived prediction, but it can be altered and expanded to show more information, if needed.
```Python
solution_DFrame = pd.DataFrame(
    {"PassengerId": test_DFrame_temp.PassengerId, "Survived": random_forest_prediction})
solution_DFrame.to_csv("solution_ML.csv", index=False)
```
The final results are displayed in the command prompt and saved as a .CSV with the name 'solution_ML.csv'.

# Conclusion
The Titanic was a human tragedy, but with the help of statistics and Machine Learning, the humankind can learn from it and produce a lot of improvements to ships, warning mechanism and so on.

This is not the only repository about this problem, so there is a lot of room to improve:
* Use different models and for this particular model, use Hyperparameters tuning
* Blend the models to have better results
* Dig more in the data and eventually build new features

This application has a lot more features to be implemented and to be improved, but considering that it is my first big project in python, it was a lot of fun and I learned a lot about data science and machine learning techniques
