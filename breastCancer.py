#Import libararies
from sklearn import datasets
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd
import seaborn as sns

#load dataset
cancerData = datasets.load_breast_cancer()
#load into Panadas dataframe
cancer = pd.DataFrame(cancerData.data, columns=cancerData.feature_names)

#########################################################
############### Exploratory Data Analysis ###############
#########################################################
#print the first 5 samples
display(cancer.head())
#include the target labels - (where 0 represents malignant cancer and 1 represents benign)
print("\nAdding the target variable to the dataframe...\n")
cancer['Target'] = cancerData.target
display(cancer.head())

#check for missing values in the data
print("Number of missing values: ", cancer.isnull().sum())

#plot the distribution of target values
sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.displot(cancer['Target'])
plt.show()

#create a correlation matrix to measure linear relationship between each of the features
## this stage is by no means necessary, but is interesting to see
corrMatrix = cancer.corr().round(2)
#plot the matrix on a heatmap
sns.heatmap(data=corrMatrix, annot=True)
plt.show()

#########################################################
#################### Data Splitting #####################
#########################################################
from sklearn.model_selection import train_test_split

#split data into training/test sets with 70/30 ratio
#random_state controls how much the order of the data is randomised before splitting
x_train, x_test, y_train, y_test = train_test_split(cancerData.data, cancerData.target, test_size=0.3, random_state=109)

#create scatterplot for training data
plt.scatter(x_train[:,0], x_train[:,1])
plt.title("Scatter plot of Traing data")
plt.show()

#########################################################
######################## SVM ############################
#########################################################
from sklearn import svm

#create classifier for the svm
classifier = svm.SVC(kernel='linear')

#train the the model using the training set
classifier = classifier.fit(x_train, y_train)

"""
There are 3 variables attached to the trained classifier object that relate to the support vectors of the model:
    The support_ variable, which holds the index numbers of the samples from your training set that were 
    found to be the support vectors.

    The n_support_ variable, which produces the number of support vectors for every class.

    The support_vectors_ variable, which produces the support vectors themselves – so that you don’t 
    need to perform an array search after using support_.
"""
#get the indices of the support vectors
print("Support vector indices: ", classifier.support_)

#get the number of support vectors per class
print("SVs per class: ", classifier.n_support_)

#get the support vectors themselves
supportVectors = classifier.support_vectors_

#visualise support vectors
plt.scatter(x_train[:,0], x_train[:,1])
plt.scatter(supportVectors[:,0], supportVectors[:,1], color='red')
plt.title("Visualisation of Support Vectors")
plt.show()

#make predictions using the test set
predictions = classifier.predict(x_test)
