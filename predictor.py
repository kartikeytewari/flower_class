# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import sys

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# print_file function print data to a file
def print_file (data, file_name):
    original_stdout = sys.stdout
    with open (file_name, "w") as local_file:
        sys.stdout = local_file
        print (data)
        sys.stdout = original_stdout

# trains and test on a specified model 
def train_test(x_train, y_train, x_test, y_test, local_model):
    local_model.fit(x_train, y_train)
    predictions = local_model.predict(x_test)
    local_score = accuracy_score(y_test, predictions)
    return local_score;

# get dataset
from sklearn import datasets
iris = datasets.load_iris()

# delete the old output
option=input("Do you want to delete the files in output folder? (Y/N)")
if (option == "Y" or option == "y"):
    print ("Deleting files in output folder")
    os.system("rm -rf ./output/*")
else:
    print ("Aborting Script")
    exit()

# generate the data frame
iris_data = iris.data 
iris_data = pd.DataFrame(iris_data, columns = iris.feature_names)
iris_data['class'] = iris.target

# generating stats
print_file(iris_data.describe(), "output/stat.txt")
sns.boxplot(data = iris_data, width=0.5, fliersize=10)
sns.set(rc={'figure.figsize':(10,20)})
plt.savefig("output/fig.png")

# divide data for training and testing
x=iris_data.values[:,0:4]
y=iris_data.values[:,4]
x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.2, random_state=42)
print ("Data size for training = " + str(x_train.shape[0]))
print ("Data size for testing = " + str(x_test.shape[0]))

# training and testing
# KNN neighbours
score_knn = train_test(x_train, y_train, x_test, y_test, KNeighborsClassifier())
print ("Score from KNN Neighbours model = " + str(score_knn))

# Support Vector Machine model
score_svc = train_test(x_train, y_train, x_test, y_test, SVC())
print ("Score from Support Vector Machine model = " + str(score_svc))

# Random forest classifier model
score_random_forest = train_test(x_train, y_train, x_test, y_test, RandomForestClassifier(n_estimators=5))
print ("Score from Random forest classifier model = " + str(score_random_forest))

# Logistic Regression
score_logistic = train_test(x_train, y_train, x_test, y_test, LogisticRegression(max_iter=200))
print ("Score from Logistic Regression Algorithm = " + str(score_logistic))
