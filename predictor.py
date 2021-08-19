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
from sklearn.svm import SVC

# print_file function print data to a file
def print_file (data, file_name):
    original_stdout = sys.stdout
    with open (file_name, "w") as local_file:
        sys.stdout = local_file
        print (data)
        sys.stdout = original_stdout

# get dataset
from sklearn import datasets
iris = datasets.load_iris()

# delete the old output
option=input("Do you want to delete the files in output folder? (Y/N)")
if (option == "Y"):
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

