# -*- coding: utf-8 -*-
"""lecture10_LogisticRegression_LDA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aqV8yzZMhplrv3vZxiTiVC09fVCMRnMU
"""

# Logistic Regression

import pandas as pd
import numpy as np
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Read Data
df = pd.read_csv('/content/Week10data_students.csv')
df.head()

x = df.drop('Pass_Or_Fail',axis = 1)
y = df['Pass_Or_Fail']

# Split Data to train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)

# Apply logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)

# Make predictions 
y_pred = logistic_regression.predict(x_test)
print(y_pred)

# Get accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
accuracy_percentage = 100 * accuracy
print(accuracy_percentage)

# predicting whether a student will fail or pass
First_Student = logistic_regression.predict((np.array([4, 38]).reshape(1, -1)))
print(First_Student)

# predicting whether a student will fail or pass
Second_Student = logistic_regression.predict((np.array([8, 29]).reshape(1, -1)))
print(Second_Student)

# Heart Disease prediction using Logistic Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

# Read Data
heart_df=pd.read_csv("/content/Week10data_heart.csv")
heart_df.drop(['education'],axis=1,inplace=True)
heart_df.head()

heart_df.dropna(axis=0,inplace=True)

heart_df.describe()

x = heart_df.drop('TenYearCHD',axis = 1)
y = heart_df['TenYearCHD']

# Split train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=5)

# Apply Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
print(y_pred)

# Get Accuracy

from sklearn.metrics import accuracy_score
print('Accuracy: ' + str(accuracy_score(y_test, y_pred)*100))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


cm=confusion_matrix(y_test,y_pred)
print(cm)
plot_confusion_matrix(logreg,x_test,y_test)

# Get Probabilites of each patient to each prediction
y_pred_prob=logreg.predict_proba(x_test)[:,:]
y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no heart disease (0)','Prob of Heart Disease (1)'])
y_pred_prob_df.head()

# Plot ROC curve
from sklearn.metrics import roc_curve
y_pred_prob_yes=logreg.predict_proba(x_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

# Linear Discriminant Analysis LDA

# Load libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris flower dataset:
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the Data
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply Linear Discriminant Analysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# Use LDA with a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier() 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Plot Confusion Matrix
import pandas as pd
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

plot_confusion_matrix(classifier,X_test,y_test)

print('Accuracy' + str(accuracy_score(y_test, y_pred)))