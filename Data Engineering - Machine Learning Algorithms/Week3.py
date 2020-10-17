# -*- coding: utf-8 -*-
"""lecture3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13FJJr85ThOHBtpZQseN8QS3DF84kgl5C
"""

# Pandas : data manipulation library

import pandas as pd

dictionary = {"NAME":["ali","veli","kenan","hilal","ayse","evren"],
              "AGE":[15,16,17,33,45,66],
              "SALARY": [100,150,240,350,110,220]} 

dataFrame1 = pd.DataFrame(dictionary)

head = dataFrame1.head()
print('head of the data frame:')
print(head)
tail = dataFrame1.tail()
print('tail of the data frame:')
print(tail)

print(dataFrame1.columns)

print(dataFrame1.info())

print(dataFrame1.dtypes)

print(dataFrame1.describe())

# indexing and slicing
print(dataFrame1["AGE"])
print(dataFrame1.AGE)

dataFrame1["New feature"] = [-1,-2,-3,-4,-5,-6]

print(dataFrame1.loc[:, "AGE"])

print(dataFrame1.loc[:3, "AGE"])

print(dataFrame1.loc[:3, "AGE":"NAME"]) # logically incorret

print(dataFrame1.loc[:3, ["AGE","NAME"]])

print(dataFrame1.loc[::-1,:])

print(dataFrame1.loc[:,:"NAME"])

print(dataFrame1.loc[:,"NAME"])

print(dataFrame1.iloc[:,2])

# Filtering
filtre1 = dataFrame1.SALARY > 200

filtered_data = dataFrame1[filtre1]
print(filtered_data)

filtre2 = dataFrame1.AGE <20

dataFrame1[filtre1 & filtre2]

print(dataFrame1[dataFrame1.AGE > 60])

#  drop and concatenating

dataFrame1.drop(["New feature"],axis=1,inplace = True)
print(dataFrame1)

# for safety:
# dataFrame_dropped = dataFrame1.drop(["New feature"],axis=1)

# vertical concatenation
data1 = dataFrame1.head()
print(data1)
data2 = dataFrame1.tail()
print(data2)
data_concat = pd.concat([data1,data2],axis=0)
print(data_concat)

# horizontal concatenation

salary = dataFrame1.SALARY
print(salary)
age = dataFrame1.AGE
print(age)
data_h_concat = pd.concat([salary,age],axis=1)
print(data_h_concat)

# Matplotlib : data plotting library

import pandas as pd

df = pd.read_csv("/content/drive/My Drive/SUREYYA HOCA'S COURSE/Intro to Python/6) Visualization with Matplotlib/iris.csv")

print(df.columns)

print(df.Species.unique())

print(df.info())

print(df.describe())

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]

print(setosa.describe())
print(versicolor.describe())

import matplotlib.pyplot as plt

df1 = df.drop(["Id"],axis=1)
setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")
plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")
plt.plot(virginica.Id,virginica.PetalLengthCm,color="blue",label= "virginica")
plt.legend()
plt.xlabel("Id")
plt.ylabel("PetalLengthCm")
plt.show()

df1.plot(grid=True,alpha= 0.9)
plt.show()

# scatter plot

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]


plt.scatter(setosa.PetalLengthCm,setosa.PetalWidthCm,color="red",label="setosa")
plt.scatter(versicolor.PetalLengthCm,versicolor.PetalWidthCm,color="green",label="versicolor")
plt.scatter(virginica.PetalLengthCm,virginica.PetalWidthCm,color="blue",label="virginica")
plt.legend()
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.title("scatter plot")
plt.show()

# histogram

plt.hist(setosa.PetalLengthCm,bins= 50)
plt.xlabel("PetalLengthCm values")
plt.ylabel("frekans")
plt.title("hist")
plt.show()

# bar plot

import numpy as np

x1 = np.array([1,2,3,4,5,6,7])
x2 = ["a","b","c","d","e","f","g"]
y = x1*2+5

plt.bar(x2,y)
plt.title("bar plot")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# subplots

df1.plot(grid=True,alpha= 0.9,subplots = True)
plt.show()

setosa = df[df.Species == "Iris-setosa"]
versicolor = df[df.Species == "Iris-versicolor"]
virginica = df[df.Species == "Iris-virginica"]

plt.subplot(2,1,1)
plt.plot(setosa.Id,setosa.PetalLengthCm,color="red",label= "setosa")
plt.ylabel("setosa -PetalLengthCm")
plt.subplot(2,1,2)
plt.plot(versicolor.Id,versicolor.PetalLengthCm,color="green",label= "versicolor")
plt.ylabel("versicolor -PetalLengthCm")
plt.show()

# import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# sklearn library
from sklearn.linear_model import LinearRegression

# import data
df = pd.read_csv("/content/drive/My Drive/SUREYYA HOCA'S COURSE/Machine Learning/1) Linear Regression/linear-regression-dataset.csv",sep = ";")

# plot data
plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

# prediction

b0_ = linear_reg.intercept_
print("b0_: ",b0_)   # y eksenini kestigi nokta intercept

b1 = linear_reg.coef_
print("b1: ",b1)   # egim slope


maas_yeni = 1663 + 1138*11
print(maas_yeni)


# visualize line
array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)  # deneyim


plt.scatter(x,y)


y_head = linear_reg.predict(array)  # maas

plt.plot(array, y_head,color = "red")
plt.show()