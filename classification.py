import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC

dataset = pd.read_csv("breast-cancer.csv")
print(dataset.shape)
print(dataset.head())
print(dataset.tail())

labelencoder = LabelEncoder()
dataset["diagnosis"] = labelencoder.fit_transform(dataset["diagnosis"].values)
print(dataset.head())

train, test = train_test_split(dataset, test_size=0.3)

X_train = train.drop("diagnosis", axis=1)
y_train = train.loc[:,"diagnosis"]

X_test = test.drop("diagnosis",axis=1)
y_test = test.loc[:,"diagnosis"]

model1 = LogisticRegression()
model1.fit(X_train, y_train)

predictions = model1.predict(X_test)
print(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

model2 = LinearSVC()
model2.fit(X_train, y_train)

predictions = model2.predict(X_test)
print(predictions)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

