import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

dataset = pd.read_csv("breast-cancer.csv")
dataset.head()

label_encoder = LabelEncoder()
dataset["diagnosis"] = label_encoder.fit_transform(dataset["diagnosis"].values)

X = dataset.drop("diagnosis", axis=1)
y = dataset["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)

confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))


