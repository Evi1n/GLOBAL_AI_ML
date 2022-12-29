import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

dataset = pd.read_csv("DecisionTrees_titanic.csv")
dataset.head()

X = dataset.drop("Survived", axis=1)
y = dataset.loc[:,"Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

decisiontree = DecisionTreeClassifier()
randomforest = RandomForestClassifier()

decisiontree.fit(X_train, y_train)
randomforest.fit(X_train, y_train)

decisiontree_pred = decisiontree.predict(X_test)
randomforest_pred = randomforest.predict(X_test)

print(classification_report(y_test, decisiontree_pred))
print(classification_report(y_test, randomforest_pred))


fig = plt.figure(figsize=(10,10))
plot_tree(decisiontree,
          max_depth=2,
          feature_names=X.columns,
          filled=True,
          impurity=False,
          rounded=True,
          precision=1)

plt.show()