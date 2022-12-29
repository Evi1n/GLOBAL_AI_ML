from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

data = pd.DataFrame(X)
data.insert(784, "label", y)

print(data.head())
print(X_train[2])

for i in range(5):
  plt.imshow(X[i].reshape((28, 28)), cmap='gray')
  plt.show()

mlp = MLPClassifier(hidden_layer_sizes=1, activation = "logistic")
mlp1 = MLPClassifier(hidden_layer_sizes=100, activation = "logistic")
mlp2 = MLPClassifier(hidden_layer_sizes=1000, activation = "logistic")

mlp.fit(X_train, y_train)
mlp1.fit(X_train, y_train)
mlp2.fit(X_train, y_train)

predictions_NN = mlp.predict(X_test)
predictions_NN

predictions_NN1 = mlp1.predict(X_test)
predictions_NN1

predictions_NN2 = mlp2.predict(X_test)
predictions_NN2

print(f"Actual Value: {y_test[0]}")
print(f"Predicted Value: {predictions_NN2[0]}")


plt.imshow(X_test[0].reshape((28, 28)), cmap='gray')



print(f"Actual Value: {y_test[1]}")
print(f"Predicted Value For 1 Hidden Layer: {predictions_NN[1]}")
print(f"Predicted Value For 100 Hidden Layer: {predictions_NN1[1]}")
print(f"Predicted Value For 1000 Hidden Layer: {predictions_NN2[1]}")


plt.imshow(X_test[1].reshape((28, 28)), cmap='gray')
plt.show()

confusion_matrix(y_test, predictions_NN)
print(classification_report(y_test,predictions_NN))

confusion_matrix(y_test, predictions_NN1)
print(classification_report(y_test,predictions_NN1))

confusion_matrix(y_test, predictions_NN2)
print(classification_report(y_test,predictions_NN2))
