import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("train.csv")

X = data.drop('SalePrice', axis=1)
y = data.loc[:, 'SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

linear_reg = LinearRegression()
ridge_reg = make_pipeline(StandardScaler(with_mean=False),Ridge())

linear_reg.fit(X_train, y_train)
ridge_reg.fit(X_train, y_train)

linear_pred = linear_reg.predict(X_test)
ridge_pred = ridge_reg.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)

print(f"MSE without Ridge: {linear_mse}")
print(f"MSE with Ridge: {ridge_mse}")