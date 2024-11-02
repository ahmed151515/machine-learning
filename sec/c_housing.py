import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


data = pd.read_csv("california_housing_train.csv")

data = data.dropna()
numeric_data = data.select_dtypes(include='number')

X = numeric_data.drop('median_house_value', axis=1)
y = numeric_data['median_house_value'].values.ravel()

best = 0
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(acc)
    if acc > best:
        best = acc
        with open("sec/housingModel.pickle", "wb") as f:
            pickle.dump(model, f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = pickle.load(open("sec/housingModel.pickle", "rb"))
y_pred = model.predict(X_test)
with open("sec/housing_Model_Result.txt", "w") as f:
    f.write(f"Model Score: {model.score(X_test, y_test)}\n")
    f.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
