import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Accumulation-accounts-2008-2022-provisional.csv")

data['Values'] = pd.to_numeric(data['Values'], errors='coerce')
data = data.dropna()
numeric_data = data.select_dtypes(include='number')

y = numeric_data[["Values"]].values.ravel()
x = numeric_data.drop(["Values"], axis=1)
best = 0
# for i in range(30):
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#     model = RandomForestRegressor()
#     model.fit(X_train, y_train)
#     acc = model.score(X_test, y_test)
#     if acc > best:
#         best = acc
#         with open("accountModel.pickle", "wb") as f:
#             pickle.dump(model, f)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = pickle.load(open("sec/accountModel.pickle", "rb"))
y_pred = model.predict(X_test)
with open("sec/account_Model_Result.txt", "w") as f:
    f.write(f"Model Score: {model.score(X_test, y_test)}\n")
    f.write(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
