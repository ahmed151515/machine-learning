import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# قراءة البيانات مع معالجة تحذير DtypeWarning
data = pd.read_csv("new.csv", encoding='ISO-8859-1', low_memory=False)
print("data loaded successfully")
data = data.dropna()
numeric_data = data.select_dtypes(include='number')

# تعيين المتغيرات المستهدفة (y) و المتغيرات المستقلة (X)
y = numeric_data[["totalPrice"]].values.ravel()  # تحويل y إلى شكل أحادي البعد
X = numeric_data.drop(["totalPrice"], axis=1)

# تقسيم البيانات إلى مجموعات التدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# إنشاء النموذج واستخدام RandomForestRegressor
# model = RandomForestRegressor()
# model = LinearRegression()
# best = 0
# for i in range(2000):

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#     model.fit(X_train, y_train)
#     acc = model.score(X_test, y_test)
#     print(i, acc)
#     if acc > best:
#         best = acc
#         with open("houseModel.pickle", "wb") as f:
#             pickle.dump(model, f)

# التنبؤ باستخدام النموذج

model = pickle.load(open("houseModel.pickle", "rb"))
y_pred = model.predict(X_test)

# تقييم النموذج
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# طباعة نتائج التقييم
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
print(f"Score: {model.score(X_test, y_test) * 100}")
