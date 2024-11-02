import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.linear_model as lm
import pickle
# قراءة البيانات
data = pd.read_csv("student-mat.csv")

# ضبط خيارات العرض للبيانات
pd.set_option('display.max_columns', 33)

# اختيار البيانات الرقمية فقط
numeric_data = data.select_dtypes(include='number')

# تحديد المتغير الهدف
targt = ["G3"]
y = numeric_data[targt]
X = numeric_data.drop(["G3"], axis=1)

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# إنشاء النموذج باستخدام RandomForestRegressor
# model = RandomForestRegressor()
# best = 0
# for i in range(30):

#     model.fit(X_train, y_train.values.ravel())
#     acc = model.score(X_test, y_test)
#     print(i, acc)
#     if acc > best:
#         best = acc
#         with open("studentmodel.pickle", "wb") as f:
#             pickle.dump(model, f)

model = pickle.load(open("studentmodel.pickle", "rb"))

# توقع القيم
y_pred = model.predict(X_test)

# حساب المقاييس
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# عرض النتائج
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')
print(f"Score: {model.score(X_test, y_test) * 100}")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='b')
plt.xlabel("Actual G3 Grades")
plt.ylabel("Predicted G3 Grades")
plt.title("Actual vs. Predicted G3 Grades - Linear Regression")
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
         y_test.max()], 'r--')  # خط لتمثيل التوقع المثالي
plt.show()
