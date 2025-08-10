import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split,  cross_validate
from sklearn.preprocessing import RobustScaler
sns.set_theme()

df = pd.read_csv("/Users/iremnilcicek/Desktop/Diamonds Prices2022 2.csv")
print(df.head())
print(df.describe())
print(df.info())

df = df.drop(columns=["Unnamed: 0"])

sns.scatterplot(x="carat", y="price", data=df, alpha=0.5)
plt.title("Elmas karat ağırlığı ve fiyat ilişkisi")
plt.xlabel("Karat")
plt.ylabel("Fiyat")
#plt.show()

x = df[["carat"]]
y = df[["price"]]
model = LinearRegression()
model.fit(x, y)
print(f"Model sabiti (b): {model.intercept_[0]:.2f}")
print(f"Katsayı (w): {model.coef_.flatten()[0]:.2f}")

plt.figure(figsize=(8, 6))
sns.regplot(x="carat", y="price", data=df, ci=None, line_kws={"color": "red"})
plt.title(f"price = {model.intercept_[0]:.2f} + {model.coef_.flatten()[0]:.2f} * carat")
plt.xlabel("Karat")
plt.ylabel("Fiyat")
#plt.show()

y_pred = model.predict(x)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

yeni_elmas = pd.DataFrame([[1.5]], columns=["carat"])
tahmini_fiyat = model.predict(yeni_elmas).item()
print(f"1.5 karat elmas için tahmini fiyat: {tahmini_fiyat:.2f} $")
doviz_kuru = 40.58
tahmini_fiyat_tl = tahmini_fiyat * doviz_kuru
print(f"1.5 karat elmas için tahmini fiyat: {tahmini_fiyat_tl:.2f} TL")

# Lojistik Rregresyon Kuruyoruz
df["predicted_price"] = model.predict(df[["carat"]]).ravel()
df["profitable"] = (df["predicted_price"] < df["price"] * 0.9).astype(int)

cat_cols = ["cut", "color", "clarity"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

features = [col for col in df.columns if col not in["price", "profitable"]]
x_log = df[features]
y_log = df["profitable"]

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_log = pd.DataFrame(scaler.fit_transform(x_log), columns=features)

X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size=0.2, random_state=42)
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

print("\n[LOJİSTİK] Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

cv_results = cross_validate(log_model, X_log, y_log, cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print("\nCross-Validation Result:")
print("Accuracy:", cv_results["test_accuracy"].mean())
print("Precision:", cv_results["test_precision"].mean())
print("Recall:", cv_results["test_recall"].mean())
print("F1 Score:", cv_results["test_f1"].mean())
print("ROC AUC:", cv_results["test_roc_auc"].mean())