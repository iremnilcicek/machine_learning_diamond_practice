import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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
