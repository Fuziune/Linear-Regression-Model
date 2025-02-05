import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

house  = fetch_california_housing()
df = pd.DataFrame(house.data, columns=house.feature_names)
df["Price"] = house.target
print(df.head())
print (df.size)
#print(df.info())
#print(df.describe())1
print(df.sort_values(by="Price", ascending=False).head(990))
#print(df.isnull().sum())

sns.histplot(df['Price'], bins=80, kde=True)
plt.title("Distribution of House Prices")
plt.show()

plt.figure(figsize=(8, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

x = df.drop(columns="Price")
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

