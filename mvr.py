import pandas as pd
import numpy as np
from sklearn import metrics, linear_model

df = pd.read_csv('book1.csv')
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['area', 'bedrooms', 'age']], df.price)
print(reg.predict([[3000, 3, 40]]))    # Predict price for a house with 3000 sqft area, 3 bedrooms, and 40 years old
