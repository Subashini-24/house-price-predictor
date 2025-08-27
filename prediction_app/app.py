import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_prices.csv")

X = df[["bedrooms", "bathrooms", "sqft_living"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

st.title("🏠 House Price Prediction App")

bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms", 1, 10, 2)
sqft_living = st.slider("Living Area (sqft)", 500, 5000, 1500, step=50)

if st.button("Predict Price"):
    input_data = [[bedrooms, bathrooms, sqft_living]]
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
