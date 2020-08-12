import pandas as pd
import streamlit as st
import numpy as np
import pickle
from catboost import Pool, CatBoostRegressor


model_path = "D:\DataScience\Kaggle\Regression\streamlit_app\catboost.pickle"
model = pickle.load(open(model_path,'rb'))



train = pd.read_csv("D:\\DataScience\\Kaggle\\Regression\\streamlit_app\\train.csv")
train[["Location","Fuel_Type","Transmission","Owner_Type","Manufacturer"]] = train[["Location","Fuel_Type","Transmission","Owner_Type","Manufacturer"]].astype("category")
train = train.drop("Unnamed: 0", axis = 1).copy()

X = train.drop("Price", axis = 1).copy()
y = train.Price

train_pool = Pool(X, y, cat_features = ["Location","Fuel_Type","Transmission","Owner_Type","Manufacturer"])
catboost = CatBoostRegressor(iterations = 100, depth = 15, learning_rate = 0.01, loss_function = "RMSE")

st.title("Predict Car Prices")

def accept_user_data():
	min_year = int(X.Year.min())
	max_year = int(X.Year.max())
	location = st.selectbox("Choose a location: ",X.Location.values.unique())
	year = st.slider("Choose a year (in which you will sell the car): ", min_value = min_year, max_value = max_year)
	km_driven = st.slider("How many Kilometers driven? ", min_value = int(X.Kilometers_Driven.min()), max_value = int(X.Kilometers_Driven.max()))
	fuel_type = st.selectbox("Choose a fuel type: ", X.Fuel_Type.values.unique())
	transmission = st.selectbox("Choose a transmission: ", X.Transmission.values.unique())
	owner = st.selectbox("How many previous owners? ", X.Owner_Type.values.unique())
	mileage = st.slider("What Mileage? ", min_value = float(X.Mileage.min()), max_value = float(X.Mileage.max()))
	engine = st.slider("What Engine? ", min_value = float(X.Engine.min()), max_value = float(X.Engine.max()))
	power = st.slider("Horsepower: ", min_value = float(X.Power.min()), max_value = float(X.Power.max()))
	seats = st.selectbox("Number of seats: ", X.Seats.unique())
	maker = st.selectbox("Manufacturer: ", X.Manufacturer.values.unique())


	user_prediction_data = np.array([location, year, km_driven, fuel_type, transmission, owner, mileage, engine, power, seats, maker]).reshape(1,-1)

	return user_prediction_data


user_prediction_data = accept_user_data()
st.write("your selection: ", user_prediction_data) 

preds = model.predict(user_prediction_data)

st.write("Predicted Price (INR) given your input parameters: ", preds*100000)
st.write("Predicted Price (EUR) given your input parameters: ", (preds*100000)/88.1984)
