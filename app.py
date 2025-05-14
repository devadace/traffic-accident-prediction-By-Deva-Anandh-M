import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Title and description
st.title("🚦 Traffic Accident Risk Predictor")
st.markdown("Predict the likelihood of a traffic accident based on environmental conditions.")

# User input form
with st.form("prediction_form"):
    temperature = st.slider("Temperature (°F)", -30, 130, 70)
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    visibility = st.slider("Visibility (miles)", 0, 10, 5)
    wind_speed = st.slider("Wind Speed (mph)", 0, 100, 10)
    weather_condition = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Fog", "Snow", "Thunderstorm"])
    day_of_week = st.selectbox("Day of the Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    input_data = pd.DataFrame({
        'Temperature(F)': [temperature],
        'Humidity(%)': [humidity],
        'Visibility(mi)': [visibility],
        'Wind_Speed(mph)': [wind_speed],
        'Weather_Condition': [weather_condition],
        'Day_Of_Week': [day_of_week]
    })

    # Predict accident risk
    prediction = model.predict(input_data)[0]
    risk = "High" if prediction == 1 else "Low"
    st.success(f"Predicted Accident Risk: {risk}")
