import pandas as pd
import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")

validation_df = pd.read_csv("validation.csv")
predicted_price_df = pd.read_csv("price_predictions.csv")

st.markdown("<h1 style='text-align: center;'>House Price Prediction</h1>", unsafe_allow_html=True)

sidebar = st.sidebar
sidebar.title("Model Choosing")
option = sidebar.selectbox("What Model should I use?", ("Linear_Regression", "DecisionTreeRegressor", "Lasso", "Ridge", "SVR",
                                               "Bayes", "Gradient_Boosting", "Random_Forest"))

st.markdown("<h5>Data For Prediction</h5>", unsafe_allow_html=True)
st.dataframe(validation_df)
st.markdown("<h5>Predicted Price</h5>", unsafe_allow_html=True)
st.dataframe(predicted_price_df[option])



