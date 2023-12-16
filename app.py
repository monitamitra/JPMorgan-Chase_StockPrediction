from predict_stock import predict_stock_trend
import streamlit as st


st.title("Predicting JP Morgan & Chase Co. Stock Trends")
st.markdown("LSTM model aims to forecast the monthly stock direction of JP Morgan Chase & Co. \
        when provided with a date as its input")

col1 = st.columns(1)

with st.sidebar:
        date = st.date_input("Choose a date to predict stock trends")

with col1:
        st.text("Prediction")
        button_predict = st.button("Predict")

if button_predict and date:
        result = predict_stock_trend(date)
        st.pyplot(fig=result, clear_figure=True)
