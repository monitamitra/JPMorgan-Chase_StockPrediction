from predict_stock import predict_stock_trend
import streamlit as st


st.title("Predicting JP Morgan & Chase Co. Stock Trends")
st.markdown("LSTM model aims to forecast the monthly stock direction of JP Morgan Chase & Co. \
        when provided with a date as its input")

col1, col2 = st.columns(2)

with col1:
        user_input = st.text_input("Enter a date (YYYY-MM-DD) to predict stock trends")

with col2:
        st.text("Prediction")
        button_predict = st.button("Predict")

if button_predict and user_input:
        result = predict_stock_trend(user_input)
        st.pyplot(fig=result, clear_figure=True)
