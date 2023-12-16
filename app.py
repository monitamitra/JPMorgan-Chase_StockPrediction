from predict_stock import predict_stock_trend
import streamlit as st


st.title("Predicting JP Morgan & Chase Co. Stock Trends")
st.markdown("LSTM model aims to forecast the monthly stock direction of JP Morgan Chase & Co. \
        when provided with a date as its input")

st.header("Date")
col1, col2 = st.columns(2)

with col1:
        st.text_input("Enter a date (YYYY-MM-DD) to predict stock trends")

with col2:
        st.text("Prediction")
        
        
        st.button("Predict")
        if st.button("Predict") and st.text_input("Enter a date (YYYY-MM-DD) to predict stock trends"):
                        result = predict_stock_trend(text_input)
                        st.pyplot(fig=result, clear_figure=True)
