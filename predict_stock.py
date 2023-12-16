import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

def predict_stock_trend(input_date):
    model = load_model("Stock_Pred_LSTM.h5")

    date_split = input_date.split("-")
    start_date = date(int(date_split[0]), int(date_split[1]), int(date_split[2]))
    future_date = start_date + relativedelta(months=1)
    future_date = future_date.strftime("%Y-%m-%d")

    test_data = yf.download("JPM", start=input_date, end=future_date, progress=False)
    test_data = test_data.iloc[:, 1].values

    unscaled_x_training_data = pd.read_csv("C:/Users/mitra/Downloads/JPM_Training.csv")
    unscaled_test_data = yf.download("JPM", start=input_date, end=future_date, progress=False)
    all_data = pd.concat((unscaled_x_training_data['Open'], unscaled_test_data['Open']), axis = 0)
   
    x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
    x_test_data = np.reshape(x_test_data, (-1, 1))

    scaler = MinMaxScaler()
    training_data = pd.read_csv("C:/Users/mitra/Downloads/JPM_Training.csv").iloc[:, 1].values
    scaler.fit_transform(training_data.reshape(-1, 1))
    x_test_data = scaler.transform(x_test_data)

    # group arrays to where each entry = date in November and contains stock prices of 40 previous trading days
    final_x_test_data = []
    for i in range(40, len(x_test_data)):
        final_x_test_data.append(x_test_data[i-40:i, 0])

    final_x_test_data = np.array(final_x_test_data)

    # reshape test data to tensorflow's liking
    final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

    predictions = model.predict(final_x_test_data)
    predictions = scaler.inverse_transform(predictions)

    plotRes = plt.figure(figsize=(10, 6))
    plotRes.plot(predictions, marker='o')
    plotRes.title('Predicted Stock Trends')
    plotRes.xlabel('Date')
    plotRes.ylabel('Stock Price')
    plotRes.xticks(rotation=45)
    plotRes.tight_layout()

    return plotRes