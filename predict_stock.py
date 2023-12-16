import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.lines as mlines


def predict_stock_trend(input_date):
    model = load_model("Stock_Pred_LSTM.h5")

    start_date = input_date 
    future_date = start_date + relativedelta(months=1)
    future_date = future_date.strftime("%Y-%m-%d")

    test_data = yf.download("JPM", start=input_date, end=future_date, progress=False)
    test_data = test_data.iloc[:, 1].values

    unscaled_x_training_data = pd.read_csv("JPM_Training.csv")
    unscaled_test_data = yf.download("JPM", start=input_date, end=future_date, progress=False)
    all_data = pd.concat((unscaled_x_training_data['Open'], unscaled_test_data['Open']), axis = 0)
   
    x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
    x_test_data = np.reshape(x_test_data, (-1, 1))

    scaler = MinMaxScaler()
    training_data = pd.read_csv("JPM_Training.csv").iloc[:, 1].values
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

    fig, ax = plt.subplots()
    plt.plot(predictions, marker='o', markerfacecolor='none', color="black")
    plt.title('Predicted Stock Trend')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    
    date_form = DateFormatter("%m/%d")
    # type of date formatter
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # positions x ticks on like of axis
    plt.xticks(rotation = 40, ha = "right", fontsize = 14)
    ax.set_xlabel("Date", fontsize = 14)
    ax.xaxis.labelpad = 25.0
    ax.yaxis.labelpad = 25.0
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("blue")
    
    prediction_points = mlines.Line2D([], [], color = "black", marker = "o", markerfacecolor="None", linestyle='None', 
        label = "Daily Closing Points")
    
    plt.legend(handles = [prediction_points], fontsize = 9, loc = "upper right", bbox_to_anchor=(1.1, 1.05))

    plt.tight_layout()

    return fig