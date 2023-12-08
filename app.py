import gradio as gr
from predict_stock import predict_stock_trend


stock_trend = gr.Interface(
    fn=predict_stock_trend,
    inputs="text",
    outputs="image",
    title="JP Morgan & Chase Co. Stock Trend Predictor",
    description="Enter a date (YYYY-MM-DD) to predict stock trends"
)

stock_trend.launch()