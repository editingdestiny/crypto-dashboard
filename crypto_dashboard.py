

import logging
import time
import threading
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from dash import dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.DEBUG)

# CCXT exchange & global variables
exchange = ccxt.binance()
CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
price_data = {pair: pd.DataFrame(columns=["timestamp", "price"]) for pair in CRYPTO_PAIRS}

def fetch_historical_data(pair, limit=1440): # 1440 minutes = 1 day
    ohlcv = exchange.fetch_ohlcv(pair, timeframe='5m', limit=limit) # 5 minutes interval
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) # Create a DataFrame with open, high, low, close, volume (ohlcv)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'close']]

def fetch_price(pair):
    ticker = exchange.fetch_ticker(pair)
    return {
        "timestamp": datetime.fromtimestamp(ticker['timestamp'] / 1000),
        "price": ticker['last']
    }

# Load crypto LSTM model and scaler
try:
    lstm_model = load_model('crypto_lstm_model.h5')
except Exception as e:
    logging.error(f"Error loading crypto LSTM model: {e}")
    raise

scaler = MinMaxScaler(feature_range=(0, 1))

def data_updater():
    global price_data
    while True:
        for pair in CRYPTO_PAIRS:
            try:
                new_data = fetch_price(pair)
                print(f"Fetched data for {pair}: {new_data}")  # Debug
                price_data[pair] = pd.concat([price_data[pair], pd.DataFrame([new_data])]).tail(100)
            except Exception as e:
                print(f"Error fetching data for {pair}: {e}")
        time.sleep(1)

# Start a background thread to update data
threading.Thread(target=data_updater, daemon=True).start()

def predict_lstm_multi_step(pair, steps=60, sequence_length=120): # 60 steps = 1 hour prediction. sequence_length is the number of previous time steps to consider.
    live_data = fetch_historical_data(pair, limit=sequence_length) # Fetch historical data for the last 50 minutes
    close_prices = live_data['close'].values.reshape(-1, 1) # Reshape the close prices into a 2D array
    scaled_data = scaler.fit_transform(close_prices)
    x_input = scaled_data[-sequence_length:]
    x_input = x_input.reshape(1, sequence_length, 1)
    predictions = []
    for _ in range(steps):
        next_step = lstm_model.predict(x_input)
        predictions.append(next_step[0][0])
        x_input = np.append(x_input[:, 1:, :], [[[next_step[0][0]]]], axis=1)
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_timestamps = pd.date_range(
        start=live_data['timestamp'].iloc[-1],
        periods=steps + 1,
        freq='1min'
    )[1:]
    return future_timestamps, predicted_prices.flatten()

# Crypto Dashboard layout
crypto_dashboard_layout = html.Div([
    html.H1("Crypto Price Tracker (LSTM)", style={'text-align': 'center'}),
    dcc.Dropdown(
        id='crypto-pair-dropdown',
        options=[{'label': pair, 'value': pair} for pair in CRYPTO_PAIRS],
        value='BTC/USDT',
        style={'width': '50%', 'margin': '10px auto'}
    ),
    dcc.Dropdown(
        id='interval-dropdown',
        options=[
            {'label': '1 Minute', 'value': '1m'},
            {'label': '5 Minutes', 'value': '5m'},
            {'label': '60 Minutes', 'value': '60m'}
        ],
        value='1m',
        style={'width': '50%'}
    ),
    dcc.Graph(id='live-price-graph'),
    html.Div([
        html.H3("Actual vs Predicted Table", style={'text-align': 'center'}),
        dash_table.DataTable(
            id='comparison-table',
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Actual Price", "id": "actual_price"},
                {"name": "Predicted Price", "id": "predicted_price"},
                {"name": "Difference", "id": "difference"},
                {"name": "Accuracy (%)", "id": "accuracy"}
            ],
            style_table={'width': '80%', 'margin': '0 auto'},
            style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
            style_cell={'textAlign': 'center', 'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'}
        )
    ]),
    dcc.Interval(
        id='graph-update',
        interval=300000,  # 5 minutes in milliseconds
        n_intervals=0
    )
])

def register_callbacks(app):
    @app.callback(
        [Output('live-price-graph', 'figure'),
         Output('comparison-table', 'data'),
         Output('graph-update', 'interval')],
        [Input('crypto-pair-dropdown', 'value'),
         Input('interval-dropdown', 'value'),
         Input('graph-update', 'n_intervals')]
    )
    def update_graph_and_table(selected_pair, selected_interval, n_intervals):
        interval_map = {'1m': 60000, '5m': 300000, '60m': 3600000}
        update_interval = interval_map.get(selected_interval, 300000)
        
        if not price_data[selected_pair].empty:
            historical_data = fetch_historical_data(selected_pair, limit=288)
            future_timestamps, predicted_prices = predict_lstm_multi_step(selected_pair)
            actual_timestamps = historical_data['timestamp']
            actual_prices = historical_data['close']
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual_timestamps,
                y=actual_prices,
                mode='lines',
                name="Actual Prices",
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=list(future_timestamps),
                y=[round(price, 2) for price in predicted_prices],
                mode='lines',
                name="Predicted Prices",
                line=dict(dash='dash', color='orange')
            ))
            last_20_actual_prices = actual_prices.tail(20)
            last_20_actual_timestamps = actual_timestamps.tail(20)
            last_20_predicted_prices = [round(price, 2) for price in predicted_prices][:20]
            table_data = [{
                "timestamp": last_20_actual_timestamps.iloc[i],
                "actual_price": f"{last_20_actual_prices.iloc[i]:.2f}",
                "predicted_price": f"{last_20_predicted_prices[i]:.2f}",
                "difference": f"{abs(last_20_actual_prices.iloc[i] - last_20_predicted_prices[i]):.2f}",
                "accuracy": f"{round(100 - (abs(last_20_actual_prices.iloc[i] - last_20_predicted_prices[i]) / last_20_actual_prices.iloc[i]) * 100, 2):.2f}"
            } for i in range(20)]
            fig.update_layout(
                title="Actual vs Predicted Prices",
                xaxis_title="Time",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                showlegend=True
            )
            return fig, table_data, update_interval

        raise PreventUpdate
