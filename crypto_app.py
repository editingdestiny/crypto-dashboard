import logging
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import ccxt
import pandas as pd
from datetime import datetime
import threading
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from dash.exceptions import PreventUpdate
import os
from threading import Lock

data_lock = Lock()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Check if the model exists, if not, build it
if os.path.exists('crypto_lstm_model.h5'):
    lstm_model = load_model('crypto_lstm_model.h5')
else:
    # Assuming x_train and y_train are pre-defined or handled elsewhere
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=20, batch_size=32)
    model.save('crypto_lstm_model.h5')
    lstm_model = load_model('crypto_lstm_model.h5')

# Initialize app and exchange
app = Dash(__name__)
server = app.server
exchange = ccxt.binance({
    'rateLimit': 1200,
    'timeout': 30000,  # Timeout set to 30 seconds
})

CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# Global data storage
price_data = {pair: pd.DataFrame(columns=["timestamp", "price", "predicted_price", "difference", "accuracy"]) for pair in CRYPTO_PAIRS}

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)  # Predict the next price
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def fetch_historical_data(pair, limit=500):
    ohlcv = exchange.fetch_ohlcv(pair, timeframe='1m', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'close']]

def fetch_price(pair):
    ticker = exchange.fetch_ticker(pair)
    return {
        "timestamp": datetime.fromtimestamp(ticker['timestamp'] / 1000),
        "price": ticker['last']
    }

def predict_lstm(pair, sequence_length=50):
    historical_data = fetch_historical_data(pair, limit=sequence_length)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(historical_data['close'].values.reshape(-1, 1))
    x_input = np.array([scaled_data[-sequence_length:]])
    predicted_price = lstm_model.predict(x_input)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

def data_updater():
    global price_data
    while True:
        for pair in CRYPTO_PAIRS:
            new_data = fetch_price(pair)
            predicted_price = predict_lstm(pair)
            actual_price = new_data["price"]
            difference = actual_price - predicted_price
            accuracy = (1 - abs(difference / actual_price)) * 100

            # Update the price_data with actual, predicted, and calculated values
            with data_lock:
                price_data[pair] = pd.concat([
                    price_data[pair],
                    pd.DataFrame([{
                        "timestamp": new_data["timestamp"],
                        "price": actual_price,
                        "predicted_price": predicted_price,
                        "difference": difference,
                        "accuracy": accuracy
                    }])
                ]).tail(100)  # Keep the last 100 data points

        time.sleep(120)  # Update every 2 minutes

# Start the data updater thread
threading.Thread(target=data_updater, daemon=True).start()

# Layout of the app
app.layout = html.Div([
# Embed OpenWebUI chatbox in an iframe
    html.Div([
        html.Iframe(
            src='http://localhost:8080',
            width='100%',
            height='600px',
            style={'border': 'none'}
        )
    ]),
    html.H1("Crypto Price Tracker", style={'text-align': 'center'}),
    dcc.Dropdown(
        id='crypto-pair-dropdown',
        options=[{'label': pair, 'value': pair} for pair in CRYPTO_PAIRS],
        value='BTC/USDT',
        style={'width': '50%', 'margin': '10 auto', 'backgroundColor': 'yellow', 'color': 'white'}
    ),
    dcc.Graph(id='live-price-graph'),
    html.Br(),
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
        interval=120000,  # Update every 120 seconds
        n_intervals=0
    ),
])

# Callback to update the graph and table
@app.callback(
    [Output('live-price-graph', 'figure'),
     Output('comparison-table', 'data')],
    [Input('crypto-pair-dropdown', 'value'),
     Input('graph-update', 'n_intervals')]
)
def update_graph_and_table(selected_pair, n_intervals):
    global price_data

    if not price_data[selected_pair].empty:
        # Create the live price graph
        fig = go.Figure()

        # Live prices
        fig.add_trace(go.Scatter(
            x=price_data[selected_pair]['timestamp'],
            y=price_data[selected_pair]['price'],
            mode='lines',
            name="Live Price"
        ))

        # Predict price for the next 2 minutes
        predicted_price = predict_lstm(selected_pair)
        future_timestamp = price_data[selected_pair]['timestamp'].iloc[-1] + pd.Timedelta(minutes=2)

        # Predicted prices
        fig.add_trace(go.Scatter(
            x=[price_data[selected_pair]['timestamp'].iloc[-1], future_timestamp],
            y=[price_data[selected_pair]['price'].iloc[-1], predicted_price],
            mode='lines',
            name="Predicted Price",
            line=dict(dash='dot', color='orange')
        ))

        # Extend X-axis range for better visualization
        x_min = price_data[selected_pair]['timestamp'].iloc[-1] - pd.Timedelta(minutes=10)
        x_max = future_timestamp

        # Update graph layout
        fig.update_layout(
            title=f"Live and Predicted Prices for {selected_pair}",
            xaxis=dict(
                title="Time",
                tickformat="%H:%M",  # Show hour:minute only
                dtick=60000,  # Tick every minute
                range=[x_min, x_max],
            ),
            yaxis=dict(
                title="Price",
                tickformat=".2f"  # Show full prices with 2 decimal points
            ),
            template="plotly_dark"
        )

        # Prepare data for the table
        last_10_prices = price_data[selected_pair].tail(10)
        table_data = []
        for i, row in last_10_prices.iterrows():
            predicted_price = predict_lstm(selected_pair)
            actual_price = row['price']
            table_data.append({
                "timestamp": row['timestamp'],
                "actual_price": actual_price,
                "predicted_price": predicted_price,
                "difference": abs(actual_price - predicted_price),
                "accuracy": 100 - (abs(actual_price - predicted_price) / actual_price) * 100
            })

        return fig, table_data

    raise PreventUpdate
# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000, debug=True)
