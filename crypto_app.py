import logging
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import threading
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from dash.exceptions import PreventUpdate
import os

# Thread safety lock for shared data access
data_lock = threading.Lock()

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load or build the LSTM model
def load_or_build_model():
    if os.path.exists('crypto_lstm_model.h5'):
        logging.info("Loading pre-trained LSTM model...")
        return load_model('crypto_lstm_model.h5')
    else:
        logging.info("No pre-trained model found. Building a new LSTM model...")
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(50, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        # Placeholder for training (x_train and y_train need to be defined)
        # model.fit(x_train, y_train, epochs=20, batch_size=32)
        model.save('crypto_lstm_model.h5')
        return model

lstm_model = load_or_build_model()

# Initialize Dash app and Binance exchange client
app = dash.Dash(__name__)
server = app.server

exchange = ccxt.binance({
    'rateLimit': 1200,
    'timeout': 30000,
})

CRYPTO_PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

# Global storage for price data
price_data = {pair: pd.DataFrame(columns=["timestamp", "price", "predicted_price", "difference", "accuracy"]) for pair in CRYPTO_PAIRS}

# Fetch historical data from Binance
def fetch_historical_data(pair, limit=500):
    try:
        ohlcv = exchange.fetch_ohlcv(pair, timeframe='1m', limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'close']]
    except Exception as e:
        logging.error(f"Error fetching historical data for {pair}: {e}")
        return pd.DataFrame(columns=['timestamp', 'close'])

# Fetch the latest price from Binance
def fetch_price(pair):
    try:
        ticker = exchange.fetch_ticker(pair)
        return {
            "timestamp": datetime.fromtimestamp(ticker['timestamp'] / 1000),
            "price": ticker['last']
        }
    except Exception as e:
        logging.error(f"Error fetching live price for {pair}: {e}")
        return None

# Predict the next price using the LSTM model
def predict_lstm(pair, sequence_length=50):
    try:
        historical_data = fetch_historical_data(pair, limit=sequence_length)
        if historical_data.empty:
            raise ValueError("Insufficient historical data for prediction.")
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(historical_data['close'].values.reshape(-1, 1))
        
        x_input = np.array([scaled_data[-sequence_length:]])
        predicted_price = lstm_model.predict(x_input)
        predicted_price = scaler.inverse_transform(predicted_price)
        
        return predicted_price[0][0]
    except Exception as e:
        logging.error(f"Error predicting price for {pair}: {e}")
        return None

# Background thread to update data periodically
def data_updater():
    global price_data
    
    while True:
        for pair in CRYPTO_PAIRS:
            new_data = fetch_price(pair)
            if new_data is None:
                continue
            
            predicted_price = predict_lstm(pair)
            if predicted_price is None:
                continue
            
            actual_price = new_data["price"]
            difference = actual_price - predicted_price
            accuracy = (1 - abs(difference / actual_price)) * 100
            
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
                ]).tail(100)  # Keep only the last 100 records
        
        time.sleep(120)  # Update every 2 minutes

threading.Thread(target=data_updater, daemon=True).start()

# App layout
app.layout = html.Div([
    html.H1("Crypto Price Tracker", style={'text-align': 'center'}),
    
    dcc.Dropdown(
        id='crypto-pair-dropdown',
        options=[{'label': pair, 'value': pair} for pair in CRYPTO_PAIRS],
        value='BTC/USDT',
        style={'width': '50%', 'margin': '10px auto'}
    ),
    
    dcc.Graph(id='live-price-graph'),
    
    html.Div([
        html.H3("Actual vs Predicted Table", style={'text-align': 'center'}),
        
        dash_table.DataTable(
            id='comparison-table',
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Actual Price", "id": "price"},
                {"name": "Predicted Price", "id": "predicted_price"},
                {"name": "Difference", "id": "difference"},
                {"name": "Accuracy (%)", "id": "accuracy"}
            ],
            style_table={'width': '80%', 'margin': '0 auto'},
            style_header={'backgroundColor': '#333', 'color': '#fff'},
            style_cell={'textAlign': 'center'}
        )
    ]),
    
    dcc.Interval(
        id='graph-update',
        interval=120000,
        n_intervals=0
    )
])

# Callbacks to update graph and table
@app.callback(
    [Output('live-price-graph', 'figure'),
     Output('comparison-table', 'data')],
    [Input('crypto-pair-dropdown', 'value'),
     Input('graph-update', 'n_intervals')]
)
def update_graph_and_table(selected_pair, n_intervals):
    global price_data
    
    with data_lock:
        if not price_data[selected_pair].empty:
            df = price_data[selected_pair]
            
            # Create graph figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['price'],
                mode='lines',
                name="Actual Price"
            ))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['predicted_price'],
                mode='lines+markers',
                name="Predicted Price",
                line=dict(dash='dot')
            ))
            
            fig.update_layout(
                title=f"Live and Predicted Prices for {selected_pair}",
                xaxis_title="Time",
                yaxis_title="Price",
                template="plotly_dark"
            )
            
            # Prepare table data (last 10 rows)
            table_data = df.tail(10).to_dict('records')
            
            return fig, table_data
    
    raise PreventUpdate

# Run the app
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000, debug=True)

