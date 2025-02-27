import os
import logging
import numpy as np
from datetime import datetime
from PIL import Image
from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
from dash.exceptions import PreventUpdate
from tensorflow.keras.models import load_model

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Persistent log file path
LOG_FILE = "predictions.log"
with open(LOG_FILE, "a") as f:
    pass

# Load the pre-trained handwriting model
try:
    handwriting_model = load_model('handwriting_model.h5', compile=False)
except Exception as e:
    logging.error(f"Error loading handwriting model: {e}")
    raise

def process_canvas_json(json_data):
    """ Convert DashCanvas json_data into a properly formatted 28x28 grayscale image. """
    if not json_data:
        print("❌ No data received from the canvas.")
        return None

    try:
        image_array = parse_jsonstring(json_data, shape=(280, 280))

        # Fix: Clip out-of-bounds indices to stay within [0, 279]
        image_array = np.clip(image_array, 0, 255)

        print(f"✅ Parsed image shape: {image_array.shape}")

        # Ensure stroke visibility by increasing contrast
        image_array = image_array.astype('float32')
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-6) * 255
        image_array = image_array.astype('uint8')

        # Invert colors for MNIST compatibility (white digit on black background)
        image_array = 255 - image_array

        # Convert to grayscale PIL image and resize to 28x28
        im = Image.fromarray(image_array, mode='L').resize((28, 28))
        im.save("processed_canvas.png")  # Save for debugging

        # Convert to NumPy array and normalize
        img_array = np.array(im).astype('float32') / 255.0
        img_array = 1 - img_array  # Ensure white strokes on black background

        print(f"✅ Processed Image Shape: {img_array.shape}")  # Debugging output
        return img_array.reshape(1, 28, 28, 1)
    
    except IndexError as e:
        print(f"🚨 IndexError: {e}. Fixing out-of-bounds coordinates.")
        return None  # Avoid crashing, return None instead



# Layout
handwriting_layout = html.Div([
    html.H2("Handwriting Number Prediction Model (FinBert)", style={'text-align': 'center'}),
    DashCanvas(
        id='canvas',
        width=280,
        height=280,
        lineWidth=10,
        lineColor='black',
        hide_buttons=["zoom", "pan", "line", "rectangle", "select"]
    ),
    html.Div([
        html.Button('Predict', id='predict-button', n_clicks=0,
                    style={'marginTop': '20px', 'marginRight': '10px'}),
        html.Button('Clear Canvas', id='clear-button', n_clicks=0,
                    style={'marginTop': '20px'})
    ]),
    html.Div(id='prediction-output', style={'fontSize': 24, 'marginTop': '20px', 'textAlign': 'center'}),
    
    # Feedback Section (Initially hidden)
    html.Div(id='feedback-section', style={'display': 'none'}, children=[
        html.P("Was the prediction correct?"),
        html.Button('Yes', id='feedback-yes', n_clicks=0, style={'marginRight': '10px'}),
        html.Button('No', id='feedback-no', n_clicks=0),
        html.Div(id='correction-section', style={'display': 'none'}, children=[
            html.P("Please enter the correct digit:"),
            dcc.Input(id='correct-digit', type='text', maxLength=1),
            html.Button('Submit Correction', id='submit-correction', n_clicks=0)
        ])
    ]),
    
    dcc.Interval(id='log-interval', interval=5000, n_intervals=0),
    html.Div(id='log-output', style={'marginTop': '20px', 'textAlign': 'left', 'whiteSpace': 'pre-wrap'}),
    html.Div(id='dummy', style={'display': 'none'})  # Hidden dummy div for clientside refresh
])

def register_callbacks(app):
    @app.callback(
        [Output('prediction-output', 'children'),
         Output('feedback-section', 'style'),
         Output('correction-section', 'style')],
        [Input('predict-button', 'n_clicks'),
         Input('feedback-yes', 'n_clicks'),
         Input('feedback-no', 'n_clicks'),
         Input('submit-correction', 'n_clicks')],
        [State('canvas', 'json_data'),
         State('correct-digit', 'value'),
         State('prediction-output', 'children')]
    )
    def handle_prediction_and_feedback(predict_clicks, yes_clicks, no_clicks, correction_clicks,
                                       json_data, correct_digit, prediction_text):
        ctx = callback_context
        if not ctx.triggered:
            print("❌ No button was clicked.")
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"✅ Button clicked: {trigger_id}")

        if trigger_id == 'predict-button':
            if not json_data:
                print("❌ No drawing detected in json_data!")
                return "Please draw a character first.", {'display': 'none'}, {'display': 'none'}

            print(f"✅ json_data received: {json_data[:100]}")  # Debugging output
            img_array = process_canvas_json(json_data)
            if img_array is None:
                return "Error processing image.", {'display': 'none'}, {'display': 'none'}

            pred = handwriting_model.predict(img_array)
            pred_digit = int(np.argmax(pred))
            pred_message = f"Predicted Character: {pred_digit}"

            print(f"✅ Model Prediction: {pred_message}")

            # Log the prediction persistently
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {pred_message} (Awaiting Feedback)\n")

            return pred_message, {'display': 'block'}, {'display': 'none'}

        elif trigger_id == 'feedback-yes':
            with open(LOG_FILE, "a") as f:
                f.write(f"✅ Prediction Confirmed: {prediction_text}\n")
            return prediction_text, {'display': 'none'}, {'display': 'none'}

        elif trigger_id == 'feedback-no':
            print("✅ Showing correction section.")
            return prediction_text, {'display': 'block'}, {'display': 'block'}

        elif trigger_id == 'submit-correction' and correct_digit and correct_digit.isdigit():
            with open(LOG_FILE, "a") as f:
                f.write(f"❌ Incorrect Prediction: {prediction_text}. Corrected to: {correct_digit}\n")
            return f"Corrected Character: {correct_digit}", {'display': 'none'}, {'display': 'none'}

        return PreventUpdate

    @app.callback(
        Output('log-output', 'children'),
        [Input('log-interval', 'n_intervals')]
    )
    def update_log(n_intervals):
        try:
            with open(LOG_FILE, "r") as f:
                log_text = f.read()
        except Exception as e:
            log_text = f"Error reading log file: {e}"
        return log_text

    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) {
                return window.dash_clientside.no_update;
            }
            window.location.reload();
            return "";
        }
        """,
        Output('dummy', 'children'),
        Input('clear-button', 'n_clicks')
    )
