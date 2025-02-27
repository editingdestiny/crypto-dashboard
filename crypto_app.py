import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output
from crypto_dashboard import crypto_dashboard_layout, register_callbacks as register_crypto_callbacks
from handwriting_model import handwriting_layout, register_callbacks as register_handwriting_callbacks
from sentiment_deployment import sentiment_layout, register_callbacks as register_sentiment_callbacks

# Initialize Dash app with Bootstrap styling
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], suppress_callback_exceptions=True)  # Change theme as needed
server = app.server

# Custom CSS Styling for Tabs
TAB_STYLE = {
    "padding": "15px",
    "fontSize": "18px",
    "fontWeight": "bold",
    "borderBottom": "3px solid transparent",
    "transition": "border-bottom 0.3s ease-in-out",
}

SELECTED_TAB_STYLE = {
    "backgroundColor": "#1E1E1E",
    "borderBottom": "3px solid #00ccff",
    "color": "#00ccff",
    "boxShadow": "0px 4px 10px rgba(0, 255, 255, 0.3)",
}

# Custom Layout with Modern Tabs
app.layout = html.Div([
    html.H1("SD Predicts", style={"textAlign": "center", "marginBottom": "20px", "color": "#00ccff"}),
    
    dbc.Tabs(
        [
            dbc.Tab(label="Crypto Dashboard", tab_id="crypto_dashboard", tab_style=TAB_STYLE,
                    active_tab_style=SELECTED_TAB_STYLE),
            dbc.Tab(label="Handwriting Model", tab_id="handwriting", tab_style=TAB_STYLE,
                    active_tab_style=SELECTED_TAB_STYLE),
            dbc.Tab(label="Sentiment Analysis", tab_id="sentiment_analysis", tab_style=TAB_STYLE, active_tab_style=SELECTED_TAB_STYLE),
        ],
        id="tabs",
        active_tab="crypto_dashboard",
        style={"backgroundColor": "#1E1E1E", "borderRadius": "10px", "padding": "10px"}
    ),

    html.Div(id="tab-content", style={"padding": "20px"})
])

# Callback to switch between tabs
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_content(active_tab):
    if active_tab == "handwriting":
        return handwriting_layout
    elif active_tab == "sentiment_analysis":
        return sentiment_layout  # Return Sentiment Layout
    return crypto_dashboard_layout

# Register callbacks for individual modules
register_crypto_callbacks(app)
register_handwriting_callbacks(app)
register_sentiment_callbacks(app)  

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=5000, debug=True)
