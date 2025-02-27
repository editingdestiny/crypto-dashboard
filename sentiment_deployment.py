import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from dash import html, dcc, Input, Output, dash_table

# Load Sentiment Data
FILE_PATH = "sentiment_results.csv"

def load_data():
    """Load and return the latest sentiment analysis results."""
    try:
        df = pd.read_csv(FILE_PATH)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Country", "Headline", "Sentiment", "Confidence"])  # Return empty df if file not found

df = load_data()

# Define Sentiment Dashboard Layout
sentiment_layout = dbc.Container([
    html.H2("Sentiment Analysis", style={"textAlign": "center", "color": "#00ccff"}),

    # Dropdown for Country Selection
    dbc.Row([
        dbc.Col([
            html.Label("Select a Country:", style={"color": "white"}),
            dcc.Dropdown(
                id="country-dropdown",
                options=[{"label": c, "value": c} for c in sorted(df["Country"].unique())],
                value=None,  # Default to None (All)
                placeholder="Select a country",
                clearable=True,
                style={"color": "black"}
            )
        ], width=4),
    ], justify="center"),
    
    html.Br(),

    # Row for Graphs and Data Table Side-by-Side
    dbc.Row([
        # Wider Data Table (6 columns)
        dbc.Col([
            html.H4("Headlines & Sentiments", style={"color": "#00ccff"}),
            dash_table.DataTable(
                id="articles-table",
                columns=[
                    {"name": "Headline", "id": "Headline"},
                    {"name": "Sentiment", "id": "Sentiment"},
                    {"name": "Confidence", "id": "Confidence"}
                ],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#1E1E1E", "color": "#00ccff"},
                style_data={"backgroundColor": "#2a2a2a", "color": "white"},
                style_cell={
                    "textAlign": "left",  # Align text to the left
                    "whiteSpace": "normal",  # Enable wrapping
                    "height": "auto",  # Allow auto height
                    "maxWidth": "400px",  # Increased width
                    "wordWrap": "break-word"  # Ensure words break properly
                },
                style_data_conditional=[
                    {
                        "if": {"column_id": "Headline"},
                        "whiteSpace": "normal",
                        "height": "auto",
                        "maxWidth": "400px",  # Increase column width for headlines
                        "wordWrap": "break-word"
                    }
                ],
                page_size=10
            )
        ], width=6),  # Increased width from 4 to 6

        # Narrower Graphs (6 columns)
        dbc.Col([
            dcc.Graph(id="sentiment-country"),
            dcc.Graph(id="sentiment-distribution"),
            dcc.Graph(id="confidence-boxplot")
        ], width=6),  # Reduced width from 8 to 6
    ]),

    # Interval Component for Refreshing Data
    dcc.Interval(
        id="interval-component",
        interval=2 * 60 * 60 * 1000,  # Refresh every 2 hours
        n_intervals=0
    )
], fluid=True)

# Callback Function to Refresh Data & Filter by Country
def register_callbacks(app):
    @app.callback(
        [Output("sentiment-distribution", "figure"),
         Output("confidence-boxplot", "figure"),
         Output("sentiment-country", "figure"),
         Output("articles-table", "data")],
        [Input("interval-component", "n_intervals"),
         Input("country-dropdown", "value")]
    )
    def update_graphs(n_intervals, selected_country):
        df = load_data()  # Reload data

        # Filter data by selected country if applicable
        if selected_country:
            df = df[df["Country"] == selected_country]

        # Sentiment Graphs
        fig_sentiment_country = px.bar(df, x="Country", color="Sentiment", title="Sentiment by Country", barmode="group")
        fig_sentiment = px.histogram(df, x="Sentiment", title="Sentiment Distribution", color="Sentiment", barmode="group")
        fig_confidence = px.box(df, x="Sentiment", y="Confidence", title="Confidence Score Distribution", color="Sentiment")

        # Convert Data for Table
        table_data = df.to_dict("records")

        return fig_sentiment, fig_confidence, fig_sentiment_country, table_data
