from dash import Dash, html

app = Dash(__name__)

app.layout = html.Div("Hello, Dash!")

if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=8050, debug=True)
