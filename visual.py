import dash
from dash import dcc, html
import plotly.graph_objs as go

from LogKeyModel_predict import (P, R, F1)

# Create bar chart trace
trace1 = go.Bar(
    x=['Precision', 'Recall', 'F1-score'],
    y=[P/100, R/100, F1/100],
    name='Your Model',
    marker=dict(color='blue', opacity=0.4)
)

# Create Dash app and layout
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Scores by different models'),
    html.Div(children=[
        dcc.Graph(
            id='bar-chart',
            figure={
                'data': [trace1],
                'layout': go.Layout(
                    title='Scores by different models',
                    xaxis=dict(title='Measure'),
                    yaxis=dict(title='Scores'),
                    barmode='group'
                )
            }
        )
    ])
])

# if __name__ == '__main__':
#     app.run_server(debug=False)
