import numpy as np
import dash
from dash import html, dcc
import plotly.graph_objs as go

# Data for bar chart
PCA = (0.98, 0.67, 0.79)
LSTM = (0.9526, 0.9903, 0.9711)

# Data for line chart
x = [8, 9, 10, 11]
FP = [605, 588, 495, 860]
FN = [465, 333, 108, 237]
TP = [4123 - fn for fn in FN]
P = [tp / (tp + fp) for tp, fp in zip(TP, FP)]
R = [tp / (tp + fn) for tp, fn in zip(TP, FN)]
F1 = [2 * p * r / (p + r) for p, r in zip(P, R)]

# Create bar chart trace
trace1 = go.Bar(
    x=['Precision', 'Recall', 'F1-score'],
    y=PCA,
    name='PCA',
    marker=dict(color='blue', opacity=0.4)
)

trace2 = go.Bar(
    x=['Precision', 'Recall', 'F1-score'],
    y=LSTM,
    name='LSTM',
    marker=dict(color='red', opacity=0.4)
)

# Create line chart trace
trace3 = go.Scatter(
    x=x,
    y=P,
    mode='lines',
    name='Precision',
    line=dict(color='red', dash='dot')
)

trace4 = go.Scatter(
    x=x,
    y=R,
    mode='lines',
    name='Recall',
    line=dict(color='blue', dash='dot')
)

trace5 = go.Scatter(
    x=x,
    y=F1,
    mode='lines',
    name='F1-score',
    line=dict(color='black', dash='dot')
)

# Create Dash app and layout
app = dash.Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Scores by different models'),
    html.Div(children=[
        dcc.Graph(
            id='bar-chart',
            figure={
                'data': [trace1, trace2],
                'layout': go.Layout(
                    title='Scores by different models',
                    xaxis=dict(title='Measure'),
                    yaxis=dict(title='Scores'),
                    barmode='group'
                )
            }
        )
    ]),
    html.Div(children=[
        dcc.Graph(
            id='line-chart',
            figure={
                'data': [trace3, trace4, trace5],
                'layout': go.Layout(
                    title='Scores by window size',
                    xaxis=dict(title='window_size'),
                    yaxis=dict(title='scores'),
                    legend=dict(
                        x=0.7,
                        y=1
                    ),
                    xaxis_tickvals=x,
                    yaxis_range=[0.5, 1]
                )
            }
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
