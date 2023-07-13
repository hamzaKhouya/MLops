import dash
from dash import html, dcc, Input, Output
#import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import plotly.express as px

x = [20,10, 5,0]
FP = [234, 400, 600, 970]
FN = [40, 103, 200, 500]
TP = [4123 - fn for fn in FN]
P = [tp / (tp + fp) for tp, fp in zip(TP, FP)]
R = [tp / (tp + fn) for tp, fn in zip(TP, FN)]
F1 = [2 * p * r / (p + r) for p, r in zip(P, R)]

LOGO = "https://cd.foundation/wp-content/uploads/sites/78/2020/05/mlopscard.png"
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Col(html.Img(src=LOGO, height="100px",
                             className="float-start")),
            dbc.Collapse(
                [
                    dbc.NavItem(dbc.NavLink("Differents metrics", href="/",
                                            active="exact", className="btn btn-outline-info m-1")),
                ],
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
            dbc.Col(
            dcc.Dropdown(
                id="metric",
                options=["Precision","Recall","F1-Score"],
                multi=True,
                placeholder="Metrics",
                style = {'float' : 'left' , 'width' : '300px' }
            ),
            className="mt-1"
        )
        ], fluid=True
    ),
    className="my-1",
    dark=True,
)

figures = dbc.Row([
                 dbc.Col(dcc.Graph(id = 'precision'),width={'size':4 , "order": 1}, 
                                 style = {'border-style': 'solid', 'border-color': 'gray'}),
                 dbc.Col(dcc.Graph(id = 'recall'),width={'size': 4, "order": 2},
                              style = {'border-style': 'solid', 'border-color': 'gray'}),
                 dbc.Col(dcc.Graph(id = 'f1-score'),width={'size': 4, "order": 2},
                              style = {'border-style': 'solid', 'border-color': 'gray'})])

# Create Dash app and layout
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

app.layout = html.Div(children=[navbar, figures])

@app.callback(
    [Output('precision', 'figure'),
     Output('recall', 'figure'),
     Output('f1-score', 'figure'),
     Input('metric' ,'value')
     ])
def update_graph(selected_metric):
    
    precision = px.line(x=x,y=P)
 
    recall = px.line(x=x,y=R)
    
    f1_score = px.line(x=x,y=F1)
                            
    return precision, recall, f1_score
    
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
