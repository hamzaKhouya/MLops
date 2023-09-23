import torch
import torch.nn as nn
import time
import argparse
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import os
import pandas as pd

# Device configuration
device = torch.device("cpu")


def generate(name):
    hdfs = set()
    with open(os.path.join(os.path.abspath('data/' + name)), 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(hdfs)))
    return hdfs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Hyperparameters
num_classes = 28
input_size = 1
num_epochs = 10
batch_size = 256
model_path = os.path.join(os.path.abspath('model/Adam_batch_size={}_epoch={}.pt'.format(str(batch_size), str(num_epochs))))
parser = argparse.ArgumentParser()
parser.add_argument('-num_layers', default=2, type=int)
parser.add_argument('-hidden_size', default=64, type=int)
parser.add_argument('-window_size', default=10, type=int)
parser.add_argument('-num_candidates', default=9, type=int)
args = parser.parse_args()
num_layers = args.num_layers
hidden_size = args.hidden_size
window_size = args.window_size
num_candidates = args.num_candidates

model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print('model_path: {}'.format(model_path))
test_normal_loader = generate('hdfs_test_normal')
test_abnormal_loader = generate('hdfs_test_abnormal')

TP_list = []
FP_list = []
FN_list = []

# Store prediction values for each Iteration
predictions_normal = []
predictions_abnormal = []

# Test the model
start_time = time.time()
with torch.no_grad():
    for Iteration in range(num_epochs):
        TP = 0
        FP = 0

        if Iteration == 0:
            test_normal_loader_Iteration = list(test_normal_loader)[:window_size]
            test_abnormal_loader_Iteration = list(test_abnormal_loader)[:window_size]
        else:
            test_normal_loader_Iteration = list(test_normal_loader)[:window_size + Iteration]
            test_abnormal_loader_Iteration = list(test_abnormal_loader)[:window_size + Iteration]

        for line in test_normal_loader_Iteration:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    FP += 1
                    break
            predictions_normal.append(output.tolist())

        for line in test_abnormal_loader_Iteration:
            for i in range(len(line) - window_size):
                seq = line[i:i + window_size]
                label = line[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    TP += 1
                    break
            predictions_abnormal.append(output.tolist())

        FN = len(test_abnormal_loader_Iteration) - TP
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)

elapsed_time = time.time() - start_time
print('elapsed_time: {:.3f}s'.format(elapsed_time))


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
                              style = {'border-style': 'solid', 'border-color': 'gray'})
                 ])

# Create Dash app and layout
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME])

app.layout = html.Div(children=[navbar, figures])
# Compute precision, recall, and F1-measure for each Iteration
precision_list = [TP / (TP + FP) for TP, FP in zip(TP_list, FP_list)]
recall_list = [TP / (TP + FN) for TP, FN in zip(TP_list, FN_list)]
f1_score_list = [2 * P * R / (P + R) for P, R in zip(precision_list, recall_list)]
data = {
    'Iteration': list(range(num_epochs)),
    'Precision': precision_list,
    'Recall': recall_list,
    'F1-score': f1_score_list
}

df = pd.DataFrame(data)
new_data = df.round(2)

@app.callback(
    [Output('precision', 'figure'),
     Output('recall', 'figure'),
     Output('f1-score', 'figure'),
     Input('metric' ,'value')
     ])
def update_graph(selected_metric):
    
    precision = px.line(new_data,x='Iteration',y='Precision', markers=True,
                        title ='Precision du Modèle', text="Precision",
                        template='ggplot2')
    
    precision.update_traces(textposition="bottom right")
 
    recall = px.line(new_data,x='Iteration',y='Recall', markers=True,
                     title ='Recall du Modèle', text="Recall",
                     template='ggplot2')
    
    recall.update_traces(textposition="bottom right")
    
    f1_score = px.line(new_data,x='Iteration',y='F1-score', markers=True,
                       title ='F1-Score du Modèle',text="F1-score",
                       template='ggplot2')
    
    f1_score.update_traces(textposition="top left")
    
    return precision, recall, f1_score
    
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')
