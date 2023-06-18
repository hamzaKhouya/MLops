import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
def generate_predict(name):
    # If you what to replicate the DeepLog paper results(Actually, I have a better result than DeepLog paper results),
    # you should use the 'list' not 'set' to obtain the full dataset, I use 'set' just for test and acceleration.
    hdfs = set()
    # hdfs = []
    with open('data/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs.add(tuple(ln))
            # hdfs.append(tuple(ln))
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
model_path= 'model/' + log + '.pt'
model.load_state_dict(torch.load(model_path))
model.eval()
print('model_path: {}'.format(model_path))
test_normal_loader = generate_predict('hdfs_test_normal')
test_abnormal_loader = generate_predict('hdfs_test_abnormal')
TP = 0
FP = 0
# Test the model
start_time = time.time()
with torch.no_grad():
    for line in test_normal_loader:
        for i in range(len(line) - window_size):
            print('Contour #: : {}'.i)
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if label not in predicted:
                FP += 1
                break
with torch.no_grad():
    for line in test_abnormal_loader:
        for i in range(len(line) - window_size):
            print('Contour #: : {}'.i)
            seq = line[i:i + window_size]
            label = line[i + window_size]
            seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
            label = torch.tensor(label).view(-1).to(device)
            output = model(seq)
            predicted = torch.argsort(output, 1)[0][-num_candidates:]
            if label not in predicted:
                TP += 1
                break
elapsed_time = time.time() - start_time
print('elapsed_time: {:.3f}s'.format(elapsed_time))
# Compute precision, recall and F1-measure
FN = len(test_abnormal_loader) - TP
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print('false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(FP, FN, P, R, F1))
print('Finished Predicting')


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

