import dash
from dash import dcc, html, dash_table
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
import io
import base64
import networkx as nx
import time

# Load and preprocess the dataset
df = pd.read_csv('WSNBFSFdataset.csv')
df = df[['Time', 'S_Node', 'Node_id', 'Rest_Energy', 'Trace_Level', 'Mac_Type_Pckt', 'Source_IP_Port', 
         'Des_IP_Port', 'Packet_Size', 'TTL', 'Hop_Count', 'Broadcast_ID', 'Dest_Node_Num', 
         'Dest_Seq_Num', 'Src_Node_ID', 'Src_Seq_Num', 'Class']]
df['Class'] = df['Class'].astype(int)

# Map class labels to human-readable names
class_labels = {0: 'Normal', 1: 'Blackhole', 2: 'Forwarding', 3: 'Flooding'}
df['Class'] = df['Class'].map(class_labels)

# Separate features and target
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Convert any string columns to numeric using LabelEncoder
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        X[column] = label_encoders[column].fit_transform(X[column])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the decision tree
dt_classifier = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = dt_classifier.predict(X_test)

# Evaluation metrics
classification_rep = classification_report(y_test, y_pred, target_names=list(class_labels.values()), output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred, labels=list(class_labels.values()))

# Class distribution in test data
test_class_counts = pd.Series(y_test).value_counts()

# Create confusion matrix heatmap
fig_cm = plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_labels.values()), yticklabels=list(class_labels.values()))
plt.title("Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()

# Convert the confusion matrix plot to base64 string for embedding in Dash
cm_img = io.BytesIO()
fig_cm.savefig(cm_img, format='png')
cm_img.seek(0)
cm_img_b64 = base64.b64encode(cm_img.getvalue()).decode()

# Create the class distribution pie chart
fig_pie = go.Figure(data=[go.Pie(labels=test_class_counts.index, values=test_class_counts.values, hole=0.3)])
fig_pie.update_layout(title="Class Distribution in Test Data", showlegend=True)

# Format the classification report as a table
classification_rep_df = pd.DataFrame(classification_rep).T.reset_index()
classification_rep_df.rename(columns={
    'index': 'Class',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1-score': 'F1-Score',
    'support': 'Support'
}, inplace=True)

classification_rep_df = classification_rep_df[['Class', 'Precision', 'Recall', 'F1-Score', 'Support']]

# Initialize network graph
G = nx.Graph()
unique_nodes = pd.concat([df['S_Node'], df['Node_id'], df['Dest_Node_Num']]).unique()

G.add_nodes_from(unique_nodes)

for _, row in df.iterrows():
    G.add_edge(row['S_Node'], row['Node_id'])
    G.add_edge(row['Node_id'], row['Dest_Node_Num'])

node_positions = nx.spring_layout(G)

# Live simulation state
class SimulationState:
    def __init__(self):
        self.current_step = 0
        self.simulation_active = True

sim_state = SimulationState()

# Dash application
app = dash.Dash(__name__)
app.layout = html.Div(children=[

    html.H1("Network Traffic Classifier Dashboard", style={
        'textAlign': 'center',
        'padding': '20px',
        'color': 'white',
        'fontFamily': 'Arial, sans-serif'
    }),

    html.Div(children=[
        html.H3("Classification Report", style={'textAlign': 'center', 'color': 'red'}),
        dash_table.DataTable(
            columns=[{"name": col, "id": col} for col in classification_rep_df.columns],
            data=classification_rep_df.to_dict('records'),
            style_table={'overflowX': 'auto', 'margin': '20px 0'},
            style_header={
                'backgroundColor': '#333',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#444',
                }
            ],
        ),
    ], style={'padding': '20px'}),

    html.Div(children=[
        html.H3("Class Distribution (Test Data)", style={'textAlign': 'center', 'color': 'red'}),
        dcc.Graph(figure=fig_pie)
    ], style={'padding': '20px'}),

    html.Div(children=[
        html.H3("Confusion Matrix", style={'textAlign': 'center', 'color': 'red'}),
        html.Img(src=f"data:image/png;base64,{cm_img_b64}")
    ], style={'padding': '20px', 'textAlign': 'center'}),

    html.Div(children=[
        html.H3("Live Network State", style={'textAlign': 'center', 'color': 'red'}),
        dash_table.DataTable(
            id='live-table',
            columns=[
                {'name': 'Time', 'id': 'Time'},
                {'name': 'Source Node', 'id': 'S_Node'},
                {'name': 'Target Node', 'id': 'Node_id'},
                {'name': 'Rest Energy', 'id': 'Rest_Energy'},
                {'name': 'Packet Size', 'id': 'Packet_Size'},
                {'name': 'Prediction', 'id': 'Prediction'},
            ],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center', 'padding': '5px'}
        ),
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
    ], style={'padding': '20px'}),

], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#181818', 'padding': '20px'})

@app.callback(
    dash.dependencies.Output('live-table', 'data'),
    [dash.dependencies.Input('interval-component', 'n_intervals')]
)
def update_network_state(n_intervals):
    if sim_state.current_step >= len(df) or not sim_state.simulation_active:
        return []

    row = df.iloc[sim_state.current_step]
    features = row.drop(['Class', 'Time']).to_frame().T
    for column, encoder in label_encoders.items():
        features[column] = encoder.transform(features[column])
    features = features[X.columns]  # Ensure column order matches the training data

    prediction = dt_classifier.predict(features)[0]
    state = {
        'Time': row['Time'],
        'S_Node': row['S_Node'],
        'Node_id': row['Node_id'],
        'Rest_Energy': row['Rest_Energy'],
        'Packet_Size': row['Packet_Size'],
        'Prediction': prediction
    }
    sim_state.current_step += 1
    return [state]


if __name__ == '__main__':
    app.run_server(debug=True)
