from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
class SimulationState:
    def __init__(self):
        self.current_step = 0
        self.simulation_active = False
        self.data = None
        self.model = None
        self.node_positions = None
        self.label_encoders = {}

sim_state = SimulationState()

def load_sample_data():
    """Create sample data if no CSV is provided"""
    return pd.DataFrame({
        'Time': range(10),
        'S_Node': [f'N{i}' for i in range(10)],
        'Node_id': [f'N{i+1}' for i in range(10)],
        'Rest_Energy': np.random.uniform(0, 100, 10),
        'Trace_Level': np.random.randint(0, 5, 10),
        'Mac_Type_Pckt': ['type_' + str(i) for i in range(10)],
        'Source_IP_Port': [f'192.168.1.{i}:80' for i in range(10)],
        'Des_IP_Port': [f'192.168.1.{i+1}:80' for i in range(10)],
        'Packet_Size': np.random.randint(100, 1000, 10),
        'TTL': np.random.randint(1, 64, 10),
        'Hop_Count': np.random.randint(1, 10, 10),
        'Broadcast_ID': np.random.randint(1000, 9999, 10),
        'Dest_Node_Num': [f'N{i+2}' for i in range(10)],
        'Dest_Seq_Num': np.random.randint(1, 100, 10),
        'Src_Node_ID': [f'N{i}' for i in range(10)],
        'Src_Seq_Num': np.random.randint(1, 100, 10),
        'Class': np.random.randint(0, 4, 10)
    })

def load_data(file_path='WSNBFSFdataset.csv'):
    try:
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            logger.warning(f"Data file not found at {file_path}. Using sample data.")
            return load_sample_data()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return load_sample_data()

def initialize_simulation():
    try:
        logger.info("Initializing simulation...")
        
        # Load data
        sim_state.data = load_data()
        
        # Prepare features
        X = sim_state.data.drop(['Class', 'Time'], axis=1)
        y = sim_state.data['Class']
        
        # Handle categorical variables
        categorical_columns = ['S_Node', 'Node_id', 'Mac_Type_Pckt', 'Source_IP_Port', 
                             'Des_IP_Port', 'Dest_Node_Num', 'Src_Node_ID']
        
        for column in categorical_columns:
            sim_state.label_encoders[column] = LabelEncoder()
            X[column] = sim_state.label_encoders[column].fit_transform(X[column])
        
        # Train model
        sim_state.model = DecisionTreeClassifier(max_depth=10, random_state=42)
        sim_state.model.fit(X, y)
        
        # Create network graph
        G = nx.Graph()
        unique_nodes = pd.concat([
            sim_state.data['S_Node'],
            sim_state.data['Node_id'],
            sim_state.data['Dest_Node_Num']
        ]).unique()
        
        G.add_nodes_from(unique_nodes)
        
        # Add edges
        for _, row in sim_state.data.iterrows():
            G.add_edge(row['S_Node'], row['Node_id'])
            G.add_edge(row['Node_id'], row['Dest_Node_Num'])
        
        # Calculate layout
        sim_state.node_positions = nx.spring_layout(G)
        pos_dict = {str(node): {'x': float(pos[0]), 'y': float(pos[1])} 
                   for node, pos in sim_state.node_positions.items()}
        
        logger.info("Simulation initialized successfully")
        return pos_dict
    
    except Exception as e:
        logger.error(f"Error initializing simulation: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_simulation')
def start_simulation():
    sim_state.simulation_active = True
    sim_state.current_step = 0
    return jsonify({'status': 'started'})

@app.route('/stop_simulation')
def stop_simulation():
    sim_state.simulation_active = False
    return jsonify({'status': 'stopped'})

@app.route('/get_network_state')
def get_network_state():
    try:
        if sim_state.current_step >= len(sim_state.data):
            return jsonify({'status': 'completed'})
        print("ok1")
        
        row = sim_state.data.iloc[sim_state.current_step]
        print("ok2")
        # Prepare features for prediction
        features = row.drop(['Class', 'Time'])
        print("ok3")
        
        # Encode categorical features
        for column, encoder in sim_state.label_encoders.items():
            features[column] = encoder.transform([features[column]])[0]
        print("ok4")
        
        # Make prediction
        prediction = int(sim_state.model.predict([features])[0])
        print("ok5")
        
        state = {
            'source_node': str(row['S_Node']),
            'target_node': str(row['Node_id']),
            'dest_node': str(row['Dest_Node_Num']),
            'prediction': prediction,
            'time_step': row['Time'],
            'rest_energy': float(row['Rest_Energy']),
            'packet_size': int(row['Packet_Size']),
            'hop_count': int(row['Hop_Count'])
        }
        print("ok6")
        sim_state.current_step += 1
        print("ok7")
        return jsonify(state)
    
    except Exception as e:
        logger.error(f"Error getting network state: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_initial_layout')
def get_initial_layout():
    try:
        layout = initialize_simulation()
        return jsonify(layout)
    except Exception as e:
        logger.error(f"Error getting initial layout: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, port=5000)