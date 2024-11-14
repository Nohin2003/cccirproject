import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

def build_network_traffic_classifier(data):
    # Separate features and target
    X = data.drop(['Class'], axis=1)
    y = data['Class']
    
    # Convert any string columns to numeric using LabelEncoder
    label_encoders = {}
    for column in X.columns:
        if X[column].dtype == 'object':
            label_encoders[column] = LabelEncoder()
            X[column] = label_encoders[column].fit_transform(X[column])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train the decision tree
    dt_classifier = DecisionTreeClassifier(
        max_depth=10,          # Prevent overfitting
        min_samples_split=5,   # Minimum samples required to split
        min_samples_leaf=2,    # Minimum samples required at leaf node
        random_state=42
    )
    
    dt_classifier.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = dt_classifier.predict(X_test)
    
    # Calculate feature importance
    feature_importance = dict(zip(X.columns, dt_classifier.feature_importances_))
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Evaluation metrics
    evaluation_metrics = {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return dt_classifier, feature_importance, evaluation_metrics

def plot_results(feature_importance, confusion_mat, class_counts):
    # Function to plot class distribution as a pie chart
    def plot_class_distribution(class_counts):
        # Mapping class numbers to descriptive labels
        class_labels = {0: 'Normal', 1: 'Blackhole', 2: 'Forwarding', 3: 'Flooding'}
        
        # Generate descriptive labels for the pie chart
        labels = [f"{class_labels[int(c)]} ({class_counts[c]})" for c in class_counts.index]
        
        # Plot class distribution as a pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(
            class_counts.values, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=sns.color_palette("viridis", len(class_counts))
        )
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.show()
    
    # Call the plot_class_distribution function to display the pie chart
    plot_class_distribution(class_counts)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance.keys(), feature_importance.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


# Load dataset
df = pd.read_csv('WSNBFSFdataset.csv')
df = df[['Time', 'S_Node', 'Node_id', 'Rest_Energy', 'Trace_Level', 'Mac_Type_Pckt', 'Source_IP_Port', 
         'Des_IP_Port', 'Packet_Size', 'TTL', 'Hop_Count', 'Broadcast_ID', 'Dest_Node_Num', 
         'Dest_Seq_Num', 'Src_Node_ID', 'Src_Seq_Num', 'Class']]
df['Class'] = df['Class'].astype(int)

# Plot class distribution
class_counts = df['Class'].value_counts()
print("Class Distribution:\n", class_counts)

# Build the model and plot results
model, importance, metrics = build_network_traffic_classifier(df)
print("\nClassification Report:")
print(metrics['classification_report'])

# Plot the results
plot_results(importance, metrics['confusion_matrix'], class_counts)
