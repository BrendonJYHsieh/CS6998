import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('deduplicated_part_0.csv')

# List of feature columns we'll use for tokenization
tls_feature_columns = [
    'Client Cipher Suites', 
    'TLS Extension Types', 
    'TLS Extension Lengths', 
    'TLS Elliptic Curves'
]

# Tokenize hex strings into 4-char tokens, prefixed with column name
def tokenize_hex(hex_string, column_name):
    if column_name == 'Client Cipher Suites':
        column_name = 'CCS'
    if column_name == 'TLS Extension Types':
        column_name = 'TET'
    if column_name == 'TLS Extension Lengths':
        column_name = 'TEL'
    if column_name == 'TLS Elliptic Curves':
        column_name = 'TEC'
        
    if not isinstance(hex_string, str):
        return []
    # Use 4-character tokens instead of 2-character
    tokens = [hex_string[i:i+4] for i in range(0, len(hex_string), 4)]
    # Make tokens unique based on column by prefixing with column identifier
    return [f"{column_name}_{token}" for token in tokens]

# Build vocabulary from all features
all_tokens = set()
for column in tls_feature_columns:
    for hex_string in df[column]:
        all_tokens.update(tokenize_hex(hex_string, column))

token_to_idx = {token: idx+1 for idx, token in enumerate(all_tokens)}  # 0 reserved for padding
vocab_size = len(token_to_idx) + 1
print(f"Vocabulary size: {vocab_size}")

# Convert target variable to numeric
label_encoder = LabelEncoder()
df['OS_encoded'] = label_encoder.fit_transform(df['Ground Truth OS'])
num_classes = len(label_encoder.classes_)
print(f"OS classes: {label_encoder.classes_}")

# Maximum sequence length for each feature
max_length = 50  # Adjusted since we now have 4-char tokens instead of 2-char

# Function to convert a hex string to token indices
def convert_to_indices(hex_string, column_name, max_len):
    tokens = tokenize_hex(hex_string, column_name)
    indices = [token_to_idx.get(token, 0) for token in tokens]
    # Pad or truncate to max_length
    if len(indices) > max_len:
        return indices[:max_len]
    else:
        return indices + [0] * (max_len - len(indices))

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Feature engineering for traditional ML models
def prepare_ml_features(df, feature_cols, vectorizers=None, train=False):
    # Join all hex strings for each feature with spaces
    features = {}
    for col in feature_cols:
        features[col] = df[col].fillna('').astype(str)
    
    # Create or use vectorizers for each feature column
    if vectorizers is None:
        vectorizers = {}
    
    transformed_features = []
    
    for col in feature_cols:
        if train:
            # Only fit vectorizers on training data
            vectorizers[col] = CountVectorizer(analyzer=lambda x: tokenize_hex(x, col), 
                                              binary=True, max_features=1000)
            feature_matrix = vectorizers[col].fit_transform(features[col])
        else:
            # Use pre-fitted vectorizers for test data
            feature_matrix = vectorizers[col].transform(features[col])
            
        transformed_features.append(feature_matrix)
    
    # Combine all features horizontally
    X = np.hstack([f.toarray() for f in transformed_features])
    y = df['OS_encoded'].values
    
    return X, y, vectorizers

# Prepare features for ML models - fit on training data
X_train, y_train, vectorizers = prepare_ml_features(train_df, tls_feature_columns, train=True)
print(f"Training feature matrix shape: {X_train.shape}")

# Train Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree model trained.")

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained.")

# Evaluate models on test set - use pre-fitted vectorizers
X_test, y_test, _ = prepare_ml_features(test_df, tls_feature_columns, vectorizers=vectorizers, train=False)
print(f"Test feature matrix shape: {X_test.shape}")

# Evaluate Decision Tree
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")
print("Decision Tree Classification Report:")
print(classification_report(y_test, dt_predictions, target_names=label_encoder.classes_))

# Evaluate Random Forest
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions, target_names=label_encoder.classes_))

# Compare model performances
print("\nModel Comparison:")
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Feature importance analysis
print("\nTop 10 most important features for Random Forest:")
feature_names = []
for col in tls_feature_columns:
    names = [f"{col}_{f}" for f in vectorizers[col].get_feature_names_out()]
    feature_names.extend(names)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:10]  # Top 10 features
for i in indices:
    if i < len(feature_names):
        print(f"{feature_names[i]}: {importances[i]:.4f}")
