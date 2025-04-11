import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import glob
import time

# Merge all CSV files in the folder
folder_path = './'  # Current directory, change if needed
all_files = glob.glob(os.path.join(folder_path, '*.csv'))

# Create an empty list to store individual dataframes
dfs = []

# Read each CSV file and append to the list
for file in all_files:
    file_df = pd.read_csv(file)
    print(f"Loaded {file} with {len(file_df)} rows")
    dfs.append(file_df)

# Concatenate all dataframes
df = pd.concat(dfs, ignore_index=True)
print(f"Merged dataset contains {len(df)} rows")

# OPTIMIZATION 1: Sample the data to reduce training time
# Use a smaller subset for development/testing
#sample_size = min(1000000, len(df))  # Adjust based on your computational resources
df_sampled = df
print(f"Using {len(df_sampled)} samples for model training")

# Basic data exploration
print(df_sampled.head())
print(df_sampled['Ground Truth OS'].value_counts())

# Preprocessing
# Handle categorical features
categorical_features = ['TLS Client Version']
numerical_features = ['TLS SNI length', 'Session ID']

# OPTIMIZATION 2: More efficient feature extraction
def extract_hex_features(hex_string):
    if not isinstance(hex_string, str):
        return {}
    
    # Extract basic metrics instead of complex parsing
    features = {
        'length': len(hex_string),
        'unique_chars': len(set(hex_string)),
        'has_zeros': '00' in hex_string,
        # Add a few more simple features that don't require heavy computation
        'starts_with_c0': hex_string.startswith('c0') if len(hex_string) > 1 else False,
        'has_ff': 'ff' in hex_string
    }
    return features

# Apply to relevant columns
print("Extracting features from hex data...")
start_time = time.time()
df_sampled['cipher_features'] = df_sampled['Client Cipher Suites'].apply(extract_hex_features)
df_sampled['extension_features'] = df_sampled['TLS Extension Types'].apply(extract_hex_features)
print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds")

# Extract the features as separate dataframes to avoid slow apply(pd.Series)
cipher_df = pd.DataFrame(df_sampled['cipher_features'].tolist(), index=df_sampled.index)
extension_df = pd.DataFrame(df_sampled['extension_features'].tolist(), index=df_sampled.index)

# Prepare feature matrix more efficiently - ensure all parts have same index
print("Preparing feature matrix...")
X = pd.concat([
    pd.get_dummies(df_sampled[categorical_features], sparse=True),  # Use sparse matrices
    df_sampled[numerical_features],
    cipher_df,
    extension_df
], axis=1)

# Print shapes to debug
print(f"Shape of one-hot encoded categorical features: {pd.get_dummies(df_sampled[categorical_features]).shape}")
print(f"Shape of numerical features: {df_sampled[numerical_features].shape}")
print(f"Shape of cipher features: {cipher_df.shape}")
print(f"Shape of extension features: {extension_df.shape}")
print(f"Final X shape: {X.shape}")

y = df_sampled['Ground Truth OS']
print(f"y shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# OPTIMIZATION 4: Use a lighter model configuration
print("Training model...")
start_time = time.time()
model = RandomForestClassifier(
    n_estimators=50,       # Reduced from 100
    max_depth=10,          # Limit tree depth
    min_samples_split=10,  # Require more samples per split
    n_jobs=-1,             # Use all available cores
    random_state=42
)
model.fit(X_train, y_train)
print(f"Model training completed in {time.time() - start_time:.2f} seconds")

# Evaluate
print("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Identify most important features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("Top 10 most important features:")
print(feature_importance.head(10))