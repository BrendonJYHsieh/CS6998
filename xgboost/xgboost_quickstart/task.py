"""Encrypted Echoes: Task module for federated TLS fingerprinting."""

from logging import INFO
import os
import joblib
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from flwr.common import log
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split as sklearn_train_test_split

def parse_hex_list(raw_str):
    """Parse hex string into list of 4-character hex chunks"""
    if pd.isna(raw_str):
        return []

    # keep hex chars only, split every 4-hex
    clean = "".join(x for x in raw_str if x in "0123456789abcdefABCDEF")
    return [clean[i : i + 4].lower() for i in range(0, len(clean), 4)]

def to_slots(series, k, prefix):
    """Convert series of lists to fixed-length columns with missing values padded"""
    return pd.DataFrame(
        series.apply(lambda lst: (lst + ["MISSING"] * k)[:k]).to_list(),
        columns=[f"{prefix}_pos{i}" for i in range(k)],
    )

def preprocess_single_file(file_path, keep_cols=None):
    """Preprocess a single CSV file and return preprocessed data"""
    if keep_cols is None:
        keep_cols = [
            "TLS Client Version",
            "Client Cipher Suites",
            "TLS Extension Types",
            "TLS Extension Lengths",
            "TLS Elliptic Curves",
            "Ground Truth OS",
        ]
    
    df = pd.read_csv(file_path, usecols=keep_cols, low_memory=False)
    df = df.dropna().reset_index(drop=True)
    
    df["cipher_list"] = df["Client Cipher Suites"].apply(parse_hex_list)
    df["group_list"] = df["TLS Elliptic Curves"].apply(parse_hex_list)
    df["ext_id_list"] = df["TLS Extension Types"].apply(parse_hex_list)
    df["ext_len_list"] = df["TLS Extension Lengths"].apply(parse_hex_list)
    
    # Define constants for feature extraction
    K_CIPHER = 8    # first 8 cipher IDs
    K_GROUP = 8     # first 8 supported-group IDs
    K_EXT = 100     # first 100 extension IDs
    K_EXLEN = 100   # first 100 extension lengths
    
    X_raw = pd.concat(
        [
            to_slots(df["cipher_list"], K_CIPHER, "cipher"),
            to_slots(df["group_list"], K_GROUP, "group"),
            to_slots(df["ext_id_list"], K_EXT, "extid"),
            to_slots(df["ext_len_list"], K_EXLEN, "extlen"),
            df[["TLS Client Version"]],
        ],
        axis=1,
    )
    
    y = df["Ground Truth OS"]
    
    return X_raw, y

def load_data(partition_id, num_clients):
    """Load partition TLS data for federated learning."""
    # Get list of CSV files
    files = sorted(glob.glob("flows_anonymized/*_ground_truth_tls_only.csv"))
    
    if not files:
        # Try with a different path if files aren't found
        files = sorted(glob.glob("flows_anonymized/*_ground_truth_tls_only.csv"))
        
    if not files:
        raise FileNotFoundError("Could not find TLS data files in flows_anonymized directory")
        
    log(INFO, f"Found {len(files)} data files")
    
    # Create data partitions
    partitioned_files = np.array_split(files, num_clients)
    client_files = partitioned_files[partition_id]
    
    log(INFO, f"Client {partition_id} assigned {len(client_files)} files")
    
    all_X_raw = []
    all_y = []
    
    # Load and preprocess client's files
    for file_path in client_files:
        X_raw, y = preprocess_single_file(file_path)
        all_X_raw.append(X_raw)
        all_y.append(y)
    
    # Combine data from all client files
    X_raw_combined = pd.concat(all_X_raw, ignore_index=True) if all_X_raw else pd.DataFrame()
    y_combined = pd.concat(all_y, ignore_index=True) if all_y else pd.Series()
    
    if len(X_raw_combined) == 0:
        raise ValueError(f"No data was loaded for client {partition_id}")
    
    log(INFO, f"Client {partition_id} loaded {len(X_raw_combined)} examples")
    
    # Check if encoder exists or create a new one
    encoder_path = "preprocessed/tls_onehot_encoder.joblib"
    if os.path.exists(encoder_path):
        pre = joblib.load(encoder_path)
    else:
        onehot = OneHotEncoder(handle_unknown="ignore")
        pre = ColumnTransformer([("oh", onehot, X_raw_combined.columns)], sparse_threshold=0.3)
        pre.fit(X_raw_combined)
        os.makedirs("preprocessed", exist_ok=True)
        joblib.dump(pre, encoder_path)
    
    # Transform data
    X_encoded = pre.transform(X_raw_combined)
    
    # Get label encodings
    y_int, os_labels = pd.factorize(y_combined)
    
    # Store os_labels for future use if not already saved
    os_labels_path = "preprocessed/os_labels.joblib"
    if not os.path.exists(os_labels_path):
        os.makedirs("preprocessed", exist_ok=True)
        joblib.dump(os_labels, os_labels_path)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = sklearn_train_test_split(
        X_encoded, y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    
    # Create DMatrix objects
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    valid_dmatrix = xgb.DMatrix(X_val, label=y_val)
    
    num_train = X_train.shape[0]
    num_val = X_val.shape[0]
    
    log(INFO, f"Client {partition_id} prepared {num_train} training and {num_val} validation examples")
    
    return train_dmatrix, valid_dmatrix, num_train, num_val

def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict