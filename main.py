import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

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
    if not isinstance(hex_string, str):
        return []
    # Use 4-character tokens instead of 2-character
    tokens = [hex_string[i:i+4] for i in range(0, len(hex_string), 4)]
    # Make tokens unique based on column by prefixing with column identifier
    return [f"{column_name[:3]}_{token}" for token in tokens]

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

# Custom dataset class
class TLSDataset(Dataset):
    def __init__(self, dataframe, feature_cols, token_to_idx, max_length):
        self.df = dataframe
        self.feature_cols = feature_cols
        self.token_to_idx = token_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process each feature column
        features = []
        for col in self.feature_cols:
            indices = convert_to_indices(row[col], col, self.max_length)
            features.append(indices)
        
        # Stack features
        x = torch.tensor(features, dtype=torch.long)
        y = torch.tensor(row['OS_encoded'], dtype=torch.long)
        
        return x, y

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create datasets
train_dataset = TLSDataset(train_df, tls_feature_columns, token_to_idx, max_length)
test_dataset = TLSDataset(test_df, tls_feature_columns, token_to_idx, max_length)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model definition
class TLSFingerprint(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_features):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # LSTM for each feature
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * num_features, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, num_features, max_length]
        batch_size, num_features, _ = x.shape
        
        feature_outputs = []
        for i in range(num_features):
            # Get feature tokens
            feature_tokens = x[:, i, :]  # [batch_size, max_length]
            
            # Embed tokens
            embedded = self.embedding(feature_tokens)  # [batch_size, max_length, embed_dim]
            
            # Pass through LSTM
            lstm_out, (h_n, _) = self.lstm(embedded)
            
            # Use the last hidden state
            feature_outputs.append(h_n.squeeze(0))
            
        # Concatenate all feature representations
        combined = torch.cat(feature_outputs, dim=1)  # [batch_size, num_features * hidden_dim]
        
        # Pass through fully connected layers
        x = self.fc1(combined)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Hyperparameters
embed_dim = 64
hidden_dim = 128
learning_rate = 0.001
num_epochs = 30

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TLSFingerprint(
    vocab_size=vocab_size, 
    embed_dim=embed_dim, 
    hidden_dim=hidden_dim, 
    num_classes=num_classes,
    num_features=len(tls_feature_columns)
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print training stats
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Final evaluation
model.eval()
with torch.no_grad():
    all_predictions = []
    all_labels = []
    
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())
    
    # Convert numeric predictions back to labels
    predicted_os = label_encoder.inverse_transform(all_predictions)
    true_os = label_encoder.inverse_transform(all_labels)
    
    # Calculate accuracy
    accuracy = (predicted_os == true_os).mean()
    print(f'Final accuracy: {accuracy * 100:.2f}%')
    
    # Optional: Display confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    print(confusion_matrix(true_os, predicted_os))
    print(classification_report(true_os, predicted_os))