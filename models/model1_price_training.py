import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime, timedelta

# Circom prime field modulus
PRIME = 21888242871839275222246405745257275088548364400416034343698204186575808495617

def mod_field(x):
    """Ensure number is within the prime field"""
    return str(int(x) % PRIME)  # Convert to string to match original format

def scale_for_circom(value, scale_factor=10**36):
    """Scale floating point values to integers for circom"""
    scaled = int(value * scale_factor)
    return mod_field(scaled)

def calculate_dot_product(inputs, weights):
    """Calculate dot product manually to match circom's computation"""
    result = 0
    for i in range(len(inputs)):
        result += int(inputs[i]) * int(weights[i])
    return result

def calculate_dense_outputs(inputs, weights, scale_factor):
    """Calculate dense layer outputs and remainders that satisfy the constraint:
    out[i] * n + remainder[i] === dot[i].out[0][0] + bias[i]
    """
    n_outputs = len(weights[0])
    outputs = []
    remainders = []
    
    for i in range(n_outputs):
        output_weights = [row[i] for row in weights]
        dot_product = calculate_dot_product(inputs, output_weights)
        output = dot_product // scale_factor
        remainder = dot_product % scale_factor
        outputs.append(str(output))
        remainders.append(str(remainder))
    
    return outputs, remainders

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weights with small positive values
        nn.init.uniform_(self.weight, 0.001, 0.1)
    
    def forward(self, x):
        # Ensure weights stay positive using ReLU
        positive_weights = torch.relu(self.weight)
        return torch.matmul(x, positive_weights.t())

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Initialize with non-zero weights
        self.dense1 = nn.Linear(3, 2)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.constant_(self.dense1.bias, 0.1)
        
        self.relu = nn.ReLU()
        
        self.dense2 = nn.Linear(2, 1)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.constant_(self.dense2.bias, 0.1)
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

def fetch_sol_data(start_date="2023-01-01"):
    """Fetch SOL price data and prepare features"""
    # Fetch data
    sol = yf.download("SOL-USD", start=start_date)
    
    # Create a copy to avoid SettingWithCopyWarning
    df = sol.copy()
    
    # Calculate technical indicators
    df['SMA7'] = df['Close'].rolling(window=7).mean()
    df['SMA21'] = df['Close'].rolling(window=21).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Calculate price momentum
    df['Price_Momentum'] = df['Close'].values.flatten() / df['SMA21'].values.flatten()
    df['Price_Momentum'] = df['Price_Momentum'].fillna(1.0)
    
    # Normalize volume by its moving average
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Ratio'] = df['Volume'].values.flatten() / df['Volume_MA5'].values.flatten()
    df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    # Select features that are most relevant for SOL price prediction
    features = df[[
        'Close',  # Current price
        'Volume_Ratio',  # Volume trend
        'Price_Momentum'  # Price momentum
    ]].values
    
    # Target is next day's closing price
    targets = df['Close'].shift(-1).values[:-1]
    
    print("\nFeature Statistics:")
    print(f"Latest Close Price: ${features[-1][0]:.2f}")
    print(f"Latest Volume Ratio: {features[-1][1]:.2f}")
    print(f"Latest Price Momentum: {features[-1][2]:.2f}")
    
    return features[:-1], targets, df

def prepare_circom_input(model, input_data, price_scaler):
    """Prepare model weights and inputs in circom compatible format"""
    print("\n=== Input Data ===")
    print(f"Raw input: {input_data[0]}")
    
    # Scale input data
    scaled_input = [scale_for_circom(x) for x in input_data[0]]
    print(f"Scaled input: {scaled_input}")
    
    # Get weights from first dense layer and scale them
    dense1_weights = torch.relu(model.dense1.weight).data.numpy()
    print("\n=== Dense1 Weights ===")
    print(f"Raw weights:\n{dense1_weights}")
    
    dense1_weights_scaled = [[mod_field(scale_for_circom(w)) for w in row] for row in dense1_weights.T]
    print(f"Scaled weights:\n{dense1_weights_scaled}")
    
    # Calculate Dense1 outputs and remainders
    dense1_out_scaled, dense1_remainder = calculate_dense_outputs(
        scaled_input, dense1_weights_scaled, 10**36
    )
    print("\n=== Dense1 Output ===")
    print(f"Scaled output: {dense1_out_scaled}")
    print(f"Remainders: {dense1_remainder}")
    
    # ReLU activation
    relu_out_scaled = [str(max(0, int(x))) for x in dense1_out_scaled]
    print("\n=== ReLU Output ===")
    print(f"Scaled output: {relu_out_scaled}")
    
    # Get weights from second dense layer
    dense2_weights = torch.relu(model.dense2.weight).data.numpy()
    print("\n=== Dense2 Weights ===")
    print(f"Raw weights:\n{dense2_weights}")
    
    dense2_weights_scaled = [[mod_field(scale_for_circom(w))] for w in dense2_weights[0]]
    print(f"Scaled weights:\n{dense2_weights_scaled}")
    
    # Calculate Dense2 outputs and remainders
    dense2_out_scaled, dense2_remainder = calculate_dense_outputs(
        relu_out_scaled, dense2_weights_scaled, 10**36
    )
    print("\n=== Final Output ===")
    print(f"Scaled output: {dense2_out_scaled}")
    print(f"Remainders: {dense2_remainder}")
    
    # Print predicted price
    predicted_scaled = float(dense2_out_scaled[0]) / (10**36)
    predicted_price = price_scaler.inverse_transform([[predicted_scaled]])[0][0]
    print(f"\nPredicted SOL Price: ${predicted_price:.2f}")
    
    circom_input = {
        "in": scaled_input,
        "Dense32weights": dense1_weights_scaled,
        "Dense32bias": ["0", "0"],
        "Dense32out": dense1_out_scaled,
        "Dense32remainder": dense1_remainder,
        "ReLUout": relu_out_scaled,
        "Dense21weights": dense2_weights_scaled,
        "Dense21bias": ["0"],
        "Dense21out": dense2_out_scaled,
        "Dense21remainder": dense2_remainder
    }
    
    print("\n=== Final Circom Input ===")
    print(json.dumps(circom_input, indent=2))
    
    return circom_input

def train_model(X, y, learning_rate=0.001, epochs=1000):
    model = Model1()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    # Calculate initial prediction for monitoring
    with torch.no_grad():
        initial_pred = model(X_tensor[-1:])
        print(f"Initial prediction before training: {initial_pred.item():.4f}")
    
    best_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_model = None
    
    # Calculate mean absolute percentage error
    def calc_mape(pred, actual):
        return torch.mean(torch.abs((actual - pred) / actual)) * 100
    
    print("\nTraining Progress:")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Force weights to be positive after each update
        with torch.no_grad():
            model.dense1.weight.data = torch.relu(model.dense1.weight.data)
            model.dense2.weight.data = torch.relu(model.dense2.weight.data)
        
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                mape = calc_mape(outputs, y_tensor)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}, MAPE: {mape:.2f}%')
                
                # Monitor weights to ensure they're not zero
                print(f'Dense1 weight mean: {model.dense1.weight.data.mean():.6f}')
                print(f'Dense2 weight mean: {model.dense2.weight.data.mean():.6f}')
                
                # Validate prediction on latest data
                latest_pred = outputs[-1].item()
                actual = y_tensor[-1].item()
                print(f'Latest scaled prediction: {latest_pred:.4f}, Actual: {actual:.4f}')
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model = Model1()
            best_model.load_state_dict(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                print(f"Best loss: {best_loss:.8f}")
                break
    
    return best_model if best_model is not None else model

if __name__ == "__main__":
    # Fetch more recent data for SOL
    features, targets, sol_df = fetch_sol_data(start_date="2023-06-01")
    
    # Scale features to [0, 1] range
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Scale features and targets
    features_scaled = feature_scaler.fit_transform(features)
    targets_scaled = price_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
    
    # Print scaling info
    print("\nScaling Information:")
    print(f"Feature ranges: {feature_scaler.data_min_} to {feature_scaler.data_max_}")
    print(f"Price range: ${price_scaler.data_min_[0]:.2f} to ${price_scaler.data_max_[0]:.2f}")
    
    # Train model
    train_size = int(len(features_scaled) * 0.8)
    model = train_model(features_scaled[:train_size], 
                       targets_scaled[:train_size],
                       learning_rate=0.001,
                       epochs=1000)
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/sol_price_model.pth')
    print("\nModel saved to models/sol_price_model.pth")
    
    # Generate prediction for the latest data point
    latest_data = feature_scaler.transform(features[-1:])
    
    # Test prediction before circom
    with torch.no_grad():
        test_input = torch.FloatTensor(latest_data)
        test_output = model(test_input)
        test_price = price_scaler.inverse_transform([[test_output.item()]])[0][0]
        print(f"\nDirect PyTorch Prediction: ${test_price:.2f}")
    
    # Scale the weights and prepare circom input
    def scale_for_circom(x):
        return str(int(x * (10**36)))
    
    circom_input = prepare_circom_input(model, latest_data, price_scaler)
    
    # Save circom input to json
    with open('models/model1_price_input.json', 'w') as f:
        json.dump(circom_input, f, indent=2) 