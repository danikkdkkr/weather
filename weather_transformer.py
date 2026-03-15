from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# --- Using Meteostat ---
from meteostat import Point, Daily, Stations, Hourly

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=4, output_dim=11):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer(s)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=0.2, batch_first=True)

        # Fully-connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden + cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))   # out: (batch, seq_len, hidden_dim)

        # Take last time step
        out = out[:, -1, :]               # (batch, hidden_dim)

        # Fully-connected output
        out = self.fc(out)                # (batch, output_dim)
        return out
    
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=30, horizon=1, step=1):
        """
        data: pd.DataFrame or np.ndarray, with all features (including target columns)
        seq_len: number of past timesteps per input
        horizon: steps ahead to predict (usually 1 = next day)
        """
        self.X = data.values if hasattr(data, "values") else np.asarray(data)
        self.seq_len = seq_len
        self.horizon = horizon
        self.step = step
        self.n = len(self.X)
        self.max_start = self.n - seq_len - horizon + 1

    def __len__(self):
        return (self.max_start + self.step - 1) // self.step

    def __getitem__(self, idx):
        i = idx * self.step
        x = self.X[i : i + self.seq_len]                     # (seq_len, n_features)
        y = self.X[i + self.seq_len : i + self.seq_len + self.horizon, :11]  # first 10 cols = 

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y.squeeze(), dtype=torch.float32),
        )



def prep_last_days(data):
    pass

def normalize_data(data):
    pass

# Define the location (latitude, longitude)
location = Point(48.78, 9.18)  # Stuttgart approx

# Fetch daily data for a time window
start = datetime(1970, 1, 1)
end = datetime(2024, 12, 31)
data = Daily(location, start, end).fetch()
df = pd.DataFrame(data)

#consider seasonality
df.axes[0] = pd.to_datetime(df.axes[0])
day_of_year = df.axes[0].dayofyear
df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)
df['snow'] = df['snow'].fillna(0)
df = df.dropna()
df["wdir_rad"] = np.deg2rad(df["wdir"])
df["wdir_sin"] = np.sin(df["wdir_rad"])
df["wdir_cos"] = np.cos(df["wdir_rad"])
df = df.drop(columns=["wdir","wdir_rad"])

#normalize values
# Columns already in [-1, 1]
skip_cols = ["day_sin", "day_cos", "wdir_sin", "wdir_cos"]
# Select columns to scale
cols_to_scale = [c for c in df.columns if c not in skip_cols]
# Create a copy so we don’t modify the original DataFrame
df_scaled = df.copy()
# Fit-transform only the selected columns
scaler = MinMaxScaler(feature_range=(-1, 1))
df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
joblib.dump(scaler, "scaler.pkl")

#train test split
train_size = int(len(df) * 0.8)
train = df_scaled.iloc[:train_size]
test = df_scaled.iloc[train_size:]

target_cols = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun"]
seq_len = 30
horizon = 3
train_ds = SequenceDataset(train, seq_len=seq_len, horizon=horizon, step=1)
test_ds = SequenceDataset(test, seq_len=seq_len, horizon=horizon, step=1)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)


n_features = train.shape[1]
output_dim = 11

model = LSTMModel(input_dim=n_features, output_dim=output_dim*horizon)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- Training parameters ---
max_epochs = 150
patience = 15
best_val = float("inf")
trigger = 0

train_losses = []
val_losses = []

# --- Training loop with validation and early stopping ---
for epoch in range(max_epochs):
    model.train()
    train_loss_epoch = 0.0
    n_train_batches = 0

    for X, y in train_loader:
        optimizer.zero_grad()
        y_hat = model(X)
        y_hat = y_hat.view(y_hat.size(0), horizon, output_dim)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        n_train_batches += 1

    train_loss_epoch /= n_train_batches
    train_losses.append(train_loss_epoch)

    # --- Validation ---
    model.eval()
    val_loss_epoch = 0.0
    n_val_batches = 0
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = model(X)
            y_hat = y_hat.view(y_hat.size(0), horizon, output_dim)
            val_loss_epoch += criterion(y_hat, y).item()
            n_val_batches += 1

    val_loss_epoch /= n_val_batches
    val_losses.append(val_loss_epoch)

    print(f"Epoch {epoch:03d} | Train: {train_loss_epoch:.5f} | Val: {val_loss_epoch:.5f}")

    # --- Early stopping ---
    if val_loss_epoch < best_val:
        best_val = val_loss_epoch
        trigger = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        trigger += 1
        if trigger >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

# Determine which columns were scaled
scaled_cols = list(scaler.feature_names_in_)           # features the scaler knows
all_cols = list(df.columns)                            # full DataFrame columns (incl. sin/cos)
unscaled_cols = [c for c in all_cols if c not in scaled_cols]  # e.g. ['day_sin','day_cos','wdir_sin','wdir_cos']

# Convert to numpy arrays
y_hat_np = y_hat[-1].cpu().numpy()      # (horizon, output_dim)
y_true_np = y[-1].cpu().numpy()

# Build DataFrame for easy column alignment
target_cols = ["tavg", "tmin", "tmax", "prcp", "snow",
               "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun"]

# Create DataFrames with predicted & true values, in correct column order
pred_df = pd.DataFrame(y_hat_np, columns=target_cols)
true_df = pd.DataFrame(y_true_np, columns=target_cols)

# Separate scaled vs unscaled subsets
pred_scaled_part = pred_df[[c for c in target_cols if c in scaled_cols]].copy()
true_scaled_part = true_df[[c for c in target_cols if c in scaled_cols]].copy()

# Inverse-transform only the scaled subset
inv_pred_scaled = pd.DataFrame(
    scaler.inverse_transform(pred_scaled_part),
    columns=pred_scaled_part.columns
)
inv_true_scaled = pd.DataFrame(
    scaler.inverse_transform(true_scaled_part),
    columns=true_scaled_part.columns
)

# Merge back with the unscaled columns (keep them as-is)
inv_pred = pred_df.copy()
inv_true = true_df.copy()

for c in inv_pred_scaled.columns:
    inv_pred[c] = inv_pred_scaled[c]
    inv_true[c] = inv_true_scaled[c]



print("\n--- Last Prediction vs True Values ---")
for i, name in enumerate(target_cols):
    for h in range(horizon):
        pred_val = inv_pred.iloc[h, i]  
        true_val = inv_true.iloc[h, i]   
        print(f"Day +{h+1:>2}: {name:<6s} | Pred: {pred_val:8.3f} | True: {true_val:8.3f} | Diff: {pred_val-true_val:8.3f}")


# --- Plot losses ---
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("L1 Loss (MAE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()