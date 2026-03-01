from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

matplotlib.use("TkAgg")  # add at top
# --- Using Meteostat ---
from meteostat import Point, Daily, Stations, Hourly

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout,
                 output_dim, horizon):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

        # map last hidden state -> (horizon * num_targets)
        self.fc = nn.Linear(hidden_dim, horizon * output_dim)

    def forward(self, x):
        B = x.size(0)
        h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))     # (B, seq_len, hidden_dim)
        out = out[:, -1, :]                 # (B, hidden_dim) last timestep
        out = self.fc(out)                  # (B, horizon * num_targets)
        out = out.view(B, self.horizon, self.output_dim)  # (B, H, D)
        return out
    
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=30, horizon=3, step=1):
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
        y = self.X[i + self.seq_len : i + self.seq_len + self.horizon, :11]  # first 11 cols = 

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y.squeeze(), dtype=torch.float32),
        )
    
class Params():
    def __init__(self, seq_len, horizon, hidden_dim, num_layers, lr, dropout):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = lr
        self.dropout = dropout



def get_stationdata(pos, start, end):
    location = Point(*pos)
    data = Daily(location, start, end).fetch()
    df = pd.DataFrame(data)
    return df

def handle_data(df):
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
    return df

def normalize(df):
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
    return df_scaled, scaler

def train_test_split(split, df_scaled):
    train_size = int(len(df_scaled) * split)
    train = df_scaled.iloc[:train_size]
    test = df_scaled.iloc[train_size:]
    return train, test

def split_and_load(split, df_scaled,params):
    train, test = train_test_split(split, df_scaled)
    train_ds = SequenceDataset(train, seq_len=params.seq_len, horizon=params.horizon, step=1)
    test_ds = SequenceDataset(test, seq_len=params.seq_len, horizon=params.horizon, step=1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False, drop_last=False)
    return train_loader, test_loader

def train(max_epochs, patience, train_loader, test_loader, model, criterion, optimizer, params):
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
                val_loss_epoch += criterion(y_hat, y).item()
                n_val_batches += 1

        val_loss_epoch /= n_val_batches
        val_losses.append(val_loss_epoch)

        #print(f"Epoch {epoch:03d} | Train: {train_loss_epoch:.5f} | Val: {val_loss_epoch:.5f}")

        # --- Early stopping ---
        if val_loss_epoch < best_val:
            best_val = val_loss_epoch
            trigger = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            trigger += 1
            if trigger >= patience:
                #print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    # --- Plot losses ---
    """ plt.ion()
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss (MAE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01) """
    return min(val_losses)

def predict_and_compare(best_model, scaler, target_cols, params, test_loader):
    # Determine which columns were scaled
    scaled_cols = list(scaler.feature_names_in_)           # features the scaler knows
    with torch.no_grad():
        for X, y in test_loader:
            y_hat = best_model(X)
    # Convert to numpy arrays
    y_hat_np = y_hat[-1].cpu().numpy()      # (horizon, output_dim)
    y_true_np = y[-1].cpu().numpy()

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


    DRIZZLE_THRESHOLD_MM = 0.3
    # Clip negatives just in case and zero-out drizzle
    inv_pred["prcp"] = np.clip(inv_pred["prcp"], 0, None)
    inv_pred["snow"] = np.clip(inv_pred["snow"], 0, None)

    inv_pred.loc[inv_pred["prcp"] < DRIZZLE_THRESHOLD_MM, "prcp"] = 0.0
    inv_pred.loc[inv_pred["snow"] < DRIZZLE_THRESHOLD_MM, "snow"] = 0.0


    print("\n--- Last Prediction vs True Values ---")
    for i, name in enumerate(target_cols):
        for h in range(params.horizon):
            pred_val = inv_pred.iloc[h, i]  
            true_val = inv_true.iloc[h, i]   
            #print(f"Day +{h+1:>2}: {name:<6s} | Pred: {pred_val:8.3f} | True: {true_val:8.3f} | Diff: {pred_val-true_val:8.3f}")

def perms(hidden_dims, num_layerss, lrs, dropouts):
    res = []
    for i in range(len(hidden_dims)):
        for j in range(len(num_layerss)):
            for k in range(len(lrs)):
                for l in range(len(dropouts)):
                    res.append([hidden_dims[i], num_layerss[j], lrs[k], dropouts[l]])
    return res

def main(params_list):
    params = Params(*params_list)

    #train test split
    split = 0.8
    train_loader, test_loader = split_and_load(split, df_scaled, params)

    #create model
    target_cols = ["tavg", "tmin", "tmax", "prcp", "snow", "wdir_sin", "wdir_cos", "wspd", "wpgt", "pres", "tsun"]
    n_features = df_scaled.shape[1]
    output_dim = 11
    model = LSTMModel(input_dim=n_features, hidden_dim=params.hidden_dim, num_layers=params.num_layers, dropout=params.dropout, output_dim=output_dim, horizon=params.horizon)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)


    # --- Training parameters ---
    max_epochs = 1000
    patience = 30
    loss = train(max_epochs, patience, train_loader, test_loader, model, criterion, optimizer, params)
    best_model = LSTMModel(
        input_dim=n_features,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_layers,
        dropout=params.dropout,
        output_dim=len(target_cols),
        horizon=params.horizon
    )
    best_model.load_state_dict(torch.load("best_model.pt"))
    best_model.eval()
    predict_and_compare(best_model, scaler, target_cols, params, test_loader)
    return loss, best_model

pos = (48.78, 9.18)
start = datetime(1970, 1, 1)
end = datetime(2024, 12, 31)
df = get_stationdata(pos, start, end)
df = handle_data(df)
df_scaled, scaler = normalize(df)

#load hyperparameters
hidden_dims = [20, 32, 48, 64, 84, 128]
num_layerss  = [1, 2, 3, 4 ,5]
lrs = [5e-4, 1e-3, 2e-3]
dropouts = [0.0, 0.1, 0.2, 0.3]
param_perms = perms(hidden_dims, num_layerss, lrs, dropouts)
seq_len = 30
horizon = 3
params_lists = [[seq_len, horizon] + param_perms[i] for i in range(len(param_perms))]
tab = []
for elem in params_lists:
    loss, model = main(elem)
    tab.append([loss, model, elem])
    print(tab[-1])

print("best params are:")
print(min(tab))
