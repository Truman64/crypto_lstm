import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dataset import CryptoDataset
from model import CryptoLSTM
from data_loader import load_crypto_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 240

# ----------------- Load Data -----------------
df = load_crypto_data(symbol="BTCUSDT")
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "quote_asset_volume", "number_of_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol",
    "best_bid", "best_ask", "mid_price", "spread",
    "order_imbalance", "cum_bid_qty", "cum_ask_qty",
    "wavg_bid", "wavg_ask", "price_change_1m",
    "price_change_3m", "vwap", "taker_buy_ratio",
    "rolling_mean_5", "rolling_std_5", "rolling_volume_mean_5",
    "momentum_5", "rolling_mean_15", "rolling_std_15",
    "rolling_volume_mean_15", "rsi_14",
    "hour", "day_of_week", "trades_per_min"
]
TARGET_COL = "hit_05percent_within_15m"

# ----------------- Preprocessing -----------------
scaler = StandardScaler()
df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

train_dataset = CryptoDataset(train_df, FEATURE_COLUMNS, TARGET_COL, seq_len=SEQ_LEN)
test_dataset = CryptoDataset(test_df, FEATURE_COLUMNS, TARGET_COL, seq_len=SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------- Model -----------------
model = CryptoLSTM(input_size=len(FEATURE_COLUMNS)).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------- Training -----------------
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader):.4f}")
