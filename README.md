# Crypto Price Movement Prediction with LSTM

This project demonstrates a simple **LSTM-based deep learning model** for predicting short-term cryptocurrency price movements. The model uses historical crypto data and technical features to classify whether a coin’s price will move by at least 0.05% within 15 minutes.  

The project is structured for clarity and maintainability, with proper separation of data loading, preprocessing, model definition, training, and evaluation. It is suitable for **recruiters or hiring managers** to quickly understand your skills in **Python, PyTorch, time-series modeling, and ML project organization**.

---

## 🚀 Features

- Load and preprocess historical cryptocurrency data from PostgreSQL.
- Feature engineering and scaling for LSTM input.
- Sequence-based modeling using **LSTM** for time-series classification.
- End-to-end pipeline: **training**, **testing**, and **evaluation**.
- Binary classification: predicting if the price hits a 0.05% change within 15 minutes.
- Evaluation includes:
  - Confusion Matrix
  - Precision, Recall, F1-score
  - Classification report

---

## 📂 Project Structure

crypto_lstm_project/
├── data/ # Store raw or processed CSV datasets
├── notebooks/ # Optional Jupyter notebooks for exploration
├── src/
│ ├── data_loader.py # Load crypto data from PostgreSQL
│ ├── dataset.py # PyTorch Dataset class for sequence preparation
│ ├── model.py # LSTM model definition
│ ├── train.py # Training loop with optimizer and loss
│ ├── evaluate.py # Evaluate model on test set
│ └── utils.py # Helper functions (e.g., scalers, sequence creation)
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/crypto-lstm-project.git
cd crypto-lstm-project

2. Install dependencies
pip install -r requirements.txt

3. Set up PostgreSQL database

Ensure you have a PostgreSQL database with historical crypto data. Update the connection string in src/data_loader.py:
"postgresql+psycopg2://postgres:yourpassword@localhost:5432/crypto_data"

4. Run the project

Training:
python src/train.py

Evaluation:
python src/evaluate.py

## Usage

Load Data
data_loader.py retrieves historical OHLCV and derived features for your crypto symbol. Example:
df = load_crypto_data(symbol="BTCUSDT", start="2025-09-08", end="2025-11-13")
Prepare Dataset
dataset.py converts raw data into sequences for LSTM training.

Train LSTM Model
train.py trains the LSTM for binary classification (hit_05percent_within_15m) with configurable hyperparameters.

Evaluate Model
evaluate.py prints confusion matrix, precision, recall, F1-score, and classification report.

Technical Details

Framework: PyTorch

Model: LSTM, 2 layers, hidden size 64

Loss Function: Binary Cross-Entropy

Optimizer: Adam

Device Support: CPU & GPU (auto-detected)

Input Features: OHLCV, orderbook, technical indicators, volume metrics, RSI, momentum

Sequence Length: 240 (configurable)

Target: Binary label (hit_05percent_within_15m)

Evaluation Metrics

The evaluation script outputs:

Confusion Matrix

Precision, Recall, F1-score

Classification report for positive/negative movements

Helps gauge model performance on short-term price predictions

Notes & Best Practices

Sequences are time-ordered, no shuffling in test set to avoid data leakage.

Features are standardized using StandardScaler.

Training and evaluation are separated for clarity and reproducibility.

LSTM is simple yet effective for demonstration and portfolio purposes.

Project is structured for recruiters to quickly read, understand, and run.