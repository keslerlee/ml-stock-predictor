import requests
import pandas as pd
import numpy as np
import os 
import json
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---

BASE_URL = "https://api.twelvedata.com/time_series"

TRAIN_TICKERS = ['AAPL', 'MSFT', 'GOOGL']
TEST_TICKERS = ['AMZN', 'TSLA', 'NVDA']

CACHE_DIR = 'stock_cache'
CACHE_DURATION_SECONDS = 86400

FEATURES = [
    'ma_short', 
    'ma_long', 
    'rsi', 
    'roc', 
    'volatility',
    'pct_change', 
    'daily_range',
    'candle_direction',
    'high_to_close',
    'low_to_close',
    'pct_change_lag1',
    'pct_change_lag2',
    'pct_change_lag5',
    'pct_change_lag10',
    'rsi_lag1',
    'rsi_lag2',
    'rsi_lag5',
    'rsi_lag10',
    'candle_direction_lag1'
]

# --- DATA FETCHING ---

def get_stock_data(symbol, api_key, interval='1day', output_size=5000):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{interval}.json")
    
    if os.path.exists(cache_file):
        file_mod_time = os.path.getmtime(cache_file)
        time_diff = time.time() - file_mod_time
        
        if time_diff < CACHE_DURATION_SECONDS:
            print(f"Loading data for {symbol} from local cache...")
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.sort_index(inplace=True)
                return df
            except Exception as e:
                print(f"Error loading cache file: {e}. Fetching new data.")
    
    print(f"Fetching fresh data for {symbol}...")
    params = {
        'symbol': symbol,
        'interval': interval,
        'outputsize': output_size,
        'apikey': api_key
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'values' in data:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.sort_index(inplace=True)
            return df
        else:
            print(f"Error fetching data for {symbol}: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"API Error for {symbol}: {e}")
        return pd.DataFrame()

# --- FEATURE ENGINEERING ---

def create_features(df, rsi_period=14, roc_period=20, ma_short=10, ma_long=50):
    df['pct_change'] = df['close'].pct_change()
    df['ma_short'] = df['close'].rolling(window=ma_short).mean()
    df['ma_long'] = df['close'].rolling(window=ma_long).mean()
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    df['roc'] = df['close'].pct_change(periods=roc_period)
    
    df['daily_range'] = (df['high'] - df['low']) / df['open']
    df['candle_direction'] = (df['close'] > df['open']).astype(int)
    df['high_to_close'] = (df['high'] - df['close']) / df['high']
    df['low_to_close'] = (df['close'] - df['low']) / df['low']
    df.replace([np.inf, -np.inf], 0, inplace=True)

    df['pct_change_lag1'] = df['pct_change'].shift(1)
    df['pct_change_lag2'] = df['pct_change'].shift(2)
    df['pct_change_lag5'] = df['pct_change'].shift(5)
    df['pct_change_lag10'] = df['pct_change'].shift(10)
    df['rsi_lag1'] = df['rsi'].shift(1)
    df['rsi_lag2'] = df['rsi'].shift(2)
    df['rsi_lag5'] = df['rsi'].shift(5)
    df['rsi_lag10'] = df['rsi'].shift(10)
    df['candle_direction_lag1'] = df['candle_direction'].shift(1)

    # --- TARGET VARIABLE ---
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    df.dropna(inplace=True)
    return df

# --- MODEL TRAINING ---

def train_model(api_key):
    print("--- Starting Model Training ---")
    all_train_data = []
    
    for symbol in TRAIN_TICKERS:
        df = get_stock_data(symbol, api_key)
        if not df.empty:
            df_features = create_features(df)
            all_train_data.append(df_features)

    if not all_train_data:
        print("No valid training data found.")
        return None

    train_df = pd.concat(all_train_data)

    X_train = train_df[FEATURES].values
    y_train = train_df['target'].values

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

# --- MODEL TESTING ---

def test_model(model, api_key):
    if model is None:
        print("No model to test.")
        return

    print("\n--- Starting Model Testing ---")
    
    for symbol in TEST_TICKERS:
        df = get_stock_data(symbol, api_key)
        if df.empty:
            continue

        df_features = create_features(df)

        X_test = df_features[FEATURES].values
        y_test = df_features['target'].values
        
        if len(y_test) == 0:
            print(f"Skipping {symbol}: Not enough data after feature creation.")
            continue

        y_pred = model.predict(X_test)
        
        print(f"\nResults for {symbol}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))

# --- MAIN EXECUTION ---

def load_api_key_from_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('API_KEY='):
                    return line.strip().split('=')[1].strip('\"')
    except FileNotFoundError:
        return None

def main():
    api_key = load_api_key_from_env()
    
    if not api_key:
        print("API Key not found.")
        return

    model = train_model(api_key)
    test_model(model, api_key)

if __name__ == '__main__':
    main()