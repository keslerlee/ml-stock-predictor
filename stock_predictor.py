import requests
import pandas as pd
import numpy as np
import os 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- CONFIGURATION ---

BASE_URL = "https://api.twelvedata.com/time_series"

TRAIN_TICKERS = ['AAPL', 'MSFT', 'GOOGL']
TEST_TICKERS = ['AMZN', 'TSLA', 'NVDA']

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
    'rsi_lag1',
    'candle_direction_lag1'
]

# --- DATA FETCHING ---

def get_stock_data(symbol, api_key, interval='1day', output_size=5000):
    print(f"Fetching data for {symbol}...")
    params = {
        'symbol': symbol,
        'interval': interval,
        'apikey': api_key,
        'outputsize': output_size,
        'format': 'JSON'
    }
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'error':
            print(f"Error fetching data for {symbol}: {data.get('message', 'Unknown error')}")
            return None
            
        df = pd.DataFrame(data['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        
        df = df.apply(pd.to_numeric, errors='coerce')
        
        df.sort_index(ascending=True, inplace=True)
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed for {symbol}: {e}")
        return None
    except KeyError:
        print(f"Unexpected JSON structure for {symbol}. Skipping.")
        return None
    except Exception as e:
        print(f"An error occurred for {symbol}: {e}")
        return None

# --- FEATURE ENGINEERING ---

def create_features(df, rsi_period=14, roc_period=20, ma_short=10, ma_long=50):
    # 1. Close-based features
    df['pct_change'] = df['close'].pct_change()

    df['ma_short'] = df['close'].rolling(window=ma_short).mean()
    df['ma_long'] = df['close'].rolling(window=ma_long).mean()

    df['roc'] = (df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)
    
    df['volatility'] = df['pct_change'].rolling(window=30).std()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 2. Open/High/Low-based features
    df['daily_range'] = df['high'] - df['low']
    
    df['candle_direction'] = (df['close'] > df['open']).astype(int)
    
    safe_daily_range = df['daily_range'].apply(lambda x: max(x, 1e-9)) 
    
    df['high_to_close'] = (df['high'] - df['close']) / safe_daily_range
    
    df['low_to_close'] = (df['close'] - df['low']) / safe_daily_range
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    df['pct_change_lag1'] = df['pct_change'].shift(1)
    df['pct_change_lag2'] = df['pct_change'].shift(2)
    
    df['rsi_lag1'] = df['rsi'].shift(1)
    
    df['candle_direction_lag1'] = df['candle_direction'].shift(1)
    
    # --- TARGET VARIABLE ---
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    return df

# --- MODEL TRAINING ---

def train_model(api_key):
    print("--- Starting Model Training ---")
    all_train_data = []
    
    for ticker in TRAIN_TICKERS:
        data = get_stock_data(ticker, api_key) 
        if data is not None:
            features_df = create_features(data)
            all_train_data.append(features_df)
    
    if not all_train_data:
        print("No training data was fetched. Exiting.")
        return None
        
    train_df = pd.concat(all_train_data)
    
    train_df = train_df.dropna()
    
    if train_df.empty:
        print("No valid training data after processing. Exiting.")
        return None

    X_train = train_df[FEATURES]
    y_train = train_df['target']
    
    print(f"\nTraining model on {len(X_train)} data points...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("Model training complete.")
    
    try:
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': FEATURES,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        
        print("\nFeature Importances:")
        print(feature_importance_df)
    except Exception as e:
        print(f"Could not calculate feature importances: {e}")

    return model

# --- MODEL TESTING ---

def test_model(model, api_key):
    if model is None:
        print("No model to test.")
        return
        
    print("\n--- Starting Model Testing ---")
    
    for ticker in TEST_TICKERS:
        print(f"\n--- Evaluating: {ticker} ---")
        test_data = get_stock_data(ticker, api_key) 
        
        if test_data is None:
            continue
            
        test_df = create_features(test_data)
        
        test_df = test_df.dropna()
        
        if test_df.empty:
            print(f"Not enough data for {ticker} after processing.")
            continue
            
        X_test = test_df[FEATURES]
        y_test = test_df['target']
        
        predictions = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test, 
            predictions, 
            target_names=['Down (0)', 'Up (1)'],
            zero_division=0
        )
        
        print(f"Accuracy for {ticker}: {accuracy * 100:.2f}%")
        print("Classification Report:")
        print(report)

# --- MAIN EXECUTION ---

def load_api_key_from_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip().startswith('API_KEY='):
                    key = line.strip().split('=', 1)[1].strip().strip('"')
                    if key and key != "YOUR_API_KEY":
                        return key
    except FileNotFoundError:
        return None 
    except Exception as e:
        print(f"An error occurred while reading the .env file: {e}")
        return None

    return "YOUR_API_KEY" 

def main():
    print("===== Stock Market Prediction Model =====")
    
    api_key = load_api_key_from_env()

    if api_key is None or api_key == "YOUR_API_KEY":
        print("=" * 40)
        print("ERROR: API key not found.")
        print("       Please create a .env file and add your API key.")
        print("=" * 40)
        return

    model = train_model(api_key)
    
    test_model(model, api_key)

if __name__ == "__main__":
    main()