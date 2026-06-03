import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def engineer_features(df):
    """
    Safely engineer features. This handles missing columns 
    without crashing on scalar default values.
    """
    df = df.copy()
    
    # 1. Target creation (1 for Win, 0 for Loss)
    if 'pos' in df.columns:
        df['won'] = (pd.to_numeric(df['pos'], errors='coerce') == 1).astype(int)
    else:
        df['won'] = 0 # Default if no result data
    
    # 2. Jockey Win Rate
    if 'jockey' in df.columns:
        jockey_stats = df.groupby('jockey')['won'].mean().to_dict()
        df['jockey_win_rate'] = df['jockey'].map(jockey_stats).fillna(0.1)
    else:
        df['jockey_win_rate'] = 0.1
    
    # 3. Days Since Last Run (FIXED: Checks column existence first)
    if 'runner_last_run' in df.columns:
        df['days_since_run'] = pd.to_numeric(
            df['runner_last_run'].astype(str).str.extract(r'(\d+)')[0], 
            errors='coerce'
        ).fillna(365)
    else:
        df['days_since_run'] = 365
    
    # 4. Numeric Features (FIXED: Uses .get safely on the DataFrame)
    df['or'] = pd.to_numeric(df.get('or', 0), errors='coerce').fillna(0)
    df['wgt'] = pd.to_numeric(df.get('wgt', 0), errors='coerce').fillna(0)
    
    return df

def train():
    # Load and clean
    df = pd.read_csv(r'C:\Users\musty\Downloads\Horseracing\data\raceform.csv', low_memory=False)
    df = engineer_features(df)
    
    features = ['or', 'wgt', 'jockey_win_rate', 'days_since_run']
    X = df[features]
    y = df['won']
    
    # Train
    print("[STATUS] Training model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
    model.fit(X, y)
    
    # Save
    if not os.path.exists('models'): os.makedirs('models')
    # The '3' is a good balance between file size and speed
    joblib.dump(model, 'models/race_predictor.pkl', compress=5)
    print("[SUCCESS] Model trained and saved.")

if __name__ == "__main__":
    train()