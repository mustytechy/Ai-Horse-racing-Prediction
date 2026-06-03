import pandas as pd

def engineer_features(df):
    df = df.copy()
    
    # 1. Jockey Stats (Using 0.15 as a standard baseline for daily predictions)
    if 'jockey' in df.columns:
        # If we had a live stat mapping, we'd apply it here. 
        # For daily cards without history in the CSV, we use the baseline.
        pass
    df['jockey_win_rate'] = 0.15 
    
    # 2. Days Since Last Run
    if 'runner_last_run' in df.columns:
        df['days_since_run'] = pd.to_numeric(
            df['runner_last_run'].astype(str).str.extract(r'(\d+)')[0], 
            errors='coerce'
        ).fillna(365)
    else:
        df['days_since_run'] = 365
        
    # 3. Official Rating (Handles BOTH 'or' from training and 'runner_ofr' from daily cards)
    if 'or' in df.columns:
        df['or'] = pd.to_numeric(df['or'], errors='coerce').fillna(0)
    elif 'runner_ofr' in df.columns:
        df['or'] = pd.to_numeric(df['runner_ofr'], errors='coerce').fillna(0)
    else:
        df['or'] = 0

    # 4. Weight (Handles BOTH 'wgt' from training and 'runner_lbs' from daily cards)
    if 'wgt' in df.columns:
        df['wgt'] = pd.to_numeric(df['wgt'], errors='coerce').fillna(0)
    elif 'runner_lbs' in df.columns:
        df['wgt'] = pd.to_numeric(df['runner_lbs'], errors='coerce').fillna(0)
    else:
        df['wgt'] = 0

    return df