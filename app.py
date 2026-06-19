import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import plotly.express as px
import re
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Racing Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    # Find the exact absolute path of the folder this script is running from
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    model = xgb.XGBClassifier()
    # Safely construct absolute file paths to prevent cloud loading errors
    model.load_model(os.path.join(BASE_DIR, 'models', 'xgb_model.json'))
    
    with open(os.path.join(BASE_DIR, 'models', 'categorical_mappings.json'), 'r') as f:
        mappings = json.load(f)
        
    # Load the Horse Encyclopedia with error handling
    try:
        horse_stats = pd.read_csv(os.path.join(BASE_DIR, 'models', 'horse_stats.csv'))
    except FileNotFoundError:
        horse_stats = pd.DataFrame(columns=['horse', 'career_runs', 'win_strike_rate'])
        
    return model, mappings, horse_stats

def process_new_data(df, mappings, horse_stats):
    # Standardize column naming conventions
    df.columns = df.columns.str.strip().str.lower()
    
    rename_map = {
        'course': 'racecourse', 'ran': 'runners', 'type': 'race type', 
        'horse name': 'horse', 'name': 'horse', 'race time': 'time'
    }
    df = df.rename(columns=rename_map)
    
    # --- ULTIMATE NAME CLEANING LOGIC ---
    def clean_name(name):
        name = str(name)
        name = re.sub(r'\([^)]*\)', '', name)  # Remove country codes like (IRE), (GB)
        name = re.sub(r'[^\w\s]', '', name)    # Strip punctuation marks
        name = ' '.join(name.split()).lower()  # Normalize spacing and casing
        return name
    
    df['horse_clean'] = df['horse'].apply(clean_name)
    horse_stats['horse_clean'] = horse_stats['horse'].apply(clean_name)
    
    # --- FIX: ELIMINATE DUPLICATES IN ENCYCLOPEDIA ---
    # Keeps the latest record entry to prevent stacked multi-colored bars in the charts
    horse_stats = horse_stats.drop_duplicates(subset=['horse_clean'], keep='last')
    
    if 'horse' not in df.columns:
        df['horse'] = "Unknown"
    if 'time' not in df.columns:
        df['time'] = "TBA"
        
    # Drop existing feature columns from raw data if present to avoid merge collisions
    for col in ['career_runs', 'win_strike_rate']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Merge using normalized string keys
    df = pd.merge(df, horse_stats.drop(columns=['horse']), on='horse_clean', how='left')
    
    features_needed = ['draw', 'runners', 'career_runs', 'win_strike_rate']
    for col in features_needed:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    # --- FIX: CLEAN UP DISPLAY DECIMALS ---
    df['win_strike_rate'] = df['win_strike_rate'].round(2)
        
    # Map text values to categorical integer codes
    df['course_code'] = df['racecourse'].map(mappings['course']).fillna(0)
    df['going_code'] = df['going'].map(mappings['going']).fillna(0)
    
    # Drop intermediate processing columns
    df = df.drop(columns=['horse_clean'])
    
    return df

# --- UI START ---
st.title("🏇 AI Horse Racing Prediction Hub")
model, mappings, horse_stats = load_model_assets()

# --- SIDEBAR: DATA INGESTION ---
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Today's Racecard (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
    
    processed_df = process_new_data(raw_df, mappings, horse_stats)
    
    # Extra safety check: avoid duplicate evaluations if a horse is listed twice in card
    processed_df = processed_df.drop_duplicates(subset=['horse', 'time'])
    
    # Calculate predictions using exact feature columns
    feature_cols = ['course_code', 'going_code', 'draw', 'runners', 'career_runs', 'win_strike_rate']
    processed_df['win_prob'] = (model.predict_proba(processed_df[feature_cols])[:, 1] * 100).round(2)
    
    processed_df['racecourse'] = processed_df['racecourse'].astype(str).str.title()
    processed_df['time'] = processed_df['time'].astype(str)
    processed_df['race_display'] = processed_df['racecourse'] + " - " + processed_df['time']
    
    # --- HIGHLIGHT METRICS ---
    st.markdown("### 🏆 Top Predictions for Today")
    top_picks = processed_df.sort_values('win_prob', ascending=False).head(5)
    
    cols = st.columns(len(top_picks))
    for idx in range(len(top_picks)):
        row = top_picks.iloc[idx]
        horse_name = str(row.get('horse', 'Unknown'))
        win_prob = row.get('win_prob', 0)
        
        course_str = str(row.get('racecourse', 'Unknown'))
        time_str = str(row.get('time', 'TBA'))
        draw_val = int(row.get('draw', 0))
        sub_text = f"{course_str} @ {time_str} | Draw: {draw_val}"
        
        cols[idx].metric(label=horse_name, value=f"{win_prob}%", delta=sub_text, delta_color="off")

    # --- BRAIN EVALUATION VISUALIZATION ---
    st.markdown(" ")
    with st.expander("🧠 Inside the AI's Brain: See What Factors Matter Most", expanded=False):
        st.markdown("""
        This live visualization shows the exact **mathematical importance** your XGBoost model assigns to each factor. 
        The longer the bar, the more heavily the AI relied on that specific metric to calculate today's win probabilities.
        """)
        
        # Read raw gain/importance metrics directly from the model
        importances = model.feature_importances_
        
        feature_labels = {
            'course_code': '🗺️ Course Layout & Track Suitability',
            'going_code': '💧 Track Going (Ground Conditions)',
            'draw': '🚪 Stall / Draw Position',
            'runners': '👥 Field Size (Total Runners)',
            'career_runs': '📈 Race Experience (Career Runs)',
            'win_strike_rate': '🥇 Historical Win Strike Rate'
        }
        
        importance_df = pd.DataFrame({
            'Metric Factors': [feature_labels.get(col, col) for col in feature_cols],
            'Influence Weight': importances * 100
        }).sort_values('Influence Weight', ascending=True)
        
        # Plot structural features cleanly
        fig_brain = px.bar(
            importance_df, x='Influence Weight', y='Metric Factors', orientation='h',
            color='Influence Weight', color_continuous_scale='Tealgrn',
            text='Influence Weight'
        )
        fig_brain.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_brain.update_layout(
            xaxis_title="Influence on Final Prediction Score (%)",
            yaxis_title="",
            showlegend=False,
            height=340,
            margin=dict(l=10, r=40, t=10, b=10)
        )
        fig_brain.update_coloraxes(showscale=False)
        st.plotly_chart(fig_brain, use_container_width=True)

    st.divider()

    # --- MAIN INTERACTIVE SELECTION ---
    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        selected_race = st.selectbox("Select a Race to Visualize", sorted(processed_df['race_display'].unique()))
        race_subset = processed_df[processed_df['race_display'] == selected_race].sort_values('win_prob', ascending=True)
        
        fig = px.bar(
            race_subset, x='win_prob', y='horse', orientation='h', 
            title=f"Win Probabilities: {selected_race}", color='win_prob', 
            color_continuous_scale='RdYlGn', text='win_prob'
        )
        # --- FIX: FORCES CLEAN STRING FORMATTING TO PREVENT FLOATING POINT TRAILS ---
        fig.update_traces(texttemplate='%{x:.2f}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col_data:
        st.markdown(f"#### Details: {selected_race}")
        st.dataframe(
            race_subset.sort_values('win_prob', ascending=False)[['horse', 'win_prob', 'draw', 'runners', 'win_strike_rate']], 
            use_container_width=True, hide_index=True
        )

else:
    st.info("👋 Welcome! Please upload 'Today's Racecard' CSV file in the sidebar to generate AI predictions.")
    st.write("Recommended CSV columns: 'course', 'time', 'going', 'horse', 'draw', 'runners'.")
