import streamlit as st
import pandas as pd
import xgboost as xgb
import json
import plotly.express as px
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro Racing Predictor", layout="wide")

# --- ASSET LOADING ---
@st.cache_resource
def load_model_assets():
    model = xgb.XGBClassifier()
    model.load_model('models/xgb_model.json')
    
    with open('models/categorical_mappings.json', 'r') as f:
        mappings = json.load(f)
        
    # Load the new Horse Encyclopedia
    try:
        horse_stats = pd.read_csv('models/horse_stats.csv')
    except FileNotFoundError:
        horse_stats = pd.DataFrame(columns=['horse', 'career_runs', 'win_strike_rate'])
        
    return model, mappings, horse_stats

def process_new_data(df, mappings, horse_stats):
    df.columns = df.columns.str.strip().str.lower()
    
    rename_map = {
        'course': 'racecourse', 'ran': 'runners', 'type': 'race type', 
        'horse name': 'horse', 'name': 'horse', 'race time': 'time'
    }
    df = df.rename(columns=rename_map)
    
    # --- ULTIMATE NAME CLEANING LOGIC ---
    def clean_name(name):
        name = str(name)
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'[^\w\s]', '', name)
        name = ' '.join(name.split()).lower()
        return name
    
    df['horse_clean'] = df['horse'].apply(clean_name)
    horse_stats['horse_clean'] = horse_stats['horse'].apply(clean_name)
    
    # FIX: DROP DUPLICATES IN ENCYCLOPEDIA 
    horse_stats = horse_stats.drop_duplicates(subset=['horse_clean'], keep='last')
    
    if 'horse' not in df.columns:
        df['horse'] = "Unknown"
    if 'time' not in df.columns:
        df['time'] = "TBA"
        
    # Drop existing stat columns in raw data (if any) to prevent duplication during merge
    for col in ['career_runs', 'win_strike_rate']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # Merge using the new 'clean' column, dropping the duplicate horse name first
    df = pd.merge(df, horse_stats.drop(columns=['horse']), on='horse_clean', how='left')
    
    features_needed = ['draw', 'runners', 'career_runs', 'win_strike_rate']
    for col in features_needed:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['win_strike_rate'] = df['win_strike_rate'].round(2)
        
    df['course_code'] = df['racecourse'].map(mappings['course']).fillna(0)
    df['going_code'] = df['going'].map(mappings['going']).fillna(0)
    
    # Drop the temporary clean name column
    df = df.drop(columns=['horse_clean'])
    
    return df

# --- UI START ---
st.title("🏇 AI Horse Racing Prediction Hub")
model, mappings, horse_stats = load_model_assets()

# --- SIDEBAR: UPLOAD ---
st.sidebar.header("Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Today's Racecard (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
    
    processed_df = process_new_data(raw_df, mappings, horse_stats)
    
    # Extra safety check: drop duplicates from the uploaded race card
    processed_df = processed_df.drop_duplicates(subset=['horse', 'time'])
    
    feature_cols = ['course_code', 'going_code', 'draw', 'runners', 'career_runs', 'win_strike_rate']
    processed_df['win_prob'] = (model.predict_proba(processed_df[feature_cols])[:, 1] * 100).round(2)
    
    processed_df['racecourse'] = processed_df['racecourse'].astype(str).str.title()
    processed_df['time'] = processed_df['time'].astype(str)
    processed_df['race_display'] = processed_df['racecourse'] + " - " + processed_df['time']
    
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

    # --- NEW FEATURE: BRAIN EVALUATION VISUALIZATION ---
    st.markdown(" ")
    with st.expander("🧠 Inside the AI's Brain: See What Factors Matter Most", expanded=False):
        st.markdown("""
        This live visualization shows the exact **mathematical importance** your XGBoost model assigns to each factor. 
        The longer the bar, the more heavily the AI relied on that specific metric to calculate today's win probabilities.
        """)
        
        # Pull global feature importance weights straight from the mathematical model
        importances = model.feature_importances_
        
        # Map raw column keys to beautiful, human-readable UI names
        feature_labels = {
            'course_code': '🗺️ Course Layout & Track Suitability',
            'going_code': '💧 Track Going (Ground Conditions)',
            'draw': '🚪 Stall / Draw Position Position',
            'runners': '👥 Field Size (Total Runners)',
            'career_runs': '📈 Race Experience (Career Runs)',
            'win_strike_rate': '🥇 Historical Win Strike Rate'
        }
        
        importance_df = pd.DataFrame({
            'Metric Factors': [feature_labels.get(col, col) for col in feature_cols],
            'Influence Weight': importances * 100  # Convert to clean percentages
        }).sort_values('Influence Weight', ascending=True)
        
        # Render a gorgeous horizontal bar chart using a sleek corporate tech palette
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
        fig_brain.update_coloraxes(showscale=False) # Hide color bar to keep UI clean
        st.plotly_chart(fig_brain, use_container_width=True)
    # ----------------------------------------------------

    st.divider()

    col_chart, col_data = st.columns([2, 1])
    with col_chart:
        selected_race = st.selectbox("Select a Race to Visualize", sorted(processed_df['race_display'].unique()))
        race_subset = processed_df[processed_df['race_display'] == selected_race].sort_values('win_prob', ascending=True)
        
        fig = px.bar(
            race_subset, x='win_prob', y='horse', orientation='h', 
            title=f"Win Probabilities: {selected_race}", color='win_prob', 
            color_continuous_scale='RdYlGn', text='win_prob'
        )
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