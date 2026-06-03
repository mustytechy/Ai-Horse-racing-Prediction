import streamlit as st
import pandas as pd
import joblib
import os
from utils import engineer_features

# Set page configuration
st.set_page_config(page_title="AI Horse Racing Predictor", page_icon="🏇", layout="wide")

st.title("🏇 AI Race Predictor Dashboard")
st.write("Upload your daily race cards below to let the AI scan for potential winners.")

# --- 1. SAFE MODEL LOADER ---
@st.cache_resource
def load_prediction_model():
    """Loads the model file once and caches it to keep the app lightning fast."""
    # Check root directory first (based on your recent GitHub upload)
    root_path = 'race_predictor.pkl'
    folder_path = os.path.join('models', 'race_predictor.pkl')
    
    if os.path.exists(root_path):
        return joblib.load(root_path)
    elif os.path.exists(folder_path):
        return joblib.load(folder_path)
    return None

model = load_prediction_model()

# Show status of the AI engine
if model is None:
    st.error("❌ Error: Could not load the model file ('race_predictor.pkl'). Please check your GitHub repository structure.")
    st.stop()  # Stop the app right here if the model is missing
else:
    st.sidebar.success("🤖 AI Model Engine: Active & Loaded")

# --- 2. ROBUST FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload Daily Race Card (CSV or Excel format)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Step A: Handle Excel files
        if uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.success("📊 Excel file successfully loaded!")
            
        # Step B: Handle CSV files with Unicode safety fallback
        else:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)  # Rewind the file pointer to the start
                df = pd.read_csv(uploaded_file, encoding='latin1')
            st.success("📊 CSV file successfully loaded!")

        # --- 3. FEATURE ENGINEERING & PREDICTIONS ---
        st.write("---")
        st.subheader("🔮 Processing Predictions")
        
        with st.spinner("Analyzing past form and engineering features..."):
            # Call your function from utils.py
            df_features = engineer_features(df)
        
        with st.spinner("Running predictive models..."):
            # Automatically align columns with what the scikit-learn model expects
            if hasattr(model, 'feature_names_in_'):
                X = df_features[model.feature_names_in_]
            else:
                X = df_features
                
            # Generate predictions (0 = No Win, 1 = Predicted Winner)
            predictions = model.predict(X)
            
            # Extract win probabilities if the model supports it
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = None

        # Append outcomes back onto the original readable dataframe
        df['AI_Predicted_Winner'] = predictions
        if probabilities is not None:
            df['Win_Confidence'] = [f"{p*100:.1f}%" for p in probabilities]

        # --- 4. DISPLAY RESULTS ---
        st.subheader("🏁 Prediction Results Table")
        st.write("Use the controls below to filter down to the AI's top picks.")
        
        # Interactive UI Filter
        winners_only = st.checkbox("🎯 Show AI Predicted Winners Only", value=False)
        
        display_df = df.copy()
        if winners_only:
            # Filter rows where the model outputted a '1'
            display_df = display_df[display_df['AI_Predicted_Winner'] == 1]
            
        if display_df.empty:
            st.warning("No horses were flagged as outright winners based on your current filter criteria.")
        else:
            # Interactive Streamlit data grid
            st.dataframe(display_df, use_container_width=True)
            
        # --- 5. EXPORT RESULTS ---
        st.write("---")
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Output Predictions to CSV",
            data=csv_data,
            file_name="daily_race_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ A processing error occurred: {e}")
        st.info("Tip: Double check that the uploaded spreadsheet columns perfectly match the format your model was trained on.")
