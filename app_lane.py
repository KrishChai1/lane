import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta
import random
import json

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚚 Lane Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🚚 Lane Optimization</h1>
    <p style="color: white; opacity: 0.9; margin-top: 0.5rem;">
        Transportation Management System - AI-Powered Analytics
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_model' not in st.session_state:
    st.session_state.data_model = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# US Cities
US_CITIES = {
    'Chicago': (41.8781, -87.6298),
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Houston': (29.7604, -95.3698),
    'Dallas': (32.7767, -96.7970),
    'Atlanta': (33.7490, -84.3880),
    'Miami': (25.7617, -80.1918),
    'Seattle': (47.6062, -122.3321),
    'Denver': (39.7392, -104.9903),
    'Boston': (42.3601, -71.0589)
}

CARRIERS = ['UPS', 'FedEx', 'USPS', 'DHL', 'XPO']

def generate_sample_data():
    """Generate sample data"""
    np.random.seed(42)
    loads = []
    for i in range(500):
        origin = random.choice(list(US_CITIES.keys()))
        dest = random.choice([c for c in US_CITIES.keys() if c != origin])
        loads.append({
            'Load_ID': f'LD{i+1000}',
            'Origin_City': origin,
            'Destination_City': dest,
            'Selected_Carrier': random.choice(CARRIERS),
            'Total_Weight_lbs': random.randint(1000, 45000),
            'Total_Cost': round(random.uniform(500, 5000), 2),
            'Pickup_Date': (datetime.now() - timedelta(days=random.randint(1, 60))).strftime('%Y-%m-%d'),
            'Service_Type': random.choice(['TL', 'LTL'])
        })
    return pd.DataFrame(loads)

def display_dashboard():
    """Dashboard tab"""
    if not st.session_state.data_model:
        st.info("📊 Welcome! Click 'Generate Sample Data' in sidebar to start")
        return
    
    df = st.session_state.data_model.get('mapping_load_details')
    if df is None:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Loads", len(df))
    with col2:
        st.metric("Total Cost", f"${df['Total_Cost'].sum():,.0f}")
    with col3:
        st.metric("Unique Lanes", df.groupby(['Origin_City', 'Destination_City']).ngroups)
    with col4:
        st.metric("Carriers", df['Selected_Carrier'].nunique())
    
    # Top lanes
    st.subheader("Top Lanes")
    lanes = df.groupby(['Origin_City', 'Destination_City']).size().nlargest(5)
    for (o, d), count in lanes.items():
        st.write(f"• {o} → {d}: {count} loads")

def display_ai_assistant():
    """AI Assistant tab"""
    st.subheader("🤖 AI Assistant")
    
    if not st.session_state.data_model:
        st.info("Load data first")
        return
    
    question = st.text_input("Ask about your data:")
    if question and st.button("Send"):
        response = "I can help analyze your transportation data."
        st.session_state.chat_history.append((question, response))
    
    for q, a in st.session_state.chat_history[-3:]:
        st.write(f"**You:** {q}")
        st.write(f"**Assistant:** {a}")

def main():
    with st.sidebar:
        st.header("Data Management")
        if st.button("Generate Sample Data", type="primary"):
            st.session_state.data_model['mapping_load_details'] = generate_sample_data()
            st.success("Generated 500 loads!")
            st.rerun()
        
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            st.session_state.data_model['mapping_load_details'] = pd.read_csv(uploaded)
            st.rerun()
    
    tabs = st.tabs(["📊 Dashboard", "🤖 AI Assistant"])
    with tabs[0]:
        display_dashboard()
    with tabs[1]:
        display_ai_assistant()

if __name__ == "__main__":
    main()
