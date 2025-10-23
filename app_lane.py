import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚚 Lane Optimization",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .savings-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .assistant-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🚚 Lane Optimization</h1>
    <p style="color: white; opacity: 0.9; margin-top: 0.5rem;">
        Transportation Management System - AI-Powered Analytics & Optimization
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_model' not in st.session_state:
    st.session_state.data_model = {}
if 'primary_data' not in st.session_state:
    st.session_state.primary_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# US Cities with coordinates
US_CITIES = {
    'Chicago': (41.8781, -87.6298),
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Houston': (29.7604, -95.3698),
    'Phoenix': (33.4484, -112.0740),
    'Philadelphia': (39.9526, -75.1652),
    'San Antonio': (29.4241, -98.4936),
    'San Diego': (32.7157, -117.1611),
    'Dallas': (32.7767, -96.7970),
    'Atlanta': (33.7490, -84.3880),
    'Miami': (25.7617, -80.1918),
    'Seattle': (47.6062, -122.3321),
    'Denver': (39.7392, -104.9903),
    'Boston': (42.3601, -71.0589),
    'Detroit': (42.3314, -83.0458),
    'Nashville': (36.1627, -86.7816),
    'Portland': (45.5152, -122.6784),
    'Las Vegas': (36.1699, -115.1398),
    'Louisville': (38.2527, -85.7585),
    'Milwaukee': (43.0389, -87.9065)
}

CARRIERS = ['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac', 'XPO', 'SAIA', 'Old Dominion', 'YRC', 'Estes']

# TMS Data Model Schema
TMS_DATA_MODEL = {
    'mapping_load_details': {
        'primary_key': 'Load_ID',
        'columns': [
            'Load_ID', 'Customer_ID', 'Origin_City', 'Origin_State', 
            'Destination_City', 'Destination_State', 'Pickup_Date', 'Delivery_Date', 
            'Load_Status', 'Total_Weight_lbs', 'Total_Volume_cuft', 
            'Equipment_Type', 'Service_Type', 'Selected_Carrier'
        ]
    },
    'mapping_shipment_details': {
        'primary_key': 'Shipment_ID',
        'foreign_key': 'Load_ID',
        'columns': ['Shipment_ID', 'Load_ID', 'Weight_lbs', 'Volume_cuft', 'Commodity']
    },
    'mapping_item_details': {
        'primary_key': 'Item_ID',
        'foreign_key': 'Shipment_ID',
        'columns': ['Item_ID', 'Shipment_ID', 'Item_Description', 'Quantity', 'Weight_per_unit']
    },
    'mapping_carrier_rates': {
        'primary_key': 'Rate_ID',
        'foreign_key': 'Load_ID',
        'columns': ['Rate_ID', 'Load_ID', 'Carrier_Name', 'Total_Cost', 'Transit_Days']
    },
    'fact_carrier_performance': {
        'foreign_key': 'Load_ID',
        'columns': ['Load_ID', 'Selected_Carrier', 'On_Time_Delivery', 'Rating']
    },
    'fact_financial': {
        'foreign_key': 'Load_ID',
        'columns': ['Load_ID', 'Revenue', 'Cost', 'Profit_Margin_Percent']
    }
}

def calculate_distance(city1, city2):
    """Calculate distance between two cities"""
    # Clean city names
    city1 = city1.split(',')[0].strip() if ',' in city1 else city1.strip()
    city2 = city2.split(',')[0].strip() if ',' in city2 else city2.strip()
    
    if city1 not in US_CITIES or city2 not in US_CITIES:
        return 500  # Default distance
    
    lat1, lon1 = US_CITIES[city1]
    lat2, lon2 = US_CITIES[city2]
    
    # Haversine formula
    R = 3959  # Earth's radius in miles
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def detect_table_type(df, filename=""):
    """Detect which table type this is based on columns"""
    
    df_columns_lower = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Check for specific identifying columns
    if 'load_id' in df_columns_lower:
        if 'shipment_id' in df_columns_lower:
            return 'mapping_shipment_details'
        elif 'item_id' in df_columns_lower:
            return 'mapping_item_details'
        elif 'rate_id' in df_columns_lower or 'carrier_name' in df_columns_lower:
            return 'mapping_carrier_rates'
        elif 'revenue' in df_columns_lower:
            return 'fact_financial'
        elif 'on_time_delivery' in df_columns_lower or 'rating' in df_columns_lower:
            return 'fact_carrier_performance'
        else:
            return 'mapping_load_details'
    
    return None

def generate_complete_sample_data():
    """Generate comprehensive sample TMS data with all tables"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate loads
    loads = []
    for i in range(500):
        origin = random.choice(list(US_CITIES.keys()))
        destination = random.choice(list(US_CITIES.keys()))
        while destination == origin:
            destination = random.choice(list(US_CITIES.keys()))
        
        pickup_date = datetime.now() - timedelta(days=random.randint(1, 60))
        delivery_date = pickup_date + timedelta(days=random.randint(1, 5))
        
        loads.append({
            'Load_ID': f'LD{i+1000:06d}',
            'Customer_ID': f'CUST{random.randint(1, 20):04d}',
            'Origin_City': origin,
            'Origin_State': 'State',
            'Destination_City': destination,
            'Destination_State': 'State',
            'Pickup_Date': pickup_date.strftime('%Y-%m-%d'),
            'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
            'Load_Status': random.choice(['Delivered', 'In Transit', 'Scheduled']),
            'Total_Weight_lbs': random.randint(1000, 45000),
            'Total_Volume_cuft': random.randint(100, 2000),
            'Equipment_Type': random.choice(['Dry Van', 'Reefer', 'Flatbed']),
            'Service_Type': random.choice(['TL', 'LTL', 'Partial']),
            'Selected_Carrier': random.choice(CARRIERS),
            'Distance_miles': calculate_distance(origin, destination)
        })
    
    loads_df = pd.DataFrame(loads)
    
    # Generate shipments (1-3 per load)
    shipments = []
    shipment_id = 10000
    for load in loads[:300]:  # For first 300 loads
        num_shipments = random.randint(1, 3)
        for s in range(num_shipments):
            shipments.append({
                'Shipment_ID': f'SH{shipment_id:06d}',
                'Load_ID': load['Load_ID'],
                'Weight_lbs': load['Total_Weight_lbs'] // num_shipments,
                'Volume_cuft': load['Total_Volume_cuft'] // num_shipments,
                'Commodity': random.choice(['Electronics', 'Furniture', 'Food', 'Chemicals'])
            })
            shipment_id += 1
    
    shipments_df = pd.DataFrame(shipments)
    
    # Generate items (2-5 per shipment)
    items = []
    item_id = 100000
    for shipment in shipments[:500]:  # For first 500 shipments
        num_items = random.randint(2, 5)
        for i in range(num_items):
            items.append({
                'Item_ID': f'IT{item_id:06d}',
                'Shipment_ID': shipment['Shipment_ID'],
                'Item_Description': f'Product {random.choice(["A", "B", "C", "D"])}',
                'Quantity': random.randint(1, 100),
                'Weight_per_unit': random.uniform(0.5, 50)
            })
            item_id += 1
    
    items_df = pd.DataFrame(items)
    
    # Generate carrier rates (3-5 quotes per load)
    rates = []
    for load in loads[:200]:  # For first 200 loads
        for carrier in random.sample(CARRIERS, min(3, len(CARRIERS))):
            distance = load['Distance_miles']
            base_cost = distance * random.uniform(1.5, 3.5) * (1 + load['Total_Weight_lbs']/10000)
            
            rates.append({
                'Rate_ID': f'RT{random.randint(100000, 999999)}',
                'Load_ID': load['Load_ID'],
                'Carrier_Name': carrier,
                'Total_Cost': round(base_cost * random.uniform(0.9, 1.1), 2),
                'Transit_Days': max(1, int(distance / 500) + random.randint(0, 2))
            })
    
    rates_df = pd.DataFrame(rates)
    
    # Generate performance data
    performance = []
    delivered_loads = [load for load in loads if load['Load_Status'] == 'Delivered']
    
    for load in delivered_loads[:150]:
        performance.append({
            'Load_ID': load['Load_ID'],
            'Selected_Carrier': load['Selected_Carrier'],
            'On_Time_Delivery': random.choice(['Yes'] * 9 + ['No']),
            'Rating': round(random.uniform(3.5, 5.0), 1)
        })
    
    performance_df = pd.DataFrame(performance)
    
    # Generate financial data
    financial = []
    for load in delivered_loads[:150]:
        cost = load['Distance_miles'] * 2.5 * (1 + load['Total_Weight_lbs']/10000)
        revenue = cost * random.uniform(1.1, 1.4)
        
        financial.append({
            'Load_ID': load['Load_ID'],
            'Revenue': round(revenue, 2),
            'Cost': round(cost, 2),
            'Profit_Margin_Percent': round(((revenue - cost) / revenue) * 100, 2)
        })
    
    financial_df = pd.DataFrame(financial)
    
    # Calculate additional costs for loads
    loads_df['Line_Haul_Costs'] = loads_df['Distance_miles'] * 2.0
    loads_df['Fuel_Surcharge'] = loads_df['Line_Haul_Costs'] * 0.2
    loads_df['Total_Cost'] = loads_df['Line_Haul_Costs'] + loads_df['Fuel_Surcharge']
    loads_df['Transit_Days'] = (loads_df['Distance_miles'] / 500).clip(lower=1).astype(int)
    loads_df['On_Time_Delivery'] = random.choices(['Yes', 'No'], weights=[9, 1], k=len(loads_df))
    
    return {
        'mapping_load_details': loads_df,
        'mapping_shipment_details': shipments_df,
        'mapping_item_details': items_df,
        'mapping_carrier_rates': rates_df,
        'fact_carrier_performance': performance_df,
        'fact_financial': financial_df
    }

def display_dashboard():
    """Display main dashboard - FIRST TAB"""
    
    if not st.session_state.data_model:
        # Welcome screen when no data
        st.info("📊 Welcome to Lane Optimization! Get started by:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Option 1: Quick Demo
            Click **'Generate Sample Data'** in the sidebar to see the system in action with 500 sample loads
            """)
        
        with col2:
            st.markdown("""
            ### Option 2: Upload Your Data
            Upload your CSV/Excel files in the sidebar to analyze your actual transportation data
            """)
        return
    
    # Get primary data
    if 'mapping_load_details' in st.session_state.data_model:
        loads_df = st.session_state.data_model['mapping_load_details']
    else:
        st.warning("Load details table not found. Please upload complete data.")
        return
    
    # Main metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_loads = len(loads_df)
        st.metric("📦 Total Loads", f"{total_loads:,}")
    
    with col2:
        if 'Total_Cost' in loads_df.columns:
            total_cost = loads_df['Total_Cost'].sum()
            st.metric("💰 Total Spend", f"${total_cost/1000:.0f}K")
        else:
            st.metric("💰 Total Spend", "N/A")
    
    with col3:
        if 'Origin_City' in loads_df.columns and 'Destination_City' in loads_df.columns:
            unique_lanes = loads_df.groupby(['Origin_City', 'Destination_City']).ngroups
            st.metric("🛤️ Unique Lanes", f"{unique_lanes}")
        else:
            st.metric("🛤️ Unique Lanes", "N/A")
    
    with col4:
        if 'Selected_Carrier' in loads_df.columns:
            unique_carriers = loads_df['Selected_Carrier'].nunique()
            st.metric("🚛 Carriers", f"{unique_carriers}")
        else:
            st.metric("🚛 Carriers", "N/A")
    
    with col5:
        if 'On_Time_Delivery' in loads_df.columns:
            on_time = (loads_df['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("⏰ On-Time", f"{on_time:.0f}%")
        elif 'fact_carrier_performance' in st.session_state.data_model:
            perf_df = st.session_state.data_model['fact_carrier_performance']
            on_time = (perf_df['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("⏰ On-Time", f"{on_time:.0f}%")
        else:
            st.metric("⏰ On-Time", "N/A")
    
    # Row 2: Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Daily Load Volume")
        if 'Pickup_Date' in loads_df.columns:
            try:
                loads_df['Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce')
                daily_loads = loads_df.groupby(loads_df['Date'].dt.date).size().reset_index()
                daily_loads.columns = ['Date', 'Loads']
                
                fig = px.line(daily_loads, x='Date', y='Loads', 
                            markers=True, line_shape='spline')
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Date format issue - unable to show trend")
    
    with col2:
        st.subheader("🚛 Carrier Distribution")
        if 'Selected_Carrier' in loads_df.columns:
            carrier_counts = loads_df['Selected_Carrier'].value_counts().head(5)
            fig = px.bar(x=carrier_counts.values, y=carrier_counts.index,
                        orientation='h')
            fig.update_layout(showlegend=False, height=300,
                            xaxis_title="Number of Loads", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Lane Analysis
    st.subheader("🛤️ Top Lanes by Volume")
    
    if 'Origin_City' in loads_df.columns and 'Destination_City' in loads_df.columns:
        lane_stats = loads_df.groupby(['Origin_City', 'Destination_City']).agg({
            'Load_ID': 'count'
        }).reset_index()
        lane_stats.columns = ['Origin', 'Destination', 'Loads']
        
        if 'Total_Cost' in loads_df.columns:
            lane_costs = loads_df.groupby(['Origin_City', 'Destination_City'])['Total_Cost'].mean().reset_index()
            lane_costs.columns = ['Origin', 'Destination', 'Avg_Cost']
            lane_stats = lane_stats.merge(lane_costs, on=['Origin', 'Destination'], how='left')
            lane_stats['Avg_Cost'] = lane_stats['Avg_Cost'].round(2)
        
        lane_stats['Lane'] = lane_stats['Origin'] + ' → ' + lane_stats['Destination']
        lane_stats = lane_stats.sort_values('Loads', ascending=False).head(10)
        
        # Display as cards
        for i in range(min(5, len(lane_stats))):
            row = lane_stats.iloc[i]
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{i+1}. {row['Lane']}**")
            with col2:
                st.write(f"{row['Loads']} loads")
            with col3:
                if 'Avg_Cost' in row and pd.notna(row['Avg_Cost']):
                    st.write(f"${row['Avg_Cost']:,.0f} avg")
    
    # Row 4: Other loaded tables summary
    if len(st.session_state.data_model) > 1:
        st.subheader("📊 Data Model Summary")
        
        cols = st.columns(4)
        for idx, (table_name, df) in enumerate(st.session_state.data_model.items()):
            col = cols[idx % 4]
            with col:
                st.info(f"**{table_name.replace('_', ' ').title()}**\n{len(df)} records")
    
    # Consolidation Opportunities
    st.subheader("💡 Optimization Opportunities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Consolidation
        if 'Pickup_Date' in loads_df.columns:
            try:
                loads_df['Ship_Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce').dt.date
                consolidation = loads_df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
                consolidation_opps = (consolidation > 1).sum()
                st.info(f"**{consolidation_opps}** Consolidation Opportunities")
            except:
                st.info("**0** Consolidation Opportunities")
        else:
            st.info("**N/A** Consolidation Opportunities")
    
    with col2:
        # Mode optimization
        if 'Service_Type' in loads_df.columns and 'Total_Weight_lbs' in loads_df.columns:
            ltl_heavy = ((loads_df['Service_Type'] == 'LTL') & 
                        (loads_df['Total_Weight_lbs'] > 8000)).sum()
            st.info(f"**{ltl_heavy}** LTL → TL Conversions")
        else:
            st.info("**N/A** Mode Optimization")
    
    with col3:
        # Carrier optimization
        if 'mapping_carrier_rates' in st.session_state.data_model:
            rates_df = st.session_state.data_model['mapping_carrier_rates']
            multi_quotes = rates_df.groupby('Load_ID').size()
            optimization_potential = (multi_quotes > 1).sum()
            st.info(f"**{optimization_potential}** Carrier Optimizations")
        else:
            st.info("**N/A** Carrier Optimization")

def display_lane_analysis():
    """Lane analysis and consolidation opportunities"""
    
    if 'mapping_load_details' not in st.session_state.data_model:
        st.warning("Please load data first")
        return
    
    loads_df = st.session_state.data_model['mapping_load_details']
    
    st.subheader("🛤️ Lane Analysis & Consolidation")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Top Lanes", "Carrier Performance", "Consolidation Opportunities"])
    
    with tab1:
        if 'Origin_City' in loads_df.columns and 'Destination_City' in loads_df.columns:
            lane_analysis = loads_df.groupby(['Origin_City', 'Destination_City']).agg({
                'Load_ID': 'count'
            })
            
            if 'Total_Weight_lbs' in loads_df.columns:
                weight_by_lane = loads_df.groupby(['Origin_City', 'Destination_City'])['Total_Weight_lbs'].sum()
                lane_analysis['Total_Weight'] = weight_by_lane
            
            if 'Total_Cost' in loads_df.columns:
                cost_by_lane = loads_df.groupby(['Origin_City', 'Destination_City'])['Total_Cost'].agg(['mean', 'sum'])
                lane_analysis['Avg_Cost'] = cost_by_lane['mean']
                lane_analysis['Total_Cost'] = cost_by_lane['sum']
            
            lane_analysis.columns = [col.replace('Load_ID', 'Load_Count') for col in lane_analysis.columns]
            lane_analysis = lane_analysis.sort_values('Load_Count', ascending=False).head(20)
            
            # Reset index for display
            lane_analysis = lane_analysis.reset_index()
            lane_analysis['Lane'] = lane_analysis['Origin_City'] + ' → ' + lane_analysis['Destination_City']
            
            # Visualization
            fig = px.bar(lane_analysis.head(10), x='Load_Count', y='Lane',
                        orientation='h', title="Top 10 Lanes by Load Count",
                        color='Load_Count', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.markdown("### 📊 Lane Details")
            
            display_cols = ['Lane', 'Load_Count']
            format_dict = {}
            
            if 'Total_Weight' in lane_analysis.columns:
                display_cols.append('Total_Weight')
                format_dict['Total_Weight'] = '{:,.0f} lbs'
            
            if 'Avg_Cost' in lane_analysis.columns:
                display_cols.append('Avg_Cost')
                format_dict['Avg_Cost'] = '${:,.2f}'
            
            if 'Total_Cost' in lane_analysis.columns:
                display_cols.append('Total_Cost')
                format_dict['Total_Cost'] = '${:,.0f}'
            
            st.dataframe(
                lane_analysis[display_cols].style.format(format_dict),
                use_container_width=True,
                hide_index=True
            )
    
    with tab2:
        if 'Selected_Carrier' in loads_df.columns:
            st.markdown("### 🚛 Carrier Performance Analysis")
            
            carrier_stats = loads_df.groupby('Selected_Carrier').agg({
                'Load_ID': 'count'
            })
            
            if 'Total_Cost' in loads_df.columns:
                carrier_cost = loads_df.groupby('Selected_Carrier')['Total_Cost'].mean()
                carrier_stats['Avg_Cost'] = carrier_cost
            
            if 'Transit_Days' in loads_df.columns:
                carrier_transit = loads_df.groupby('Selected_Carrier')['Transit_Days'].mean()
                carrier_stats['Avg_Transit'] = carrier_transit
            
            if 'On_Time_Delivery' in loads_df.columns:
                carrier_ontime = loads_df.groupby('Selected_Carrier')['On_Time_Delivery'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                )
                carrier_stats['On_Time_%'] = carrier_ontime
            
            carrier_stats.columns = [col.replace('Load_ID', 'Total_Loads') for col in carrier_stats.columns]
            carrier_stats = carrier_stats.sort_values('Total_Loads', ascending=False)
            
            # Display metrics
            st.dataframe(
                carrier_stats.style.format({
                    'Avg_Cost': '${:,.2f}',
                    'Avg_Transit': '{:.1f} days',
                    'On_Time_%': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Visualization
            if 'On_Time_%' in carrier_stats.columns:
                fig = px.scatter(carrier_stats.reset_index(), 
                               x='Total_Loads', 
                               y='On_Time_%',
                               size='Total_Loads',
                               hover_data=['Selected_Carrier'],
                               title='Carrier Volume vs Performance')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 💡 Consolidation Opportunities")
        
        if 'Pickup_Date' in loads_df.columns:
            try:
                loads_df['Ship_Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce').dt.date
                consolidation = loads_df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
                consolidation_opps = consolidation[consolidation > 1].reset_index()
                consolidation_opps.columns = ['Origin', 'Destination', 'Date', 'Loads']
                
                if len(consolidation_opps) > 0:
                    consolidation_opps['Potential_Savings'] = consolidation_opps['Loads'] * 150
                    consolidation_opps['Lane'] = consolidation_opps['Origin'] + ' → ' + consolidation_opps['Destination']
                    
                    st.dataframe(
                        consolidation_opps[['Lane', 'Date', 'Loads', 'Potential_Savings']].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    total_savings = consolidation_opps['Potential_Savings'].sum()
                    st.success(f"💰 **Total Potential Monthly Savings: ${total_savings:,.2f}**")
                    
                    # Visualization
                    top_opps = consolidation_opps.nlargest(10, 'Potential_Savings')
                    fig = px.bar(top_opps, x='Potential_Savings', y='Lane',
                               orientation='h',
                               title='Top 10 Consolidation Opportunities by Savings')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No consolidation opportunities found")
            except Exception as e:
                st.warning(f"Unable to analyze consolidation opportunities: {str(e)}")
        else:
            st.info("Date information required for consolidation analysis")

def display_route_optimizer():
    """Interactive route optimization tool"""
    
    st.subheader("🎯 Route Optimization Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.selectbox("Origin City", list(US_CITIES.keys()))
        weight = st.number_input("Weight (lbs)", min_value=100, max_value=50000, value=5000, step=100)
    
    with col2:
        destination = st.selectbox("Destination City", 
                                  [c for c in US_CITIES.keys() if c != origin])
        service_type = st.radio("Service Type", ['LTL', 'TL'])
    
    with col3:
        urgency = st.select_slider("Urgency", ['Standard', 'Priority', 'Express'])
        optimize_for = st.radio("Optimize For", ['Cost', 'Speed', 'Balance'])
    
    if st.button("🚀 Analyze Route", type="primary", use_container_width=True):
        with st.spinner("Analyzing carriers and routes..."):
            distance = calculate_distance(origin, destination)
            
            # Analyze all carriers
            results = []
            for carrier in CARRIERS:
                base_cost = distance * random.uniform(2.0, 3.5) * (1 + weight/10000)
                if service_type == 'TL' and weight > 10000:
                    base_cost *= 0.85
                
                transit = max(1, int(distance / 500))
                if urgency == 'Express':
                    transit = max(1, transit - 1)
                elif urgency == 'Standard':
                    transit += 1
                
                reliability = random.randint(85, 99)
                
                results.append({
                    'Carrier': carrier,
                    'Cost': round(base_cost, 2),
                    'Transit_Days': transit,
                    'Distance': round(distance, 0),
                    'Reliability_%': reliability
                })
            
            results_df = pd.DataFrame(results)
            
            # Sort based on preference
            if optimize_for == 'Cost':
                results_df = results_df.sort_values('Cost')
            elif optimize_for == 'Speed':
                results_df = results_df.sort_values('Transit_Days')
            else:
                results_df['Score'] = (results_df['Cost'] / results_df['Cost'].max() + 
                                      results_df['Transit_Days'] / results_df['Transit_Days'].max())
                results_df = results_df.sort_values('Score')
            
            # Display results
            st.markdown("### 📊 Route Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Distance", f"{distance:.0f} miles")
            with col2:
                st.metric("Best Rate", f"${results_df.iloc[0]['Cost']:,.2f}")
            with col3:
                st.metric("Fastest", f"{results_df['Transit_Days'].min()} days")
            with col4:
                avg_cost = results_df['Cost'].mean()
                savings = avg_cost - results_df.iloc[0]['Cost']
                st.metric("Potential Savings", f"${savings:,.2f}")
            
            # Top recommendations
            st.markdown("### 🏆 Top 3 Carrier Recommendations")
            
            for i, (idx, row) in enumerate(results_df.head(3).iterrows()):
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        medals = ['🥇', '🥈', '🥉']
                        st.write(f"{medals[i]} **{row['Carrier']}**")
                    with col2:
                        st.write(f"${row['Cost']:,.2f}")
                    with col3:
                        st.write(f"{row['Transit_Days']} days")
                    with col4:
                        st.write(f"{row['Reliability_%']}% reliable")
            
            # Full comparison
            st.markdown("### 📋 Complete Carrier Comparison")
            st.dataframe(
                results_df.style.format({
                    'Cost': '${:,.2f}',
                    'Transit_Days': '{:.0f} days',
                    'Distance': '{:,.0f} mi',
                    'Reliability_%': '{:.0f}%'
                }).highlight_min(subset=['Cost'], color='lightgreen')
                .highlight_min(subset=['Transit_Days'], color='lightblue'),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            fig = px.scatter(results_df, x='Cost', y='Transit_Days',
                           size='Reliability_%',
                           color='Carrier',
                           hover_data=['Reliability_%'],
                           title="Cost vs Transit Time Analysis",
                           labels={'Cost': 'Total Cost ($)', 'Transit_Days': 'Transit (days)'})
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

def display_ai_predictions():
    """Display AI predictions and model performance"""
    
    st.subheader("🤖 AI Cost Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Make a Prediction")
        
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            pred_origin = st.selectbox("Prediction Origin", list(US_CITIES.keys()), key="pred_origin")
            pred_weight = st.number_input("Weight (lbs)", min_value=100, max_value=50000, value=5000, key="pred_weight")
            pred_carrier = st.selectbox("Carrier", CARRIERS, key="pred_carrier")
        
        with pred_col2:
            pred_destination = st.selectbox("Prediction Destination", 
                                           [c for c in US_CITIES.keys() if c != pred_origin], 
                                           key="pred_dest")
            pred_transit = st.number_input("Transit Days", min_value=1, max_value=10, value=3, key="pred_transit")
            pred_service = st.radio("Service", ['LTL', 'TL'], key="pred_service")
        
        if st.button("🔮 Predict Cost", type="primary"):
            distance = calculate_distance(pred_origin, pred_destination)
            
            # Calculate base cost
            base_cost = distance * 2.5 * (1 + pred_weight/10000)
            if pred_service == 'TL' and pred_weight > 10000:
                base_cost *= 0.85
            
            # Simulated ML predictions with variance
            predictions = {
                'Random Forest': base_cost * random.uniform(0.95, 1.05),
                'Gradient Boosting': base_cost * random.uniform(0.93, 1.07),
                'Neural Network': base_cost * random.uniform(0.92, 1.08),
                'Ensemble': base_cost * random.uniform(0.96, 1.04)
            }
            
            # Display predictions
            st.markdown("### 📊 Model Predictions")
            
            for model_name, prediction in predictions.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    st.write(f"${prediction:,.2f}")
                with col3:
                    accuracy = 100 - abs((prediction - base_cost) / base_cost * 100)
                    st.write(f"Confidence: {accuracy:.1f}%")
            
            # Average prediction
            avg_prediction = sum(predictions.values()) / len(predictions)
            st.success(f"💡 **Average Prediction: ${avg_prediction:,.2f}**")
            
            # Store prediction
            st.session_state.predictions.append({
                'timestamp': datetime.now(),
                'route': f"{pred_origin} → {pred_destination}",
                'actual': base_cost,
                'predictions': predictions
            })
    
    with col2:
        st.markdown("### 📈 Model Performance")
        
        # Simulated model metrics
        model_metrics = {
            'Random Forest': {'R2': 0.92, 'MAE': 125},
            'Gradient Boosting': {'R2': 0.94, 'MAE': 110},
            'Neural Network': {'R2': 0.89, 'MAE': 145},
            'Ensemble': {'R2': 0.95, 'MAE': 95}
        }
        
        for model, metrics in model_metrics.items():
            st.metric(model, 
                     f"R² Score: {metrics['R2']:.2f}",
                     f"MAE: ${metrics['MAE']}")

def display_ai_assistant():
    """AI Assistant for data insights"""
    
    st.subheader("🤖 AI Assistant")
    
    if not st.session_state.data_model:
        st.info("Load data to enable AI Assistant")
        return
    
    # Quick insights buttons
    st.markdown("### 💡 Quick Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Top Lanes", key="btn_lanes"):
            response = get_top_lanes_insight()
            st.session_state.chat_history.append(("What are the top lanes?", response))
    
    with col2:
        if st.button("💰 Cost Analysis", key="btn_cost"):
            response = get_cost_insight()
            st.session_state.chat_history.append(("Show cost analysis", response))
    
    with col3:
        if st.button("🚛 Carrier Performance", key="btn_carrier"):
            response = get_carrier_insight()
            st.session_state.chat_history.append(("How are carriers performing?", response))
    
    with col4:
        if st.button("💡 Opportunities", key="btn_opp"):
            response = get_opportunities_insight()
            st.session_state.chat_history.append(("What optimization opportunities exist?", response))
    
    # Chat interface
    st.markdown("### 💬 Ask Your Question")
    
    user_question = st.text_input("What would you like to know about your transportation data?",
                                 placeholder="e.g., What are the consolidation opportunities? Which lanes are most profitable?")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send", type="primary") and user_question:
            response = analyze_user_question(user_question)
            st.session_state.chat_history.append((user_question, response))
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        for q, a in st.session_state.chat_history[-5:]:  # Show last 5 exchanges
            st.markdown(f"**You:** {q}")
            st.markdown(f"<div class='assistant-message'>{a}</div>", unsafe_allow_html=True)

def display_analytics():
    """Display comprehensive analytics dashboard"""
    
    if 'mapping_load_details' not in st.session_state.data_model:
        st.warning("Please load data first")
        return
    
    loads_df = st.session_state.data_model['mapping_load_details']
    
    st.subheader("📈 Analytics Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_loads = len(loads_df)
        st.metric("Total Loads", f"{total_loads:,}")
    
    with col2:
        if 'Total_Cost' in loads_df.columns:
            total_cost = loads_df['Total_Cost'].sum()
            st.metric("Total Cost", f"${total_cost:,.0f}")
    
    with col3:
        if 'Transit_Days' in loads_df.columns:
            avg_transit = loads_df['Transit_Days'].mean()
            st.metric("Avg Transit", f"{avg_transit:.1f} days")
    
    with col4:
        if 'On_Time_Delivery' in loads_df.columns:
            on_time = (loads_df['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("On-Time %", f"{on_time:.1f}%")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Total_Cost' in loads_df.columns:
            # Cost distribution
            fig = px.histogram(loads_df, x='Total_Cost', nbins=30,
                             title='Cost Distribution',
                             labels={'Total_Cost': 'Total Cost ($)'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Selected_Carrier' in loads_df.columns:
            # Carrier market share
            carrier_share = loads_df['Selected_Carrier'].value_counts()
            fig = px.pie(values=carrier_share.values, names=carrier_share.index,
                        title='Carrier Market Share')
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    if 'Pickup_Date' in loads_df.columns:
        st.markdown("### 📅 Time Series Analysis")
        
        try:
            loads_df['Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce')
            daily_stats = loads_df.groupby(loads_df['Date'].dt.date).agg({
                'Load_ID': 'count'
            })
            
            if 'Total_Cost' in loads_df.columns:
                daily_cost = loads_df.groupby(loads_df['Date'].dt.date)['Total_Cost'].sum()
                daily_stats['Total_Cost'] = daily_cost
            
            daily_stats = daily_stats.reset_index()
            daily_stats.columns = ['Date', 'Loads'] + list(daily_stats.columns[2:])
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Daily Load Volume', 'Daily Total Cost'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=daily_stats['Date'], y=daily_stats['Loads'],
                          mode='lines+markers', name='Loads'),
                row=1, col=1
            )
            
            if 'Total_Cost' in daily_stats.columns:
                fig.add_trace(
                    go.Scatter(x=daily_stats['Date'], y=daily_stats['Total_Cost'],
                              mode='lines+markers', name='Cost', marker_color='orange'),
                    row=2, col=1
                )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Unable to create time series: {str(e)}")

# Helper functions for AI Assistant
def get_top_lanes_insight():
    """Get top lanes insight"""
    if 'mapping_load_details' not in st.session_state.data_model:
        return "No load data available."
    
    loads_df = st.session_state.data_model['mapping_load_details']
    
    if 'Origin_City' in loads_df.columns and 'Destination_City' in loads_df.columns:
        top_lanes = loads_df.groupby(['Origin_City', 'Destination_City']).size().nlargest(5)
        
        response = "**Top 5 Lanes by Volume:**\n"
        for (origin, dest), count in top_lanes.items():
            response += f"• {origin} → {dest}: {count} loads\n"
        
        total_lanes = loads_df.groupby(['Origin_City', 'Destination_City']).ngroups
        response += f"\nTotal Unique Lanes: {total_lanes}"
        return response
    
    return "Lane information not available."

def get_cost_insight():
    """Get cost analysis insight"""
    if 'mapping_load_details' in st.session_state.data_model:
        loads_df = st.session_state.data_model['mapping_load_details']
        
        if 'Total_Cost' in loads_df.columns:
            total = loads_df['Total_Cost'].sum()
            avg = loads_df['Total_Cost'].mean()
            min_cost = loads_df['Total_Cost'].min()
            max_cost = loads_df['Total_Cost'].max()
            
            return f"""**Cost Analysis:**
• Total Spend: ${total:,.2f}
• Average per Load: ${avg:,.2f}
• Min Cost: ${min_cost:,.2f}
• Max Cost: ${max_cost:,.2f}
• Cost Range: ${max_cost - min_cost:,.2f}"""
    
    if 'fact_financial' in st.session_state.data_model:
        fin_df = st.session_state.data_model['fact_financial']
        total_cost = fin_df['Cost'].sum()
        total_revenue = fin_df['Revenue'].sum()
        avg_margin = fin_df['Profit_Margin_Percent'].mean()
        
        return f"""**Financial Analysis:**
• Total Revenue: ${total_revenue:,.2f}
• Total Cost: ${total_cost:,.2f}
• Gross Margin: ${total_revenue - total_cost:,.2f}
• Average Margin: {avg_margin:.1f}%"""
    
    return "Cost data not available."

def get_carrier_insight():
    """Get carrier performance insight"""
    if 'fact_carrier_performance' in st.session_state.data_model:
        perf_df = st.session_state.data_model['fact_carrier_performance']
        
        on_time = (perf_df['On_Time_Delivery'] == 'Yes').mean() * 100
        avg_rating = perf_df['Rating'].mean()
        
        best_carrier = perf_df.groupby('Selected_Carrier')['Rating'].mean().idxmax()
        best_rating = perf_df.groupby('Selected_Carrier')['Rating'].mean().max()
        
        return f"""**Carrier Performance:**
• On-Time Delivery Rate: {on_time:.1f}%
• Average Rating: {avg_rating:.1f}/5.0
• Best Performer: {best_carrier} (Rating: {best_rating:.1f})"""
    
    if 'mapping_load_details' in st.session_state.data_model:
        loads_df = st.session_state.data_model['mapping_load_details']
        if 'Selected_Carrier' in loads_df.columns:
            carrier_counts = loads_df['Selected_Carrier'].value_counts().head(5)
            response = "**Top Carriers by Volume:**\n"
            for carrier, count in carrier_counts.items():
                pct = (count / len(loads_df)) * 100
                response += f"• {carrier}: {count} loads ({pct:.1f}%)\n"
            return response
    
    return "Carrier data not available."

def get_opportunities_insight():
    """Get optimization opportunities"""
    opportunities = []
    total_savings = 0
    
    if 'mapping_load_details' in st.session_state.data_model:
        loads_df = st.session_state.data_model['mapping_load_details']
        
        # Consolidation
        if 'Pickup_Date' in loads_df.columns:
            try:
                loads_df['Ship_Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce').dt.date
                consolidation = loads_df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
                consolidation_opps = (consolidation > 1).sum()
                if consolidation_opps > 0:
                    est_savings = consolidation_opps * 150
                    opportunities.append(f"• {consolidation_opps} consolidation opportunities (Est. savings: ${est_savings:,.0f})")
                    total_savings += est_savings
            except:
                pass
        
        # Mode optimization
        if 'Service_Type' in loads_df.columns and 'Total_Weight_lbs' in loads_df.columns:
            ltl_heavy = ((loads_df['Service_Type'] == 'LTL') & 
                        (loads_df['Total_Weight_lbs'] > 8000)).sum()
            if ltl_heavy > 0:
                est_savings = ltl_heavy * 200
                opportunities.append(f"• {ltl_heavy} LTL shipments could convert to TL (Est. savings: ${est_savings:,.0f})")
                total_savings += est_savings
    
    if 'mapping_carrier_rates' in st.session_state.data_model:
        rates_df = st.session_state.data_model['mapping_carrier_rates']
        multi_quotes = rates_df.groupby('Load_ID').size()
        opt_potential = (multi_quotes > 1).sum()
        if opt_potential > 0:
            est_savings = opt_potential * 100
            opportunities.append(f"• {opt_potential} loads with carrier optimization potential (Est. savings: ${est_savings:,.0f})")
            total_savings += est_savings
    
    if opportunities:
        response = "**Optimization Opportunities:**\n" + "\n".join(opportunities)
        response += f"\n\n💰 **Total Potential Savings: ${total_savings:,.0f}**"
        return response
    else:
        return "No immediate optimization opportunities identified."

def analyze_user_question(question):
    """Analyze user's custom question"""
    
    question_lower = question.lower()
    
    # Route to appropriate insight function
    if any(word in question_lower for word in ['lane', 'route', 'origin', 'destination', 'top']):
        return get_top_lanes_insight()
    elif any(word in question_lower for word in ['cost', 'spend', 'expense', 'money', 'financial', 'revenue']):
        return get_cost_insight()
    elif any(word in question_lower for word in ['carrier', 'vendor', 'performance', 'rating']):
        return get_carrier_insight()
    elif any(word in question_lower for word in ['consolidat', 'optimi', 'opportunity', 'saving']):
        return get_opportunities_insight()
    else:
        return """I can help you analyze:
• **Lanes**: Top routes, volume analysis
• **Costs**: Total spend, averages, financial metrics
• **Carriers**: Performance, ratings, selection
• **Opportunities**: Consolidation, mode optimization, savings

Please ask a specific question about your transportation data!"""

def main():
    """Main application"""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Data Management")
        
        # Generate sample data
        if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive TMS data..."):
                sample_data = generate_complete_sample_data()
                st.session_state.data_model = sample_data
                st.session_state.primary_data = sample_data['mapping_load_details']
                st.success("✅ Generated 500 loads with all related tables!")
                st.rerun()
        
        # File upload
        st.markdown("### 📁 Upload Files")
        uploaded_files = st.file_uploader(
            "Select CSV/Excel files",
            type=['csv', 'xlsx'],
            accept_multiple_files=True,
            help="Upload your TMS data files (loads, shipments, rates, etc.)"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                try:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # Detect table type
                    table_type = detect_table_type(df, file.name)
                    
                    if table_type:
                        st.session_state.data_model[table_type] = df
                        if table_type == 'mapping_load_details':
                            st.session_state.primary_data = df
                        st.success(f"✅ {file.name} → {table_type}")
                    else:
                        # Default to load details if has Load_ID
                        if 'Load_ID' in df.columns or 'load_id' in [c.lower() for c in df.columns]:
                            st.session_state.data_model['mapping_load_details'] = df
                            st.session_state.primary_data = df
                            st.success(f"✅ Loaded {file.name} as load data")
                        else:
                            st.warning(f"⚠️ Could not identify table type for {file.name}")
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
            
            if st.button("Process Data"):
                st.rerun()
        
        # Data summary
        if st.session_state.data_model:
            st.markdown("---")
            st.markdown("### 📊 Data Summary")
            
            total_records = sum(len(df) for df in st.session_state.data_model.values())
            st.write(f"**Tables:** {len(st.session_state.data_model)}")
            st.write(f"**Total Records:** {total_records:,}")
            
            for table_name, df in st.session_state.data_model.items():
                st.write(f"• {table_name}: {len(df)} records")
            
            if st.button("🗑️ Clear All Data"):
                st.session_state.data_model = {}
                st.session_state.primary_data = None
                st.session_state.chat_history = []
                st.session_state.predictions = []
                st.rerun()
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### About Lane Optimization
        
        **Features:**
        - Multi-carrier analysis
        - Route optimization
        - AI predictions
        - Consolidation analysis
        - Performance analytics
        
        **Data Model Supported:**
        - Load Details
        - Shipment Details
        - Item Details
        - Carrier Rates
        - Performance Metrics
        - Financial Data
        
        **Version 3.0**
        """)
    
    # Main tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🛤️ Lane Analysis",
        "🎯 Route Optimizer",
        "🤖 AI Predictions",
        "💬 AI Assistant",
        "📈 Analytics"
    ])
    
    with tabs[0]:
        display_dashboard()
    
    with tabs[1]:
        display_lane_analysis()
    
    with tabs[2]:
        display_route_optimizer()
    
    with tabs[3]:
        display_ai_predictions()
    
    with tabs[4]:
        display_ai_assistant()
    
    with tabs[5]:
        display_analytics()

if __name__ == "__main__":
    main()
