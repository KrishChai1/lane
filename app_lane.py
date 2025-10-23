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
    .data-model-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #667eea;
    }
    .relationship-badge {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .assistant-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🚚 Lane Optimization</h1>
    <p style="color: white; opacity: 0.9; margin-top: 0.5rem;">
        Transport Management System - Multi-Carrier Cost Analysis & Route Optimization
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
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# TMS Data Model Schema
TMS_DATA_MODEL = {
    'mapping_load_details': {
        'primary_key': 'Load_ID',
        'columns': [
            'Load_ID', 'Customer_ID', 'Origin_City', 'Origin_State', 'Origin_Zip',
            'Destination_City', 'Destination_State', 'Destination_Zip',
            'Pickup_Date', 'Delivery_Date', 'Load_Status', 'Total_Weight_lbs',
            'Total_Volume_cuft', 'Equipment_Type', 'Service_Type', 'Special_Requirements'
        ],
        'relationships': {
            'mapping_shipment_details': 'one-to-many',
            'mapping_carrier_rates': 'one-to-many',
            'fact_carrier_performance': 'one-to-many'
        }
    },
    'mapping_shipment_details': {
        'primary_key': 'Shipment_ID',
        'foreign_key': 'Load_ID',
        'columns': [
            'Shipment_ID', 'Load_ID', 'Shipment_Number', 'Weight_lbs', 'Volume_cuft',
            'Pieces', 'Pallet_Count', 'Commodity', 'Hazmat_Flag', 'Temperature_Controlled'
        ],
        'relationships': {
            'mapping_item_details': 'one-to-many'
        }
    },
    'mapping_item_details': {
        'primary_key': 'Item_ID',
        'foreign_key': 'Shipment_ID',
        'columns': [
            'Item_ID', 'Shipment_ID', 'Item_Description', 'Quantity', 'Weight_per_unit',
            'Item_Value', 'SKU', 'Serial_Number', 'Package_Type'
        ]
    },
    'mapping_carrier_rates': {
        'primary_key': 'Rate_ID',
        'foreign_key': 'Load_ID',
        'columns': [
            'Rate_ID', 'Load_ID', 'Carrier_ID', 'Carrier_Name', 'Base_Rate',
            'Line_Haul_Cost', 'Fuel_Surcharge', 'Accessorial_Charges',
            'Total_Cost', 'Transit_Days', 'Service_Level', 'Quote_Valid_Until'
        ]
    },
    'fact_carrier_performance': {
        'primary_key': 'Performance_ID',
        'foreign_key': 'Load_ID',
        'columns': [
            'Performance_ID', 'Load_ID', 'Carrier_ID', 'Selected_Carrier',
            'On_Time_Pickup', 'On_Time_Delivery', 'Actual_Pickup_Date',
            'Actual_Delivery_Date', 'Transit_Time_Hours', 'Damage_Claims',
            'Service_Failures', 'Rating'
        ]
    },
    'dim_customer': {
        'primary_key': 'Customer_ID',
        'columns': [
            'Customer_ID', 'Customer_Name', 'Industry', 'Credit_Limit',
            'Payment_Terms', 'Account_Status', 'Annual_Revenue', 'Primary_Lane'
        ]
    },
    'dim_carrier': {
        'primary_key': 'Carrier_ID',
        'columns': [
            'Carrier_ID', 'Carrier_Name', 'DOT_Number', 'MC_Number',
            'Insurance_Coverage', 'Safety_Rating', 'Service_Types',
            'Operating_Regions', 'Fleet_Size'
        ]
    },
    'fact_financial': {
        'primary_key': 'Transaction_ID',
        'foreign_key': 'Load_ID',
        'columns': [
            'Transaction_ID', 'Load_ID', 'Revenue', 'Cost', 'Gross_Margin',
            'Profit_Margin_Percent', 'Invoice_Number', 'Payment_Status',
            'Invoice_Date', 'Payment_Date'
        ]
    }
}

# US Cities with coordinates
US_CITIES = {
    'Chicago, IL': (41.8781, -87.6298),
    'New York, NY': (40.7128, -74.0060),
    'Los Angeles, CA': (34.0522, -118.2437),
    'Houston, TX': (29.7604, -95.3698),
    'Phoenix, AZ': (33.4484, -112.0740),
    'Philadelphia, PA': (39.9526, -75.1652),
    'San Antonio, TX': (29.4241, -98.4936),
    'San Diego, CA': (32.7157, -117.1611),
    'Dallas, TX': (32.7767, -96.7970),
    'Atlanta, GA': (33.7490, -84.3880),
    'Miami, FL': (25.7617, -80.1918),
    'Seattle, WA': (47.6062, -122.3321),
    'Denver, CO': (39.7392, -104.9903),
    'Boston, MA': (42.3601, -71.0589),
    'Detroit, MI': (42.3314, -83.0458),
    'Nashville, TN': (36.1627, -86.7816),
    'Portland, OR': (45.5152, -122.6784),
    'Las Vegas, NV': (36.1699, -115.1398),
    'Louisville, KY': (38.2527, -85.7585),
    'Milwaukee, WI': (43.0389, -87.9065)
}

CARRIERS = ['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac', 'XPO', 'SAIA', 'Old Dominion', 'YRC', 'Estes']

def detect_data_model(df, filename=""):
    """Detect which TMS table this dataframe represents"""
    
    # Check column overlap with each table schema
    best_match = None
    best_score = 0
    
    df_columns_lower = [col.lower().replace(' ', '_') for col in df.columns]
    
    for table_name, schema in TMS_DATA_MODEL.items():
        schema_columns_lower = [col.lower() for col in schema['columns']]
        
        # Calculate overlap score
        common_columns = set(df_columns_lower) & set(schema_columns_lower)
        score = len(common_columns) / len(schema_columns_lower)
        
        # Check for primary key
        if schema['primary_key'].lower() in df_columns_lower:
            score += 0.2
        
        if score > best_score:
            best_score = score
            best_match = table_name
    
    # Also check filename hints
    filename_lower = filename.lower()
    if 'load' in filename_lower:
        best_match = 'mapping_load_details'
    elif 'shipment' in filename_lower:
        best_match = 'mapping_shipment_details'
    elif 'item' in filename_lower:
        best_match = 'mapping_item_details'
    elif 'carrier' in filename_lower and 'rate' in filename_lower:
        best_match = 'mapping_carrier_rates'
    elif 'performance' in filename_lower:
        best_match = 'fact_carrier_performance'
    elif 'customer' in filename_lower:
        best_match = 'dim_customer'
    elif 'financial' in filename_lower:
        best_match = 'fact_financial'
    
    return best_match, best_score

def process_uploaded_file(uploaded_file):
    """Process uploaded file and detect its table type"""
    
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Detect table type
        table_type, confidence = detect_data_model(df, uploaded_file.name)
        
        return df, table_type, confidence
    except Exception as e:
        return None, None, str(e)

def calculate_distance(city1, city2):
    """Calculate distance between two cities"""
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

def generate_sample_tms_data():
    """Generate complete TMS sample data with all tables"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Generate customers
    customers = []
    for i in range(20):
        customers.append({
            'Customer_ID': f'CUST{i+1:04d}',
            'Customer_Name': f'Customer {i+1}',
            'Industry': random.choice(['Retail', 'Manufacturing', 'Healthcare', 'Technology', 'Automotive']),
            'Credit_Limit': random.randint(50000, 500000),
            'Payment_Terms': random.choice(['Net 30', 'Net 45', 'Net 60']),
            'Account_Status': 'Active',
            'Annual_Revenue': random.randint(1000000, 50000000)
        })
    customers_df = pd.DataFrame(customers)
    
    # Generate carriers
    carriers = []
    for i, name in enumerate(CARRIERS):
        carriers.append({
            'Carrier_ID': f'CARR{i+1:04d}',
            'Carrier_Name': name,
            'DOT_Number': f'DOT{random.randint(100000, 999999)}',
            'MC_Number': f'MC{random.randint(100000, 999999)}',
            'Insurance_Coverage': random.choice(['$1M', '$2M', '$5M']),
            'Safety_Rating': random.choice(['Satisfactory', 'Conditional', 'None']),
            'Fleet_Size': random.randint(50, 5000)
        })
    carriers_df = pd.DataFrame(carriers)
    
    # Generate loads
    loads = []
    load_id = 1000
    
    for _ in range(200):
        origin = random.choice(list(US_CITIES.keys()))
        destination = random.choice(list(US_CITIES.keys()))
        while destination == origin:
            destination = random.choice(list(US_CITIES.keys()))
        
        pickup_date = datetime.now() - timedelta(days=random.randint(1, 60))
        delivery_date = pickup_date + timedelta(days=random.randint(1, 5))
        
        loads.append({
            'Load_ID': f'LD{load_id:06d}',
            'Customer_ID': random.choice(customers)['Customer_ID'],
            'Origin_City': origin.split(',')[0],
            'Origin_State': origin.split(',')[1].strip(),
            'Destination_City': destination.split(',')[0],
            'Destination_State': destination.split(',')[1].strip(),
            'Pickup_Date': pickup_date.strftime('%Y-%m-%d'),
            'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
            'Load_Status': random.choice(['Delivered', 'In Transit', 'Scheduled', 'Cancelled']),
            'Total_Weight_lbs': random.randint(1000, 45000),
            'Total_Volume_cuft': random.randint(100, 2000),
            'Equipment_Type': random.choice(['Dry Van', 'Reefer', 'Flatbed', 'Step Deck']),
            'Service_Type': random.choice(['TL', 'LTL', 'Partial']),
            'Special_Requirements': random.choice(['None', 'Team Driver', 'Hazmat', 'Temperature Controlled'])
        })
        load_id += 1
    
    loads_df = pd.DataFrame(loads)
    
    # Generate shipments (1-3 per load)
    shipments = []
    shipment_id = 10000
    
    for load in loads:
        num_shipments = random.randint(1, 3)
        for s in range(num_shipments):
            shipments.append({
                'Shipment_ID': f'SH{shipment_id:06d}',
                'Load_ID': load['Load_ID'],
                'Shipment_Number': f"{load['Load_ID']}-{s+1:02d}",
                'Weight_lbs': load['Total_Weight_lbs'] // num_shipments,
                'Volume_cuft': load['Total_Volume_cuft'] // num_shipments,
                'Pieces': random.randint(1, 50),
                'Pallet_Count': random.randint(1, 26),
                'Commodity': random.choice(['Electronics', 'Furniture', 'Food', 'Chemicals', 'Textiles']),
                'Hazmat_Flag': 'Y' if load['Special_Requirements'] == 'Hazmat' else 'N',
                'Temperature_Controlled': 'Y' if load['Special_Requirements'] == 'Temperature Controlled' else 'N'
            })
            shipment_id += 1
    
    shipments_df = pd.DataFrame(shipments)
    
    # Generate carrier rates (3-5 quotes per load)
    carrier_rates = []
    rate_id = 100000
    
    for load in loads:
        distance = calculate_distance(
            load['Origin_City'] + ', ' + load['Origin_State'],
            load['Destination_City'] + ', ' + load['Destination_State']
        )
        
        num_quotes = random.randint(3, 5)
        selected_carriers = random.sample(carriers, num_quotes)
        
        for carrier in selected_carriers:
            base_rate = random.uniform(1.5, 4.0)
            line_haul = base_rate * distance * (1 + load['Total_Weight_lbs']/10000)
            fuel_surcharge = line_haul * random.uniform(0.15, 0.25)
            accessorials = random.uniform(0, 500)
            
            carrier_rates.append({
                'Rate_ID': f'RT{rate_id:06d}',
                'Load_ID': load['Load_ID'],
                'Carrier_ID': carrier['Carrier_ID'],
                'Carrier_Name': carrier['Carrier_Name'],
                'Base_Rate': round(base_rate, 2),
                'Line_Haul_Cost': round(line_haul, 2),
                'Fuel_Surcharge': round(fuel_surcharge, 2),
                'Accessorial_Charges': round(accessorials, 2),
                'Total_Cost': round(line_haul + fuel_surcharge + accessorials, 2),
                'Transit_Days': max(1, int(distance / 500) + random.randint(0, 2)),
                'Service_Level': random.choice(['Standard', 'Expedited', 'Economy']),
                'Quote_Valid_Until': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
            })
            rate_id += 1
    
    carrier_rates_df = pd.DataFrame(carrier_rates)
    
    # Generate carrier performance (for delivered loads)
    performance = []
    perf_id = 1000000
    
    delivered_loads = [load for load in loads if load['Load_Status'] == 'Delivered']
    
    for load in delivered_loads:
        # Get the lowest cost carrier for this load (simulating selection)
        load_rates = [r for r in carrier_rates if r['Load_ID'] == load['Load_ID']]
        if load_rates:
            selected_rate = min(load_rates, key=lambda x: x['Total_Cost'])
            
            pickup_date = datetime.strptime(load['Pickup_Date'], '%Y-%m-%d')
            delivery_date = datetime.strptime(load['Delivery_Date'], '%Y-%m-%d')
            
            performance.append({
                'Performance_ID': f'PF{perf_id:07d}',
                'Load_ID': load['Load_ID'],
                'Carrier_ID': selected_rate['Carrier_ID'],
                'Selected_Carrier': selected_rate['Carrier_Name'],
                'On_Time_Pickup': random.choice(['Yes', 'No']),
                'On_Time_Delivery': random.choice(['Yes'] * 9 + ['No']),  # 90% on-time
                'Actual_Pickup_Date': pickup_date.strftime('%Y-%m-%d'),
                'Actual_Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
                'Transit_Time_Hours': (delivery_date - pickup_date).total_seconds() / 3600,
                'Damage_Claims': 0 if random.random() > 0.05 else random.randint(1, 3),
                'Service_Failures': 0 if random.random() > 0.1 else 1,
                'Rating': round(random.uniform(3.5, 5.0), 1)
            })
            perf_id += 1
    
    performance_df = pd.DataFrame(performance)
    
    # Generate financial data
    financial = []
    trans_id = 10000000
    
    for load in delivered_loads:
        # Find the selected carrier cost
        load_rates = [r for r in carrier_rates if r['Load_ID'] == load['Load_ID']]
        if load_rates:
            selected_rate = min(load_rates, key=lambda x: x['Total_Cost'])
            cost = selected_rate['Total_Cost']
            revenue = cost * random.uniform(1.1, 1.4)
            
            financial.append({
                'Transaction_ID': f'TR{trans_id:08d}',
                'Load_ID': load['Load_ID'],
                'Revenue': round(revenue, 2),
                'Cost': round(cost, 2),
                'Gross_Margin': round(revenue - cost, 2),
                'Profit_Margin_Percent': round(((revenue - cost) / revenue) * 100, 2),
                'Invoice_Number': f'INV{random.randint(100000, 999999)}',
                'Payment_Status': random.choice(['Paid', 'Pending', 'Overdue']),
                'Invoice_Date': load['Delivery_Date'],
                'Payment_Date': (datetime.strptime(load['Delivery_Date'], '%Y-%m-%d') + 
                                timedelta(days=random.randint(15, 45))).strftime('%Y-%m-%d')
            })
            trans_id += 1
    
    financial_df = pd.DataFrame(financial)
    
    return {
        'mapping_load_details': loads_df,
        'mapping_shipment_details': shipments_df,
        'mapping_carrier_rates': carrier_rates_df,
        'fact_carrier_performance': performance_df,
        'fact_financial': financial_df,
        'dim_customer': customers_df,
        'dim_carrier': carriers_df
    }

def display_data_model_overview():
    """Display overview of the TMS data model"""
    
    st.subheader("📊 TMS Data Model Overview")
    
    # Check what tables are loaded
    loaded_tables = list(st.session_state.data_model.keys())
    
    if not loaded_tables:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("📤 Please upload your TMS data files or generate sample data")
            
            st.markdown("""
            ### 📁 Expected Tables:
            - **mapping_load_details** (Core table) - Load information
            - **mapping_shipment_details** - Shipments within loads
            - **mapping_item_details** - Items within shipments
            - **mapping_carrier_rates** - Carrier quotes for loads
            - **fact_carrier_performance** - Delivery performance metrics
            - **fact_financial** - Revenue and cost data
            - **dim_customer** - Customer master data
            - **dim_carrier** - Carrier master data
            """)
        
        with col2:
            st.markdown("""
            ### 🔗 Data Relationships:
            - Load → Shipments (1:Many)
            - Shipment → Items (1:Many)
            - Load → Carrier Rates (1:Many)
            - Load → Performance (1:1)
            - Load → Financial (1:1)
            """)
    else:
        # Show loaded tables
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.metric("Tables Loaded", len(loaded_tables))
        
        with col2:
            total_records = sum(len(df) for df in st.session_state.data_model.values())
            st.metric("Total Records", f"{total_records:,}")
        
        with col3:
            if 'mapping_load_details' in st.session_state.data_model:
                st.metric("Total Loads", len(st.session_state.data_model['mapping_load_details']))
        
        # Display each loaded table
        st.markdown("### 📋 Loaded Tables")
        
        for table_name in loaded_tables:
            df = st.session_state.data_model[table_name]
            
            with st.expander(f"**{table_name}** ({len(df)} records)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.markdown("**Columns:**")
                    for col in df.columns[:10]:  # Show first 10 columns
                        st.write(f"• {col}")
                    if len(df.columns) > 10:
                        st.write(f"... and {len(df.columns) - 10} more")
                
                # Show relationships if this is the load table
                if table_name == 'mapping_load_details' and len(loaded_tables) > 1:
                    st.markdown("**Related Data:**")
                    
                    if 'mapping_shipment_details' in st.session_state.data_model:
                        shipments_df = st.session_state.data_model['mapping_shipment_details']
                        load_ids = df['Load_ID'].unique()
                        related_shipments = shipments_df[shipments_df['Load_ID'].isin(load_ids)]
                        st.write(f"• {len(related_shipments)} related shipments")
                    
                    if 'mapping_carrier_rates' in st.session_state.data_model:
                        rates_df = st.session_state.data_model['mapping_carrier_rates']
                        related_rates = rates_df[rates_df['Load_ID'].isin(load_ids)]
                        st.write(f"• {len(related_rates)} carrier quotes")
        
        # Data quality check
        if 'mapping_load_details' in st.session_state.data_model:
            st.markdown("### 🔍 Data Quality Check")
            
            loads_df = st.session_state.data_model['mapping_load_details']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                missing_values = loads_df.isnull().sum().sum()
                st.metric("Missing Values", missing_values, 
                         delta="Good" if missing_values == 0 else f"{missing_values} to fix",
                         delta_color="off" if missing_values == 0 else "normal")
            
            with col2:
                if 'Load_Status' in loads_df.columns:
                    delivered = (loads_df['Load_Status'] == 'Delivered').sum()
                    st.metric("Delivered Loads", f"{delivered}/{len(loads_df)}")
            
            with col3:
                if 'Origin_City' in loads_df.columns and 'Destination_City' in loads_df.columns:
                    unique_lanes = loads_df.groupby(['Origin_City', 'Destination_City']).ngroups
                    st.metric("Unique Lanes", unique_lanes)
            
            with col4:
                if 'Pickup_Date' in loads_df.columns:
                    try:
                        dates = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce')
                        date_range_days = (dates.max() - dates.min()).days
                        st.metric("Date Range", f"{date_range_days} days")
                    except:
                        st.metric("Date Range", "N/A")

def display_integrated_analysis():
    """Display analysis using multiple related tables"""
    
    if not st.session_state.data_model:
        st.warning("⚠️ Please upload data files first")
        return
    
    st.subheader("🔄 Integrated Analysis")
    
    # Check what tables we have
    has_loads = 'mapping_load_details' in st.session_state.data_model
    has_rates = 'mapping_carrier_rates' in st.session_state.data_model
    has_performance = 'fact_carrier_performance' in st.session_state.data_model
    has_financial = 'fact_financial' in st.session_state.data_model
    
    if has_loads and has_rates:
        st.markdown("### 💰 Cost Analysis by Lane")
        
        loads_df = st.session_state.data_model['mapping_load_details']
        rates_df = st.session_state.data_model['mapping_carrier_rates']
        
        # Merge loads with rates
        merged = loads_df.merge(rates_df, on='Load_ID', how='left')
        
        # Find best rate per load
        best_rates = merged.loc[merged.groupby('Load_ID')['Total_Cost'].idxmin()]
        
        # Analyze by lane
        lane_analysis = best_rates.groupby(['Origin_City', 'Destination_City']).agg({
            'Load_ID': 'count',
            'Total_Cost': 'mean',
            'Transit_Days': 'mean',
            'Total_Weight_lbs': 'mean'
        }).round(2)
        
        lane_analysis.columns = ['Load_Count', 'Avg_Cost', 'Avg_Transit', 'Avg_Weight']
        lane_analysis = lane_analysis.sort_values('Load_Count', ascending=False).head(10)
        
        st.dataframe(lane_analysis, use_container_width=True)
        
        # Carrier selection analysis
        st.markdown("### 🚛 Carrier Selection Analysis")
        
        carrier_selection = best_rates['Carrier_Name'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=carrier_selection.values, 
                       names=carrier_selection.index,
                       title="Optimal Carrier Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost comparison
            carrier_costs = best_rates.groupby('Carrier_Name')['Total_Cost'].mean().sort_values()
            fig = px.bar(x=carrier_costs.values, y=carrier_costs.index,
                        orientation='h',
                        title="Average Cost by Carrier",
                        labels={'x': 'Average Cost ($)', 'y': 'Carrier'})
            st.plotly_chart(fig, use_container_width=True)
    
    if has_loads and has_performance:
        st.markdown("### 📊 Performance Metrics")
        
        loads_df = st.session_state.data_model['mapping_load_details']
        perf_df = st.session_state.data_model['fact_carrier_performance']
        
        # Merge for analysis
        perf_analysis = loads_df.merge(perf_df, on='Load_ID', how='inner')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            on_time_rate = (perf_analysis['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("On-Time Delivery", f"{on_time_rate:.1f}%")
        
        with col2:
            avg_rating = perf_analysis['Rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.1f}/5.0")
        
        with col3:
            damage_rate = (perf_analysis['Damage_Claims'] > 0).mean() * 100
            st.metric("Damage Rate", f"{damage_rate:.1f}%")
        
        # Carrier performance comparison
        carrier_perf = perf_analysis.groupby('Selected_Carrier').agg({
            'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100,
            'Rating': 'mean',
            'Load_ID': 'count'
        }).round(2)
        
        carrier_perf.columns = ['On_Time_%', 'Avg_Rating', 'Load_Count']
        carrier_perf = carrier_perf.sort_values('Avg_Rating', ascending=False)
        
        st.markdown("**Carrier Performance Rankings:**")
        st.dataframe(carrier_perf, use_container_width=True)
    
    if has_loads and has_financial:
        st.markdown("### 💵 Financial Performance")
        
        loads_df = st.session_state.data_model['mapping_load_details']
        financial_df = st.session_state.data_model['fact_financial']
        
        # Financial metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = financial_df['Revenue'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.0f}")
        
        with col2:
            total_cost = financial_df['Cost'].sum()
            st.metric("Total Cost", f"${total_cost:,.0f}")
        
        with col3:
            gross_margin = financial_df['Gross_Margin'].sum()
            st.metric("Gross Margin", f"${gross_margin:,.0f}")
        
        with col4:
            avg_margin_pct = financial_df['Profit_Margin_Percent'].mean()
            st.metric("Avg Margin %", f"{avg_margin_pct:.1f}%")
        
        # Profitability by lane
        fin_loads = loads_df.merge(financial_df, on='Load_ID', how='inner')
        
        lane_profitability = fin_loads.groupby(['Origin_City', 'Destination_City']).agg({
            'Revenue': 'sum',
            'Cost': 'sum',
            'Profit_Margin_Percent': 'mean',
            'Load_ID': 'count'
        }).round(2)
        
        lane_profitability['Net_Profit'] = lane_profitability['Revenue'] - lane_profitability['Cost']
        lane_profitability = lane_profitability.sort_values('Net_Profit', ascending=False).head(10)
        
        st.markdown("**Top 10 Most Profitable Lanes:**")
        st.dataframe(
            lane_profitability.style.format({
                'Revenue': '${:,.0f}',
                'Cost': '${:,.0f}',
                'Net_Profit': '${:,.0f}',
                'Profit_Margin_Percent': '{:.1f}%'
            }),
            use_container_width=True
        )

def display_ai_assistant_enhanced():
    """Enhanced AI Assistant that understands the full data model"""
    
    st.subheader("🤖 AI Assistant - TMS Expert")
    st.markdown("Ask questions about your transportation data across all tables!")
    
    if not st.session_state.data_model:
        st.info("Upload data to enable the AI Assistant")
        return
    
    # Quick insights based on loaded tables
    st.markdown("### 💡 Quick Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Data Model Summary"):
            response = "**Loaded TMS Tables:**\n"
            for table, df in st.session_state.data_model.items():
                response += f"• **{table}**: {len(df)} records, {len(df.columns)} columns\n"
            st.session_state.chat_history.append(("Data Model Summary", response))
    
    with col2:
        if st.button("🔄 Check Relationships"):
            response = analyze_relationships()
            st.session_state.chat_history.append(("Check Relationships", response))
    
    with col3:
        if st.button("💰 Cost Optimization"):
            response = analyze_cost_optimization()
            st.session_state.chat_history.append(("Cost Optimization", response))
    
    # Chat interface
    st.markdown("### 💬 Ask Your Question")
    
    user_question = st.text_area(
        "Type your question here...", 
        placeholder="e.g., Which carriers have the best performance? What are the consolidation opportunities? Show me profitability by customer.",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Send", type="primary"):
            if user_question:
                response = analyze_complex_question(user_question)
                st.session_state.chat_history.append((user_question, response))
    
    with col2:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation")
        for question, answer in st.session_state.chat_history[-5:]:
            st.markdown(f"**You:** {question}")
            st.markdown(f"<div class='assistant-message'>{answer}</div>", unsafe_allow_html=True)

def analyze_relationships():
    """Analyze data relationships between tables"""
    
    if not st.session_state.data_model:
        return "No data loaded to analyze relationships."
    
    response = "**Data Relationship Analysis:**\n\n"
    
    if 'mapping_load_details' in st.session_state.data_model:
        loads_df = st.session_state.data_model['mapping_load_details']
        total_loads = len(loads_df)
        
        response += f"**Core Data:**\n• Total Loads: {total_loads}\n"
        
        if 'mapping_shipment_details' in st.session_state.data_model:
            shipments_df = st.session_state.data_model['mapping_shipment_details']
            avg_shipments_per_load = len(shipments_df) / total_loads
            response += f"• Total Shipments: {len(shipments_df)}\n"
            response += f"• Avg Shipments/Load: {avg_shipments_per_load:.1f}\n"
        
        if 'mapping_carrier_rates' in st.session_state.data_model:
            rates_df = st.session_state.data_model['mapping_carrier_rates']
            avg_quotes_per_load = len(rates_df) / total_loads
            response += f"• Total Carrier Quotes: {len(rates_df)}\n"
            response += f"• Avg Quotes/Load: {avg_quotes_per_load:.1f}\n"
            
            # Check for orphaned records
            load_ids = set(loads_df['Load_ID'])
            rate_load_ids = set(rates_df['Load_ID'])
            orphaned_rates = rate_load_ids - load_ids
            if orphaned_rates:
                response += f"\n⚠️ Warning: {len(orphaned_rates)} rates without corresponding loads\n"
    
    return response

def analyze_cost_optimization():
    """Analyze cost optimization opportunities"""
    
    if 'mapping_load_details' not in st.session_state.data_model:
        return "Load data required for cost optimization analysis."
    
    loads_df = st.session_state.data_model['mapping_load_details']
    response = "**Cost Optimization Opportunities:**\n\n"
    
    # Consolidation opportunities
    if 'Pickup_Date' in loads_df.columns:
        loads_df['Ship_Date'] = pd.to_datetime(loads_df['Pickup_Date'], errors='coerce').dt.date
        same_lane_same_day = loads_df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
        consolidation_opps = same_lane_same_day[same_lane_same_day > 1]
        
        if len(consolidation_opps) > 0:
            total_consolidatable = consolidation_opps.sum()
            potential_savings = total_consolidatable * 200  # Estimated $200 per consolidation
            response += f"**Consolidation:**\n"
            response += f"• {len(consolidation_opps)} consolidation opportunities\n"
            response += f"• {total_consolidatable} loads can be consolidated\n"
            response += f"• Estimated savings: ${potential_savings:,.2f}\n\n"
    
    # Mode optimization
    if 'Total_Weight_lbs' in loads_df.columns and 'Service_Type' in loads_df.columns:
        ltl_heavy = loads_df[(loads_df['Service_Type'] == 'LTL') & (loads_df['Total_Weight_lbs'] > 8000)]
        if len(ltl_heavy) > 0:
            response += f"**Mode Optimization:**\n"
            response += f"• {len(ltl_heavy)} LTL shipments over 8,000 lbs\n"
            response += f"• Consider converting to TL for cost savings\n\n"
    
    # Carrier optimization
    if 'mapping_carrier_rates' in st.session_state.data_model:
        rates_df = st.session_state.data_model['mapping_carrier_rates']
        
        # Find average savings by using best carrier
        avg_savings_per_load = []
        for load_id in loads_df['Load_ID'].unique():
            load_rates = rates_df[rates_df['Load_ID'] == load_id]['Total_Cost']
            if len(load_rates) > 1:
                potential_saving = load_rates.mean() - load_rates.min()
                avg_savings_per_load.append(potential_saving)
        
        if avg_savings_per_load:
            avg_saving = np.mean(avg_savings_per_load)
            total_potential = avg_saving * len(loads_df)
            response += f"**Carrier Selection:**\n"
            response += f"• Average saving per load: ${avg_saving:.2f}\n"
            response += f"• Total potential savings: ${total_potential:,.2f}\n"
    
    return response

def analyze_complex_question(question):
    """Analyze complex questions across multiple tables"""
    
    if not st.session_state.data_model:
        return "Please upload data to enable analysis."
    
    question_lower = question.lower()
    
    # Performance questions
    if 'performance' in question_lower or 'on-time' in question_lower:
        if 'fact_carrier_performance' in st.session_state.data_model:
            perf_df = st.session_state.data_model['fact_carrier_performance']
            
            on_time_rate = (perf_df['On_Time_Delivery'] == 'Yes').mean() * 100
            
            carrier_perf = perf_df.groupby('Selected_Carrier').agg({
                'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100,
                'Rating': 'mean'
            }).round(2)
            
            best_carrier = carrier_perf['Rating'].idxmax()
            best_rating = carrier_perf.loc[best_carrier, 'Rating']
            
            return f"""**Performance Analysis:**
• Overall On-Time Rate: {on_time_rate:.1f}%
• Best Performing Carrier: {best_carrier} (Rating: {best_rating})
• Carriers with 95%+ On-Time: {', '.join(carrier_perf[carrier_perf['On_Time_Delivery'] > 95].index.tolist())}"""
    
    # Financial questions
    elif 'profit' in question_lower or 'revenue' in question_lower or 'margin' in question_lower:
        if 'fact_financial' in st.session_state.data_model:
            fin_df = st.session_state.data_model['fact_financial']
            
            return f"""**Financial Analysis:**
• Total Revenue: ${fin_df['Revenue'].sum():,.2f}
• Total Cost: ${fin_df['Cost'].sum():,.2f}
• Gross Margin: ${fin_df['Gross_Margin'].sum():,.2f}
• Average Margin: {fin_df['Profit_Margin_Percent'].mean():.1f}%
• Loads with 20%+ Margin: {(fin_df['Profit_Margin_Percent'] > 20).sum()}"""
    
    # Customer questions
    elif 'customer' in question_lower:
        if 'dim_customer' in st.session_state.data_model and 'mapping_load_details' in st.session_state.data_model:
            customers_df = st.session_state.data_model['dim_customer']
            loads_df = st.session_state.data_model['mapping_load_details']
            
            # Join to get customer load counts
            customer_loads = loads_df['Customer_ID'].value_counts()
            top_customer_id = customer_loads.index[0]
            top_customer_loads = customer_loads.values[0]
            
            if top_customer_id in customers_df['Customer_ID'].values:
                top_customer_name = customers_df[customers_df['Customer_ID'] == top_customer_id]['Customer_Name'].values[0]
            else:
                top_customer_name = top_customer_id
            
            return f"""**Customer Analysis:**
• Total Customers: {len(customers_df)}
• Top Customer: {top_customer_name} ({top_customer_loads} loads)
• Active Customers: {len(customer_loads)}
• Industries Served: {', '.join(customers_df['Industry'].unique()[:5])}"""
    
    # Default comprehensive analysis
    else:
        return """**Available Analysis Areas:**
        
I can help analyze:
• **Performance**: On-time delivery, carrier ratings, service failures
• **Financial**: Revenue, costs, margins, profitability by lane
• **Customers**: Top customers, industry distribution, credit analysis
• **Carriers**: Performance comparison, cost analysis, selection optimization
• **Consolidation**: Same-day same-lane opportunities
• **Lanes**: Volume analysis, profitability, optimization opportunities

Please ask a specific question about any of these areas!"""

def main():
    """Main application function"""
    
    # Sidebar for data management
    with st.sidebar:
        st.header("⚙️ Data Management")
        
        st.subheader("📊 TMS Data Upload")
        
        # Generate complete sample data
        if st.button("🎲 Generate Complete TMS Sample", type="primary", use_container_width=True):
            with st.spinner("Generating complete TMS dataset..."):
                sample_data = generate_sample_tms_data()
                st.session_state.data_model = sample_data
                st.session_state.primary_data = sample_data.get('mapping_load_details')
                st.success(f"✅ Generated {len(sample_data)} tables with full relationships!")
                st.rerun()
        
        # Multiple file upload
        st.markdown("### 📁 Upload TMS Files")
        uploaded_files = st.file_uploader(
            "Select one or more files",
            type=['csv', 'xlsx'],
            accept_multiple_files=True,
            help="Upload all your TMS data files (loads, shipments, carriers, etc.)"
        )
        
        if uploaded_files:
            for file in uploaded_files:
                df, table_type, confidence = process_uploaded_file(file)
                
                if df is not None:
                    if table_type:
                        st.session_state.data_model[table_type] = df
                        st.success(f"✅ {file.name} → {table_type} ({len(df)} records)")
                        
                        # Set primary data if it's the load table
                        if table_type == 'mapping_load_details':
                            st.session_state.primary_data = df
                    else:
                        st.warning(f"⚠️ Could not identify table type for {file.name}")
                else:
                    st.error(f"❌ Error loading {file.name}")
            
            if st.button("Process Uploaded Data"):
                st.rerun()
        
        # Display loaded tables summary
        if st.session_state.data_model:
            st.markdown("---")
            st.subheader("📋 Loaded Tables")
            
            for table_name, df in st.session_state.data_model.items():
                schema = TMS_DATA_MODEL.get(table_name, {})
                primary_key = schema.get('primary_key', 'N/A')
                
                st.markdown(f"""
                <div class='data-model-card'>
                <strong>{table_name}</strong><br>
                Records: {len(df)}<br>
                Columns: {len(df.columns)}<br>
                Key: {primary_key}
                </div>
                """, unsafe_allow_html=True)
            
            # Clear data button
            if st.button("🗑️ Clear All Data"):
                st.session_state.data_model = {}
                st.session_state.primary_data = None
                st.session_state.chat_history = []
                st.rerun()
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### 📚 TMS Data Model
        
        **Core Tables:**
        - Load Details (Primary)
        - Shipment Details
        - Carrier Rates
        - Performance Metrics
        - Financial Data
        
        **Version 3.0** - Full TMS Support
        """)
    
    # Main content area
    tabs = st.tabs([
        "📊 Data Model Overview",
        "🔄 Integrated Analysis",
        "🎯 Route Optimizer",
        "🤖 AI Assistant",
        "📈 Dashboard"
    ])
    
    with tabs[0]:
        display_data_model_overview()
    
    with tabs[1]:
        display_integrated_analysis()
    
    with tabs[2]:
        # Use the route optimizer from before
        if st.session_state.primary_data is not None:
            # Set the data for compatibility
            st.session_state.data = st.session_state.primary_data
        display_route_optimizer()
    
    with tabs[3]:
        display_ai_assistant_enhanced()
    
    with tabs[4]:
        if st.session_state.primary_data is not None:
            st.session_state.data = st.session_state.primary_data
            display_analytics()
        else:
            st.info("Load the mapping_load_details table to see the dashboard")

if __name__ == "__main__":
    main()
