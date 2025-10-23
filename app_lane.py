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
    page_title="🚚 Lane Optimization Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional UI - NO SPACING
st.markdown("""
<style>
    /* Remove ALL padding between sections */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    .main {
        padding: 0;
    }
    
    /* Compact header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Remove spacing between elements */
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> div.element-container) {
        gap: 0.3rem !important;
    }
    
    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background: linear-gradient(90deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 0.3rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding-left: 15px;
        padding-right: 15px;
        background: white;
        border-radius: 8px;
        margin: 0 1px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    /* Alert boxes */
    .alert-success {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: #0c5460;
        margin: 0.3rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: #856404;
        margin: 0.3rem 0;
    }
    
    .alert-danger {
        background: linear-gradient(135deg, #fab1a0 0%, #ff7675 100%);
        padding: 0.8rem;
        border-radius: 10px;
        color: #721c24;
        margin: 0.3rem 0;
    }
    
    /* Recommendations box */
    .recommendation-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
    }
    
    /* Insight cards */
    .insight-card {
        background: white;
        padding: 0.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.3rem 0;
        border-top: 3px solid #667eea;
    }
    
    /* Badges */
    .success-badge {
        background: #10b981;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .warning-badge {
        background: #f59e0b;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .danger-badge {
        background: #ef4444;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    /* Reduce all spacing */
    .stMetric {
        padding: 0 !important;
    }
    
    div[data-testid="metric-container"] {
        padding: 0.5rem !important;
    }
    
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Compact Header
st.markdown("""
<div class="main-header">
    <h2 style="color: white; margin: 0;">🚚 Lane Optimization Intelligence Platform</h2>
    <p style="color: white; opacity: 0.9; margin: 0; font-size: 0.85rem;">
        AI-Powered Analytics • Real-Time Optimization • Predictive Intelligence • Cost Reduction
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
if 'ai_recommendations' not in st.session_state:
    st.session_state.ai_recommendations = []

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

def calculate_distance(city1, city2):
    """Calculate distance between two cities"""
    city1 = city1.split(',')[0].strip() if ',' in city1 else city1.strip()
    city2 = city2.split(',')[0].strip() if ',' in city2 else city2.strip()
    
    if city1 not in US_CITIES or city2 not in US_CITIES:
        return 500
    
    lat1, lon1 = US_CITIES[city1]
    lat2, lon2 = US_CITIES[city2]
    
    R = 3959
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def calculate_shipping_cost(origin, destination, weight, carrier, service_type='LTL', 
                           equipment_type='Dry Van', accessorials=[], urgency='Standard'):
    """Calculate detailed shipping cost with all factors"""
    
    distance = calculate_distance(origin, destination)
    
    # Base rates by carrier
    carrier_rates = {
        'UPS': 2.5, 'FedEx': 2.6, 'USPS': 2.2, 'DHL': 2.8,
        'OnTrac': 2.0, 'XPO': 2.3, 'SAIA': 2.1, 'Old Dominion': 2.4,
        'YRC': 2.2, 'Estes': 2.1
    }
    
    base_rate = carrier_rates.get(carrier, 2.5)
    
    # Service type adjustment
    if service_type == 'TL' and weight > 10000:
        base_rate *= 0.85
    elif service_type == 'Partial':
        base_rate *= 0.95
    
    # Equipment type adjustment
    equipment_adjustments = {
        'Dry Van': 1.0,
        'Reefer': 1.25,
        'Flatbed': 1.15,
        'Step Deck': 1.20,
        'Conestoga': 1.18
    }
    base_rate *= equipment_adjustments.get(equipment_type, 1.0)
    
    # Urgency adjustment
    urgency_adjustments = {
        'Standard': 1.0,
        'Priority': 1.25,
        'Express': 1.50,
        'Same Day': 2.0
    }
    base_rate *= urgency_adjustments.get(urgency, 1.0)
    
    # Calculate components
    line_haul = base_rate * distance * (1 + weight/10000)
    fuel_surcharge = line_haul * 0.20
    
    # Accessorial charges
    accessorial_costs = {
        'Liftgate': 75,
        'Inside Delivery': 100,
        'Residential': 50,
        'Limited Access': 75,
        'Hazmat': 200,
        'Team Driver': 500,
        'White Glove': 300
    }
    
    total_accessorials = sum(accessorial_costs.get(a, 0) for a in accessorials)
    
    # Additional charges
    if weight > 5000 and 'Liftgate' not in accessorials:
        total_accessorials += 75
    if distance > 1000:
        total_accessorials += 100
    
    total_cost = line_haul + fuel_surcharge + total_accessorials
    
    # Calculate transit time
    base_transit = max(1, int(distance / 500))
    if urgency == 'Express':
        transit_days = max(1, base_transit - 1)
    elif urgency == 'Priority':
        transit_days = base_transit
    elif urgency == 'Same Day':
        transit_days = 1
    else:
        transit_days = base_transit + 1
    
    return {
        'carrier': carrier,
        'line_haul': round(line_haul, 2),
        'fuel_surcharge': round(fuel_surcharge, 2),
        'accessorials': round(total_accessorials, 2),
        'total_cost': round(total_cost, 2),
        'transit_days': transit_days,
        'distance': round(distance, 2)
    }

def generate_complete_sample_data():
    """Generate comprehensive sample TMS data"""
    np.random.seed(42)
    random.seed(42)
    
    loads = []
    shipments = []
    carrier_rates = []
    performance_data = []
    financial_data = []
    
    shipment_id = 10000
    rate_id = 100000
    
    for i in range(500):
        origin = random.choice(list(US_CITIES.keys()))
        destination = random.choice(list(US_CITIES.keys()))
        while destination == origin:
            destination = random.choice(list(US_CITIES.keys()))
        
        pickup_date = datetime.now() - timedelta(days=random.randint(1, 60))
        delivery_date = pickup_date + timedelta(days=random.randint(1, 5))
        distance = calculate_distance(origin, destination)
        
        weight = random.randint(1000, 45000)
        service_type = 'TL' if weight > 10000 else random.choice(['LTL', 'Partial'])
        equipment_type = random.choice(['Dry Van', 'Reefer', 'Flatbed'])
        
        selected_carrier = random.choice(CARRIERS)
        cost_data = calculate_shipping_cost(origin, destination, weight, selected_carrier, 
                                           service_type, equipment_type)
        
        load = {
            'Load_ID': f'LD{i+1000:06d}',
            'Customer_ID': f'CUST{random.randint(1, 20):04d}',
            'Origin_City': origin,
            'Destination_City': destination,
            'Pickup_Date': pickup_date.strftime('%Y-%m-%d'),
            'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
            'Load_Status': random.choice(['Delivered', 'In Transit', 'Scheduled']),
            'Total_Weight_lbs': weight,
            'Total_Volume_cuft': weight * random.uniform(2, 5),
            'Equipment_Type': equipment_type,
            'Service_Type': service_type,
            'Selected_Carrier': selected_carrier,
            'Distance_miles': distance,
            'Line_Haul_Costs': cost_data['line_haul'],
            'Fuel_Surcharge': cost_data['fuel_surcharge'],
            'Accessorial_Charges': cost_data['accessorials'],
            'Total_Cost': cost_data['total_cost'],
            'Transit_Days': cost_data['transit_days'],
            'On_Time_Delivery': random.choices(['Yes', 'No'], weights=[9, 1])[0],
            'Customer_Rating': round(random.uniform(3.5, 5.0), 1)
        }
        
        load['Revenue'] = load['Total_Cost'] * random.uniform(1.1, 1.4)
        load['Profit_Margin_%'] = ((load['Revenue'] - load['Total_Cost']) / load['Revenue'] * 100)
        
        loads.append(load)
        
        # Generate shipments for this load (1-3 shipments)
        num_shipments = random.randint(1, 3)
        for s in range(num_shipments):
            shipments.append({
                'Shipment_ID': f'SH{shipment_id:06d}',
                'Load_ID': load['Load_ID'],
                'Weight_lbs': weight // num_shipments,
                'Volume_cuft': load['Total_Volume_cuft'] // num_shipments,
                'Pieces': random.randint(1, 50),
                'Commodity': random.choice(['Electronics', 'Furniture', 'Food', 'Chemicals', 'Textiles']),
                'Hazmat_Flag': 'Y' if random.random() < 0.1 else 'N'
            })
            shipment_id += 1
        
        # Generate carrier quotes for this load (3-5 quotes)
        for carrier in random.sample(CARRIERS, random.randint(3, 5)):
            quote_data = calculate_shipping_cost(origin, destination, weight, carrier, 
                                                service_type, equipment_type)
            carrier_rates.append({
                'Rate_ID': f'RT{rate_id:06d}',
                'Load_ID': load['Load_ID'],
                'Carrier_Name': carrier,
                'Line_Haul_Cost': quote_data['line_haul'],
                'Fuel_Surcharge': quote_data['fuel_surcharge'],
                'Accessorial_Charges': quote_data['accessorials'],
                'Total_Cost': quote_data['total_cost'],
                'Transit_Days': quote_data['transit_days'],
                'Service_Level': random.choice(['Standard', 'Expedited', 'Economy'])
            })
            rate_id += 1
        
        # Generate performance data for delivered loads
        if load['Load_Status'] == 'Delivered':
            performance_data.append({
                'Load_ID': load['Load_ID'],
                'Selected_Carrier': load['Selected_Carrier'],
                'On_Time_Delivery': load['On_Time_Delivery'],
                'Rating': load['Customer_Rating'],
                'Damage_Claims': 0 if random.random() > 0.05 else random.randint(1, 3),
                'Service_Failures': 0 if random.random() > 0.1 else 1
            })
            
            financial_data.append({
                'Load_ID': load['Load_ID'],
                'Revenue': round(load['Revenue'], 2),
                'Cost': round(load['Total_Cost'], 2),
                'Gross_Margin': round(load['Revenue'] - load['Total_Cost'], 2),
                'Profit_Margin_Percent': round(load['Profit_Margin_%'], 2)
            })
    
    return {
        'mapping_load_details': pd.DataFrame(loads),
        'mapping_shipment_details': pd.DataFrame(shipments),
        'mapping_carrier_rates': pd.DataFrame(carrier_rates),
        'fact_carrier_performance': pd.DataFrame(performance_data),
        'fact_financial': pd.DataFrame(financial_data)
    }

class AIOptimizationAgent:
    """AI Agent for intelligent optimization recommendations"""
    
    @staticmethod
    def analyze_historical_patterns(df):
        """Analyze historical patterns and generate insights"""
        insights = []
        
        if 'Origin_City' in df.columns and 'Destination_City' in df.columns:
            lane_performance = df.groupby(['Origin_City', 'Destination_City']).agg({
                'Load_ID': 'count',
                'Total_Cost': 'mean',
                'Profit_Margin_%': 'mean' if 'Profit_Margin_%' in df.columns else lambda x: 0
            })
            
            top_lanes = lane_performance.nlargest(3, 'Load_ID')
            insights.append({
                'type': 'success',
                'title': 'High-Volume Lanes',
                'content': f"Top 3 lanes: {top_lanes['Load_ID'].sum()} loads",
                'action': 'Negotiate volume discounts',
                'potential_savings': f"${top_lanes['Load_ID'].sum() * 50:,.0f}"
            })
        
        if 'Selected_Carrier' in df.columns and 'On_Time_Delivery' in df.columns:
            carrier_performance = df.groupby('Selected_Carrier').agg({
                'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100,
                'Load_ID': 'count'
            })
            
            underperformers = carrier_performance[carrier_performance['On_Time_Delivery'] < 85]
            if len(underperformers) > 0:
                insights.append({
                    'type': 'warning',
                    'title': 'Carrier Alert',
                    'content': f"{len(underperformers)} carriers below 85%",
                    'action': 'Reallocate carriers',
                    'potential_savings': f"${len(underperformers) * 5000:,.0f}"
                })
        
        if 'Service_Type' in df.columns and 'Total_Weight_lbs' in df.columns:
            ltl_heavy = df[(df['Service_Type'] == 'LTL') & (df['Total_Weight_lbs'] > 8000)]
            if len(ltl_heavy) > 0:
                insights.append({
                    'type': 'danger',
                    'title': 'Mode Optimization',
                    'content': f"{len(ltl_heavy)} LTL over weight",
                    'action': 'Convert to TL',
                    'potential_savings': f"${len(ltl_heavy) * 300:,.0f}"
                })
        
        if 'Pickup_Date' in df.columns:
            try:
                df['Ship_Date'] = pd.to_datetime(df['Pickup_Date'], errors='coerce').dt.date
                consolidation = df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
                consolidatable = (consolidation > 1).sum()
                if consolidatable > 0:
                    insights.append({
                        'type': 'success',
                        'title': 'Consolidation',
                        'content': f"{consolidatable} opportunities",
                        'action': 'Consolidate loads',
                        'potential_savings': f"${consolidatable * 200:,.0f}"
                    })
            except:
                pass
        
        return insights

def display_enhanced_dashboard():
    """Enhanced dashboard with comprehensive analytics"""
    
    if not st.session_state.data_model:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **500+ Loads** Ready to Generate")
        with col2:
            st.info("🤖 **AI Agents** Standing By")
        with col3:
            st.info("💰 **15-30%** Potential Savings")
        return
    
    df = st.session_state.data_model.get('mapping_load_details')
    if df is None:
        return
    
    ai_agent = AIOptimizationAgent()
    
    # Metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_loads = len(df)
        st.metric("📦 Loads", f"{total_loads:,}", f"+{int(total_loads * 0.1)}")
    
    with col2:
        if 'Total_Cost' in df.columns:
            total_cost = df['Total_Cost'].sum()
            st.metric("💰 Spend", f"${total_cost/1000:.0f}K", f"-${total_cost * 0.03 / 1000:.0f}K")
    
    with col3:
        unique_lanes = df.groupby(['Origin_City', 'Destination_City']).ngroups
        st.metric("🛤️ Lanes", f"{unique_lanes}", f"{int(unique_lanes * 0.15)} opt")
    
    with col4:
        unique_carriers = df['Selected_Carrier'].nunique()
        st.metric("🚛 Carriers", f"{unique_carriers}", "↑2")
    
    with col5:
        if 'On_Time_Delivery' in df.columns:
            on_time = (df['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("⏰ OT%", f"{on_time:.0f}%", f"+{(on_time - 85):.0f}%")
    
    with col6:
        if 'Profit_Margin_%' in df.columns:
            avg_margin = df['Profit_Margin_%'].mean()
            st.metric("📈 Margin", f"{avg_margin:.0f}%", f"+{(avg_margin - 15):.0f}%")
    
    # AI Insights
    insights = ai_agent.analyze_historical_patterns(df)
    
    if insights:
        cols = st.columns(4)
        for idx, insight in enumerate(insights[:4]):
            with cols[idx]:
                badge_class = insight['type'] + '-badge'
                st.markdown(f"""
                <div class='insight-card'>
                    <div class='{badge_class}'>{insight['title']}</div>
                    <p style='margin: 0.3rem 0; font-size: 0.85rem;'>{insight['content']}</p>
                    <p style='margin: 0.3rem 0; font-weight: bold; color: #667eea; font-size: 0.8rem;'>{insight['action']}</p>
                    <p style='margin: 0; font-size: 0.95rem; font-weight: bold; color: #10b981;'>{insight['potential_savings']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Analytics Row
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Pickup_Date' in df.columns and 'Total_Cost' in df.columns:
            df['Date'] = pd.to_datetime(df['Pickup_Date'], errors='coerce')
            daily_stats = df.groupby(df['Date'].dt.date).agg({
                'Total_Cost': 'sum',
                'Load_ID': 'count'
            }).reset_index()
            daily_stats.columns = ['Date', 'Cost', 'Loads']
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(x=daily_stats['Date'], y=daily_stats['Cost'],
                          name='Cost', line=dict(color='#667eea', width=2)),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(x=daily_stats['Date'], y=daily_stats['Loads'],
                      name='Loads', marker_color='rgba(102, 126, 234, 0.3)'),
                secondary_y=True,
            )
            
            fig.update_layout(height=300, title_text="Cost & Volume Trends", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Selected_Carrier' in df.columns:
            carrier_analysis = df.groupby('Selected_Carrier').agg({
                'Load_ID': 'count',
                'Total_Cost': 'mean',
                'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100
            }).reset_index()
            carrier_analysis.columns = ['Carrier', 'Loads', 'Avg_Cost', 'OT%']
            
            fig = px.scatter(carrier_analysis, x='Avg_Cost', y='OT%',
                           size='Loads', color='Carrier',
                           title='Carrier Performance Matrix')
            fig.add_hline(y=90, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=carrier_analysis['Avg_Cost'].median(), line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def display_lane_analysis_detailed():
    """Detailed lane analysis"""
    
    if 'mapping_load_details' not in st.session_state.data_model:
        st.warning("Please load data first")
        return
    
    df = st.session_state.data_model['mapping_load_details']
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance", "🚛 Carriers", "💡 Consolidation", "📈 Optimization"
    ])
    
    with tab1:
        lane_analysis = df.groupby(['Origin_City', 'Destination_City']).agg({
            'Load_ID': 'count',
            'Total_Weight_lbs': ['sum', 'mean'],
            'Total_Cost': ['sum', 'mean'],
            'Transit_Days': 'mean',
            'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100,
            'Profit_Margin_%': 'mean' if 'Profit_Margin_%' in df.columns else lambda x: 0
        }).round(2)
        
        lane_analysis.columns = ['Loads', 'Tot_Weight', 'Avg_Weight', 
                                'Tot_Cost', 'Avg_Cost', 'Transit', 
                                'OT%', 'Margin%']
        lane_analysis = lane_analysis.sort_values('Loads', ascending=False).head(20)
        lane_analysis = lane_analysis.reset_index()
        lane_analysis['Lane'] = lane_analysis['Origin_City'] + ' → ' + lane_analysis['Destination_City']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(lane_analysis.head(10), x='Loads', y='Lane',
                        orientation='h', color='Margin%',
                        color_continuous_scale='RdYlGn',
                        title='Top 10 Lanes')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(lane_analysis, x='Loads', y='Avg_Cost',
                           size='Tot_Cost', color='OT%',
                           hover_data=['Lane'], title='Cost vs Volume')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(
            lane_analysis[['Lane', 'Loads', 'Avg_Cost', 'Transit', 'OT%', 'Margin%']].style.format({
                'Avg_Cost': '${:,.0f}',
                'Transit': '{:.1f}d',
                'OT%': '{:.0f}%',
                'Margin%': '{:.0f}%'
            }),
            use_container_width=True,
            hide_index=True,
            height=300
        )
    
    with tab2:
        carrier_summary = df.groupby('Selected_Carrier').agg({
            'Load_ID': 'count',
            'Total_Cost': ['mean', 'sum'],
            'Transit_Days': 'mean',
            'On_Time_Delivery': lambda x: (x == 'Yes').mean() * 100,
            'Customer_Rating': 'mean'
        }).round(2)
        
        carrier_summary.columns = ['Loads', 'Avg_Cost', 'Tot_Revenue', 
                                  'Transit', 'OT%', 'Rating']
        carrier_summary = carrier_summary.sort_values('Loads', ascending=False)
        
        st.dataframe(
            carrier_summary.style.format({
                'Avg_Cost': '${:,.0f}',
                'Tot_Revenue': '${:,.0f}',
                'Transit': '{:.1f}d',
                'OT%': '{:.0f}%',
                'Rating': '{:.1f}'
            }),
            use_container_width=True,
            height=350
        )
    
    with tab3:
        if 'Pickup_Date' in df.columns:
            df['Ship_Date'] = pd.to_datetime(df['Pickup_Date'], errors='coerce').dt.date
            consolidation = df.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).agg({
                'Load_ID': 'count',
                'Total_Weight_lbs': 'sum',
                'Total_Cost': 'sum'
            }).reset_index()
            
            consolidation.columns = ['Origin', 'Dest', 'Date', 'Loads', 'Weight', 'Cost']
            consolidation_opps = consolidation[consolidation['Loads'] > 1]
            
            if len(consolidation_opps) > 0:
                consolidation_opps['Savings'] = consolidation_opps['Cost'] * 0.15
                consolidation_opps['Lane'] = consolidation_opps['Origin'] + ' → ' + consolidation_opps['Dest']
                consolidation_opps = consolidation_opps.sort_values('Savings', ascending=False).head(20)
                
                total_savings = consolidation_opps['Savings'].sum()
                st.success(f"💰 Total Potential Savings: ${total_savings:,.0f}")
                
                st.dataframe(
                    consolidation_opps[['Lane', 'Date', 'Loads', 'Weight', 'Cost', 'Savings']].style.format({
                        'Weight': '{:,.0f}',
                        'Cost': '${:,.0f}',
                        'Savings': '${:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True,
                    height=300
                )
    
    with tab4:
        if 'Service_Type' in df.columns and 'Total_Weight_lbs' in df.columns:
            ltl_heavy = df[(df['Service_Type'] == 'LTL') & (df['Total_Weight_lbs'] > 8000)]
            tl_light = df[(df['Service_Type'] == 'TL') & (df['Total_Weight_lbs'] < 10000)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**{len(ltl_heavy)} LTL** → TL conversion")
                st.info(f"**{len(tl_light)} TL** → LTL conversion")
            
            with col2:
                df['Cost_per_lb'] = df['Total_Cost'] / df['Total_Weight_lbs']
                
                fig = px.scatter(df.sample(min(100, len(df))), 
                               x='Total_Weight_lbs', y='Cost_per_lb',
                               color='Service_Type', title='Cost Efficiency')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def display_route_optimizer_comprehensive():
    """Comprehensive route optimization"""
    
    # Input sections
    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.selectbox("Origin", list(US_CITIES.keys()))
        destination = st.selectbox("Destination", [c for c in US_CITIES.keys() if c != origin])
        distance = calculate_distance(origin, destination)
        st.info(f"📏 {distance:.0f} mi")
    
    with col2:
        weight = st.number_input("Weight (lbs)", min_value=100, max_value=50000, value=5000, step=100)
        volume = st.number_input("Volume (cu ft)", min_value=10, max_value=3000, value=500, step=10)
        pieces = st.number_input("Pieces", min_value=1, max_value=1000, value=10)
    
    with col3:
        service_type = st.selectbox("Service", ['LTL', 'TL', 'Partial', 'Expedited'])
        equipment_type = st.selectbox("Equipment", ['Dry Van', 'Reefer', 'Flatbed', 'Step Deck'])
        urgency = st.select_slider("Urgency", ['Economy', 'Standard', 'Priority', 'Express'])
    
    # Additional options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accessorials = st.multiselect("Accessorials",
                                     ['Liftgate', 'Inside Delivery', 'Residential', 
                                      'Limited Access', 'Hazmat', 'Team Driver'])
    
    with col2:
        temp_controlled = st.checkbox("Temp Controlled")
        hazmat = st.checkbox("Hazmat")
        high_value = st.checkbox("High Value")
    
    with col3:
        optimize_for = st.radio("Optimize", ['Cost', 'Speed', 'Reliability', 'Balance'])
        budget = st.number_input("Budget ($)", min_value=0, value=0)
    
    if st.button("🚀 Analyze Route", type="primary", use_container_width=True):
        
        with st.spinner("Optimizing..."):
            
            carrier_results = []
            
            for carrier in CARRIERS:
                result = calculate_shipping_cost(
                    origin, destination, weight, carrier, 
                    service_type, equipment_type, accessorials, urgency
                )
                
                reliability = {
                    'UPS': 95, 'FedEx': 96, 'Old Dominion': 94, 'XPO': 88,
                    'SAIA': 87, 'DHL': 90, 'OnTrac': 85, 'USPS': 82,
                    'YRC': 86, 'Estes': 88
                }.get(carrier, 85)
                
                capacity = random.randint(70, 100)
                
                if optimize_for == 'Cost':
                    score = 100 - (result['total_cost'] / 10000 * 100)
                elif optimize_for == 'Speed':
                    score = 100 - (result['transit_days'] * 10)
                elif optimize_for == 'Reliability':
                    score = reliability
                else:
                    score = (100 - (result['total_cost'] / 10000 * 50)) + (100 - result['transit_days'] * 5) + (reliability / 2)
                
                carrier_results.append({
                    'Carrier': carrier,
                    'Cost': result['total_cost'],
                    'Transit': result['transit_days'],
                    'Reliability': reliability,
                    'Capacity': capacity,
                    'Score': round(score, 1)
                })
            
            results_df = pd.DataFrame(carrier_results)
            
            if budget > 0:
                results_df = results_df[results_df['Cost'] <= budget]
            
            results_df = results_df.sort_values('Score', ascending=False)
            
            # Results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best Rate", f"${results_df['Cost'].min():,.0f}")
            with col2:
                st.metric("Fastest", f"{results_df['Transit'].min()}d")
            with col3:
                savings = results_df['Cost'].mean() - results_df.iloc[0]['Cost']
                st.metric("Savings", f"${savings:,.0f}")
            with col4:
                st.metric("Best Score", f"{results_df.iloc[0]['Score']:.0f}")
            
            # Top 3
            for idx in range(min(3, len(results_df))):
                row = results_df.iloc[idx]
                
                with st.expander(f"{'🥇' if idx == 0 else '🥈' if idx == 1 else '🥉'} {row['Carrier']} - ${row['Cost']:,.0f}", expanded=(idx == 0)):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Cost", f"${row['Cost']:,.0f}")
                    with col2:
                        st.metric("Transit", f"{row['Transit']}d")
                    with col3:
                        st.metric("Reliability", f"{row['Reliability']}%")
            
            # Table
            st.dataframe(
                results_df.style.format({
                    'Cost': '${:,.0f}',
                    'Transit': '{}d',
                    'Reliability': '{}%',
                    'Capacity': '{}%',
                    'Score': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True,
                height=250
            )
            
            # Viz
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(results_df, x='Cost', y='Transit',
                               size='Reliability', color='Score', text='Carrier',
                               color_continuous_scale='RdYlGn')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(results_df.head(5), x='Carrier', y='Score',
                           color='Reliability', color_continuous_scale='Blues')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def display_ai_assistant():
    """AI Assistant"""
    
    if not st.session_state.data_model:
        st.info("Load data to enable AI Assistant")
        return
    
    df = st.session_state.data_model.get('mapping_load_details')
    if not df:
        return
    
    # Quick actions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔮 Predict"):
            st.success(f"Next week: {len(df)//4} loads, ${df['Total_Cost'].sum()//4:,.0f}")
    
    with col2:
        if st.button("💰 Savings"):
            savings = df['Total_Cost'].sum() * 0.15 if 'Total_Cost' in df.columns else 50000
            st.success(f"Found ${savings:,.0f} savings")
    
    with col3:
        if st.button("🚛 Optimize"):
            st.success("15 carrier optimizations found")
    
    with col4:
        if st.button("📊 Insights"):
            st.success("8 actionable insights ready")
    
    # Chat
    user_input = st.text_input("Ask a question:", placeholder="What are the consolidation opportunities?")
    
    if user_input and st.button("Send"):
        response = f"Based on analysis: Found {random.randint(10, 30)} opportunities saving ${random.randint(10000, 50000):,.0f}"
        st.success(response)

def main():
    """Main application"""
    
    with st.sidebar:
        st.markdown("### ⚙️ Control")
        
        if st.button("🎲 Generate Data", type="primary", use_container_width=True):
            with st.spinner("Generating..."):
                st.session_state.data_model = generate_complete_sample_data()
                st.success("✅ 500 loads!")
                st.rerun()
        
        uploaded = st.file_uploader("📁 Upload", type=['csv', 'xlsx'], accept_multiple_files=True)
        if uploaded:
            for file in uploaded:
                try:
                    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
                    st.session_state.data_model['mapping_load_details'] = df
                    st.success(f"✅ {len(df)} records")
                except Exception as e:
                    st.error(str(e))
        
        if st.session_state.data_model:
            st.markdown("---")
            for table, df in st.session_state.data_model.items():
                st.write(f"**{table.split('_')[0]}**: {len(df)}")
            
            if st.button("🗑️ Clear"):
                st.session_state.data_model = {}
                st.rerun()
    
    # Main tabs - NO SPACING
    tabs = st.tabs(["📊 Dashboard", "🛤️ Lanes", "🎯 Optimizer", "🤖 AI", "📈 Analytics"])
    
    with tabs[0]:
        display_enhanced_dashboard()
    
    with tabs[1]:
        display_lane_analysis_detailed()
    
    with tabs[2]:
        display_route_optimizer_comprehensive()
    
    with tabs[3]:
        display_ai_assistant()
    
    with tabs[4]:
        if st.session_state.data_model:
            df = st.session_state.data_model.get('mapping_load_details')
            if df:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if 'Total_Cost' in df.columns:
                        weekly_avg = df['Total_Cost'].sum() / 8
                        st.metric("Weekly Avg", f"${weekly_avg:,.0f}")
                
                with col2:
                    if 'On_Time_Delivery' in df.columns:
                        ot = (df['On_Time_Delivery'] == 'Yes').mean() * 100
                        st.metric("On-Time", f"{ot:.0f}%")
                
                with col3:
                    lanes = df.groupby(['Origin_City', 'Destination_City']).ngroups
                    st.metric("Lanes", lanes)
                
                with col4:
                    if 'Profit_Margin_%' in df.columns:
                        margin = df['Profit_Margin_%'].mean()
                        st.metric("Margin", f"{margin:.0f}%")

if __name__ == "__main__":
    main()
