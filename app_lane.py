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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
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
        AI-Powered Multi-Carrier Cost Analysis & Route Optimization
    </p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

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

CARRIERS = ['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac', 'XPO', 'SAIA', 'Old Dominion']

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

def generate_sample_data(num_loads=500):
    """Generate realistic sample transportation data"""
    
    np.random.seed(42)
    random.seed(42)
    
    data = []
    load_id = 1000
    
    for _ in range(num_loads):
        origin = random.choice(list(US_CITIES.keys()))
        destination = random.choice(list(US_CITIES.keys()))
        while destination == origin:
            destination = random.choice(list(US_CITIES.keys()))
        
        distance = calculate_distance(origin, destination)
        
        # Generate shipment details
        total_weight = random.randint(100, 25000)
        
        # Select carrier and calculate costs
        selected_carrier = random.choice(CARRIERS)
        base_rate = random.uniform(1.5, 4.0)
        
        # Calculate costs with various factors
        line_haul_cost = base_rate * distance * (1 + total_weight/10000)
        fuel_surcharge = line_haul_cost * random.uniform(0.15, 0.25)
        accessorial = random.uniform(50, 200) if random.random() > 0.7 else 0
        
        total_cost = line_haul_cost + fuel_surcharge + accessorial
        revenue = total_cost * random.uniform(1.1, 1.4)
        profit_margin = ((revenue - total_cost) / revenue) * 100
        
        transit_days = max(1, int(distance / 500) + random.randint(0, 2))
        
        pickup_date = datetime.now() - timedelta(days=random.randint(1, 30))
        delivery_date = pickup_date + timedelta(days=transit_days)
        
        data.append({
            'Load_ID': f'LD{load_id:04d}',
            'Origin_City': origin,
            'Destination_City': destination,
            'Selected_Carrier': selected_carrier,
            'Pickup_Date': pickup_date.strftime('%Y-%m-%d'),
            'Delivery_Date': delivery_date.strftime('%Y-%m-%d'),
            'Total_Weight_lbs': total_weight,
            'Distance_miles': round(distance, 2),
            'Line_Haul_Costs': round(line_haul_cost, 2),
            'Fuel_Surcharge': round(fuel_surcharge, 2),
            'Accessorial_Charges': round(accessorial, 2),
            'Total_Cost': round(total_cost, 2),
            'Revenue': round(revenue, 2),
            'Profit_Margin_%': round(profit_margin, 2),
            'Transit_Days': transit_days,
            'Service_Type': 'TL' if total_weight > 10000 else 'LTL',
            'On_Time_Delivery': 'Yes' if random.random() > 0.1 else 'No'
        })
        
        load_id += 1
    
    return pd.DataFrame(data)

def calculate_shipping_cost(origin, destination, weight, carrier, service_type='LTL'):
    """Calculate detailed shipping cost"""
    
    distance = calculate_distance(origin, destination)
    
    # Base rates by carrier
    carrier_rates = {
        'UPS': 2.5,
        'FedEx': 2.6,
        'USPS': 2.2,
        'DHL': 2.8,
        'OnTrac': 2.0,
        'XPO': 2.3,
        'SAIA': 2.1,
        'Old Dominion': 2.4
    }
    
    base_rate = carrier_rates.get(carrier, 2.5)
    
    # Service type adjustment
    if service_type == 'TL' and weight > 10000:
        base_rate *= 0.85
    
    # Calculate components
    line_haul = base_rate * distance * (1 + weight/10000)
    fuel_surcharge = line_haul * 0.20
    
    # Accessorials
    accessorials = 0
    if weight > 5000:
        accessorials += 75  # Liftgate
    if distance > 1000:
        accessorials += 100  # Long haul
    
    total_cost = line_haul + fuel_surcharge + accessorials
    
    return {
        'line_haul': round(line_haul, 2),
        'fuel_surcharge': round(fuel_surcharge, 2),
        'accessorials': round(accessorials, 2),
        'total_cost': round(total_cost, 2)
    }

def display_data_overview():
    """Display overview of uploaded data - FIRST TAB"""
    
    if st.session_state.data is None:
        st.info("📤 Please upload your data or generate sample data using the sidebar")
        
        # Quick start guide
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 🚀 Quick Start Guide
            1. Click **'Generate Sample Data'** in the sidebar for demo
            2. Or upload your CSV/Excel file
            3. Explore the analysis tabs
            4. Use the AI Assistant for insights
            """)
        with col2:
            st.markdown("""
            ### 📊 Expected Data Format
            - Load_ID
            - Origin_City
            - Destination_City
            - Selected_Carrier
            - Total_Weight_lbs
            - Total_Cost
            - Transit_Days
            """)
        return
    
    data = st.session_state.data
    st.success(f"✅ Data loaded: {len(data)} records")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Loads", f"{len(data):,}")
    
    with col2:
        total_cost = data['Total_Cost'].sum() if 'Total_Cost' in data.columns else 0
        st.metric("Total Cost", f"${total_cost:,.0f}")
    
    with col3:
        unique_lanes = data.groupby(['Origin_City', 'Destination_City']).ngroups if 'Origin_City' in data.columns else 0
        st.metric("Unique Lanes", f"{unique_lanes:,}")
    
    with col4:
        avg_weight = data['Total_Weight_lbs'].mean() if 'Total_Weight_lbs' in data.columns else 0
        st.metric("Avg Weight", f"{avg_weight:,.0f} lbs")
    
    # Data preview and stats
    tab1, tab2, tab3 = st.tabs(["📋 Data Preview", "📊 Statistics", "📈 Distributions"])
    
    with tab1:
        st.dataframe(data.head(100), use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Numerical Statistics")
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(data[numeric_cols].describe(), use_container_width=True)
        
        with col2:
            st.markdown("### Categorical Statistics")
            categorical_cols = data.select_dtypes(include=['object']).columns
            stats_data = []
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                stats_data.append({
                    'Column': col,
                    'Unique Values': data[col].nunique(),
                    'Most Common': data[col].mode()[0] if not data[col].mode().empty else 'N/A',
                    'Missing': data[col].isna().sum()
                })
            if stats_data:
                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    with tab3:
        if 'Total_Cost' in data.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(data, x='Total_Cost', nbins=30, 
                                 title='Cost Distribution')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Selected_Carrier' in data.columns:
                    carrier_counts = data['Selected_Carrier'].value_counts()
                    fig = px.pie(values=carrier_counts.values, 
                               names=carrier_counts.index,
                               title='Carrier Distribution')
                    st.plotly_chart(fig, use_container_width=True)
    
    # Download button
    csv = data.to_csv(index=False)
    st.download_button(
        label="📥 Download Processed Data",
        data=csv,
        file_name=f"lane_optimization_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

def display_lane_analysis():
    """Display lane analysis and consolidation opportunities"""
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    data = st.session_state.data
    
    # Check required columns
    required_cols = ['Origin_City', 'Destination_City', 'Load_ID']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🛤️ Top Lanes", "🚛 Carriers", "💡 Consolidation"])
    
    with tab1:
        st.markdown("### Top Lanes by Volume")
        
        lane_stats = data.groupby(['Origin_City', 'Destination_City']).agg({
            'Load_ID': 'count'
        }).reset_index()
        lane_stats.columns = ['Origin', 'Destination', 'Load_Count']
        
        # Add optional columns if they exist
        if 'Total_Weight_lbs' in data.columns:
            weight_stats = data.groupby(['Origin_City', 'Destination_City'])['Total_Weight_lbs'].sum().reset_index()
            lane_stats = lane_stats.merge(weight_stats, on=['Origin_City', 'Destination_City'], how='left')
        
        if 'Total_Cost' in data.columns:
            cost_stats = data.groupby(['Origin_City', 'Destination_City'])['Total_Cost'].mean().reset_index()
            cost_stats.columns = ['Origin', 'Destination', 'Avg_Cost']
            lane_stats = lane_stats.merge(cost_stats, on=['Origin', 'Destination'], how='left')
        
        lane_stats['Lane'] = lane_stats['Origin'] + ' → ' + lane_stats['Destination']
        lane_stats = lane_stats.sort_values('Load_Count', ascending=False).head(10)
        
        # Display table
        display_cols = ['Lane', 'Load_Count']
        if 'Total_Weight_lbs' in lane_stats.columns:
            display_cols.append('Total_Weight_lbs')
        if 'Avg_Cost' in lane_stats.columns:
            display_cols.append('Avg_Cost')
        
        st.dataframe(lane_stats[display_cols], use_container_width=True, hide_index=True)
        
        # Simple bar chart
        fig = px.bar(lane_stats, x='Lane', y='Load_Count', title='Top 10 Lanes by Volume')
        fig.update_layout(xaxis={'tickangle': -45})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if 'Selected_Carrier' in data.columns:
            st.markdown("### Carrier Performance")
            
            carrier_stats = data.groupby('Selected_Carrier').agg({
                'Load_ID': 'count'
            }).reset_index()
            carrier_stats.columns = ['Carrier', 'Total_Loads']
            
            # Add cost stats if available
            if 'Total_Cost' in data.columns:
                cost_by_carrier = data.groupby('Selected_Carrier')['Total_Cost'].mean().reset_index()
                cost_by_carrier.columns = ['Carrier', 'Avg_Cost']
                carrier_stats = carrier_stats.merge(cost_by_carrier, on='Carrier', how='left')
            
            carrier_stats = carrier_stats.sort_values('Total_Loads', ascending=False)
            
            st.dataframe(carrier_stats, use_container_width=True, hide_index=True)
            
            # Visualization
            fig = px.bar(carrier_stats, x='Carrier', y='Total_Loads', 
                        title='Loads by Carrier')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Carrier information not available in the data")
    
    with tab3:
        st.markdown("### Consolidation Opportunities")
        
        if 'Pickup_Date' in data.columns:
            # Convert to datetime safely
            try:
                data['Ship_Date'] = pd.to_datetime(data['Pickup_Date'], errors='coerce').dt.date
                
                consolidation = data.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).agg({
                    'Load_ID': 'count'
                }).reset_index()
                consolidation.columns = ['Origin', 'Destination', 'Date', 'Load_Count']
                
                # Filter for multiple loads
                consolidation_opp = consolidation[consolidation['Load_Count'] > 1]
                
                if not consolidation_opp.empty:
                    consolidation_opp['Lane'] = consolidation_opp['Origin'] + ' → ' + consolidation_opp['Destination']
                    consolidation_opp = consolidation_opp.sort_values('Load_Count', ascending=False).head(10)
                    
                    st.dataframe(
                        consolidation_opp[['Lane', 'Date', 'Load_Count']],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    total_opportunities = consolidation_opp['Load_Count'].sum()
                    st.success(f"💡 Found {len(consolidation_opp)} lanes with consolidation potential ({total_opportunities} total loads)")
                else:
                    st.info("No immediate consolidation opportunities found")
            except Exception as e:
                st.warning("Unable to analyze consolidation: Date format issue")
        else:
            st.info("Date information required for consolidation analysis")

def display_route_optimizer():
    """Interactive route optimization tool"""
    
    st.subheader("🎯 Route Optimization Tool")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        origin = st.selectbox("Origin City", list(US_CITIES.keys()), index=0)
        weight = st.number_input("Weight (lbs)", min_value=100, max_value=50000, value=5000, step=100)
    
    with col2:
        destination = st.selectbox("Destination City", list(US_CITIES.keys()), index=1)
        service_type = st.radio("Service Type", ['LTL', 'TL'])
    
    with col3:
        urgency = st.select_slider("Urgency", options=['Standard', 'Priority', 'Express'], value='Standard')
        optimize_for = st.radio("Optimize For", ['Cost', 'Speed', 'Balance'])
    
    if st.button("🚀 Analyze Route", type="primary", use_container_width=True):
        with st.spinner("Analyzing carriers..."):
            
            distance = calculate_distance(origin, destination)
            
            # Analyze all carriers
            results = []
            for carrier in CARRIERS:
                cost_data = calculate_shipping_cost(origin, destination, weight, carrier, service_type)
                
                # Calculate transit time
                base_transit = max(1, int(distance / 500))
                if urgency == 'Express':
                    transit = base_transit
                elif urgency == 'Priority':
                    transit = base_transit + 1
                else:
                    transit = base_transit + 2
                
                results.append({
                    'Carrier': carrier,
                    'Total_Cost': cost_data['total_cost'],
                    'Transit_Days': transit,
                    'Line_Haul': cost_data['line_haul'],
                    'Fuel': cost_data['fuel_surcharge'],
                    'Accessorials': cost_data['accessorials']
                })
            
            results_df = pd.DataFrame(results)
            
            # Sort based on optimization preference
            if optimize_for == 'Cost':
                results_df = results_df.sort_values('Total_Cost')
            elif optimize_for == 'Speed':
                results_df = results_df.sort_values('Transit_Days')
            else:  # Balance
                results_df['Score'] = results_df['Total_Cost'] / results_df['Total_Cost'].max() + \
                                     results_df['Transit_Days'] / results_df['Transit_Days'].max()
                results_df = results_df.sort_values('Score')
            
            # Display results
            st.markdown("### 📊 Carrier Comparison")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.metric("Distance", f"{distance:.0f} miles")
            with col2:
                st.metric("Best Rate", f"${results_df.iloc[0]['Total_Cost']:,.2f}")
            with col3:
                st.metric("Fastest", f"{results_df['Transit_Days'].min()} days")
            
            # Top 3 recommendations
            st.markdown("### 🏆 Top Recommendations")
            
            for idx, row in results_df.head(3).iterrows():
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        rank = ['🥇', '🥈', '🥉'][list(results_df.head(3).index).index(idx)]
                        st.write(f"{rank} **{row['Carrier']}**")
                    with col2:
                        st.write(f"${row['Total_Cost']:,.2f}")
                    with col3:
                        st.write(f"{row['Transit_Days']} days")
            
            # Full comparison
            st.markdown("### 📋 Full Analysis")
            st.dataframe(
                results_df.style.format({
                    'Total_Cost': '${:,.2f}',
                    'Line_Haul': '${:,.2f}',
                    'Fuel': '${:,.2f}',
                    'Accessorials': '${:,.2f}',
                    'Transit_Days': '{:.0f} days'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Visualization
            fig = px.scatter(results_df, x='Total_Cost', y='Transit_Days',
                           text='Carrier', title='Cost vs Transit Time',
                           labels={'Total_Cost': 'Cost ($)', 'Transit_Days': 'Transit (days)'})
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

def display_ai_assistant():
    """AI Assistant for answering questions about the data"""
    
    st.subheader("🤖 AI Assistant")
    st.markdown("Ask questions about your transportation data and get insights!")
    
    # Quick questions
    if st.session_state.data is not None:
        st.markdown("### 💡 Quick Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 What are my top lanes?"):
                if 'Origin_City' in st.session_state.data.columns:
                    top_lanes = st.session_state.data.groupby(['Origin_City', 'Destination_City']).size().nlargest(5)
                    response = "**Top 5 Lanes:**\n"
                    for (origin, dest), count in top_lanes.items():
                        response += f"- {origin} → {dest}: {count} loads\n"
                    st.session_state.chat_history.append(("What are my top lanes?", response))
        
        with col2:
            if st.button("💰 What's my total spend?"):
                if 'Total_Cost' in st.session_state.data.columns:
                    total = st.session_state.data['Total_Cost'].sum()
                    avg = st.session_state.data['Total_Cost'].mean()
                    response = f"**Cost Analysis:**\n- Total Spend: ${total:,.2f}\n- Average per Load: ${avg:,.2f}"
                    st.session_state.chat_history.append(("What's my total spend?", response))
        
        with col3:
            if st.button("🚛 Best performing carrier?"):
                if 'Selected_Carrier' in st.session_state.data.columns:
                    carrier_stats = st.session_state.data.groupby('Selected_Carrier')['Load_ID'].count().nlargest(1)
                    if not carrier_stats.empty:
                        best_carrier = carrier_stats.index[0]
                        count = carrier_stats.values[0]
                        response = f"**Best Performing Carrier:**\n{best_carrier} with {count} loads"
                        st.session_state.chat_history.append(("Best performing carrier?", response))
    
    # Chat interface
    st.markdown("### 💬 Ask Your Question")
    
    user_question = st.text_input("Type your question here...", placeholder="e.g., What are the consolidation opportunities?")
    
    if st.button("Send", type="primary") and user_question:
        # Simulate AI response based on question keywords
        response = analyze_question(user_question, st.session_state.data)
        st.session_state.chat_history.append((user_question, response))
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        for question, answer in st.session_state.chat_history[-5:]:  # Show last 5 Q&As
            st.markdown(f"**You:** {question}")
            st.markdown(f"<div class='assistant-message'>{answer}</div>", unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def analyze_question(question, data):
    """Analyze user question and provide relevant response"""
    
    if data is None:
        return "Please upload data first to get insights."
    
    question_lower = question.lower()
    
    # Cost related questions
    if any(word in question_lower for word in ['cost', 'spend', 'expensive', 'price']):
        if 'Total_Cost' in data.columns:
            total_cost = data['Total_Cost'].sum()
            avg_cost = data['Total_Cost'].mean()
            max_cost = data['Total_Cost'].max()
            min_cost = data['Total_Cost'].min()
            return f"""**Cost Analysis:**
- Total Spend: ${total_cost:,.2f}
- Average Cost: ${avg_cost:,.2f}
- Highest Cost: ${max_cost:,.2f}
- Lowest Cost: ${min_cost:,.2f}
- Cost Range: ${max_cost - min_cost:,.2f}"""
    
    # Lane related questions
    elif any(word in question_lower for word in ['lane', 'route', 'origin', 'destination']):
        if 'Origin_City' in data.columns and 'Destination_City' in data.columns:
            top_lanes = data.groupby(['Origin_City', 'Destination_City']).size().nlargest(5)
            response = "**Top 5 Lanes by Volume:**\n"
            for (origin, dest), count in top_lanes.items():
                response += f"- {origin} → {dest}: {count} loads\n"
            
            unique_lanes = data.groupby(['Origin_City', 'Destination_City']).ngroups
            response += f"\nTotal Unique Lanes: {unique_lanes}"
            return response
    
    # Carrier related questions
    elif any(word in question_lower for word in ['carrier', 'vendor', 'provider']):
        if 'Selected_Carrier' in data.columns:
            carrier_counts = data['Selected_Carrier'].value_counts()
            response = "**Carrier Distribution:**\n"
            for carrier, count in carrier_counts.head(5).items():
                percentage = (count / len(data)) * 100
                response += f"- {carrier}: {count} loads ({percentage:.1f}%)\n"
            return response
    
    # Consolidation questions
    elif any(word in question_lower for word in ['consolidat', 'combine', 'merge']):
        if 'Origin_City' in data.columns and 'Pickup_Date' in data.columns:
            try:
                data_temp = data.copy()
                data_temp['Ship_Date'] = pd.to_datetime(data_temp['Pickup_Date'], errors='coerce').dt.date
                same_lane_same_day = data_temp.groupby(['Origin_City', 'Destination_City', 'Ship_Date']).size()
                consolidation_opportunities = same_lane_same_day[same_lane_same_day > 1].sum()
                unique_opportunities = len(same_lane_same_day[same_lane_same_day > 1])
                
                return f"""**Consolidation Analysis:**
- Potential Consolidation Loads: {consolidation_opportunities}
- Unique Opportunities: {unique_opportunities} lane-date combinations
- Estimated Savings: ${consolidation_opportunities * 150:.2f} (at $150 per consolidation)
- Recommendation: Focus on high-volume lanes for maximum impact"""
            except:
                return "Unable to analyze consolidation opportunities due to data format issues."
    
    # Weight analysis
    elif any(word in question_lower for word in ['weight', 'heavy', 'volume']):
        if 'Total_Weight_lbs' in data.columns:
            total_weight = data['Total_Weight_lbs'].sum()
            avg_weight = data['Total_Weight_lbs'].mean()
            
            # Categorize shipments
            ltl_count = len(data[data['Total_Weight_lbs'] < 10000])
            tl_count = len(data[data['Total_Weight_lbs'] >= 10000])
            
            return f"""**Weight Analysis:**
- Total Weight: {total_weight:,.0f} lbs
- Average Weight: {avg_weight:,.0f} lbs
- LTL Shipments (<10k lbs): {ltl_count} ({ltl_count/len(data)*100:.1f}%)
- TL Shipments (≥10k lbs): {tl_count} ({tl_count/len(data)*100:.1f}%)
- Recommendation: Consider TL conversion for high-volume LTL lanes"""
    
    # Performance questions
    elif any(word in question_lower for word in ['performance', 'on-time', 'delivery', 'transit']):
        response = "**Performance Metrics:**\n"
        
        if 'On_Time_Delivery' in data.columns:
            on_time_rate = (data['On_Time_Delivery'] == 'Yes').mean() * 100
            response += f"- On-Time Delivery Rate: {on_time_rate:.1f}%\n"
        
        if 'Transit_Days' in data.columns:
            avg_transit = data['Transit_Days'].mean()
            response += f"- Average Transit Time: {avg_transit:.1f} days\n"
        
        if 'Profit_Margin_%' in data.columns:
            avg_margin = data['Profit_Margin_%'].mean()
            response += f"- Average Profit Margin: {avg_margin:.1f}%\n"
        
        return response if response != "**Performance Metrics:**\n" else "Performance metrics not available in the data."
    
    # Default response
    else:
        return """I can help you analyze:
- **Costs**: Total spend, averages, cost breakdown
- **Lanes**: Top routes, lane analysis
- **Carriers**: Performance, distribution
- **Consolidation**: Opportunities for combining shipments
- **Performance**: On-time delivery, transit times

Please ask a specific question about your transportation data!"""

def display_analytics():
    """Display analytics dashboard"""
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload or generate data first!")
        return
    
    data = st.session_state.data
    
    st.subheader("📈 Analytics Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_loads = len(data)
        st.metric("Total Loads", f"{total_loads:,}")
    
    with col2:
        if 'Total_Cost' in data.columns:
            total_cost = data['Total_Cost'].sum()
            st.metric("Total Cost", f"${total_cost:,.0f}")
    
    with col3:
        if 'Transit_Days' in data.columns:
            avg_transit = data['Transit_Days'].mean()
            st.metric("Avg Transit", f"{avg_transit:.1f} days")
    
    with col4:
        if 'On_Time_Delivery' in data.columns:
            on_time = (data['On_Time_Delivery'] == 'Yes').mean() * 100
            st.metric("On-Time %", f"{on_time:.1f}%")
    
    # Visualizations
    if 'Pickup_Date' in data.columns:
        try:
            # Time series analysis
            data['Date'] = pd.to_datetime(data['Pickup_Date'], errors='coerce')
            
            if not data['Date'].isna().all():
                daily_loads = data.groupby(data['Date'].dt.date).size().reset_index()
                daily_loads.columns = ['Date', 'Load_Count']
                
                fig = px.line(daily_loads, x='Date', y='Load_Count',
                            title='Daily Load Volume',
                            markers=True)
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Unable to create time series analysis")
    
    # Cost and carrier analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Total_Cost' in data.columns:
            fig = px.histogram(data, x='Total_Cost', nbins=30,
                             title='Cost Distribution')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Selected_Carrier' in data.columns:
            carrier_counts = data['Selected_Carrier'].value_counts()
            fig = px.pie(values=carrier_counts.values,
                       names=carrier_counts.index,
                       title='Carrier Market Share')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application function"""
    
    # Sidebar for data management
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("📊 Data Management")
        
        # Generate sample data button
        if st.button("🎲 Generate Sample Data", type="primary", use_container_width=True):
            with st.spinner("Generating data..."):
                st.session_state.data = generate_sample_data(500)
                st.success("✅ Generated 500 sample loads!")
                st.rerun()
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Your Data",
            type=['csv', 'xlsx'],
            help="Upload CSV or Excel file with transportation data"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                else:
                    st.session_state.data = pd.read_excel(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.data)} records")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        # Data info
        if st.session_state.data is not None:
            st.markdown("---")
            st.subheader("📋 Data Summary")
            st.write(f"**Records:** {len(st.session_state.data):,}")
            
            if 'Pickup_Date' in st.session_state.data.columns:
                try:
                    dates = pd.to_datetime(st.session_state.data['Pickup_Date'], errors='coerce')
                    date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                    st.write(f"**Date Range:** {date_range}")
                except:
                    pass
            
            if 'Selected_Carrier' in st.session_state.data.columns:
                st.write(f"**Carriers:** {st.session_state.data['Selected_Carrier'].nunique()}")
            
            if 'Origin_City' in st.session_state.data.columns:
                unique_lanes = st.session_state.data.groupby(['Origin_City', 'Destination_City']).ngroups
                st.write(f"**Unique Lanes:** {unique_lanes}")
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### About Lane Optimization
        
        AI-powered platform for:
        - Multi-carrier analysis
        - Route optimization
        - Cost prediction
        - Consolidation opportunities
        
        **Version 2.0**
        """)
    
    # Main content area with tabs
    if st.session_state.data is not None:
        tabs = st.tabs([
            "📊 Data Overview",
            "🛤️ Lane Analysis",
            "🎯 Route Optimizer",
            "🤖 AI Assistant",
            "📈 Analytics"
        ])
        
        with tabs[0]:
            display_data_overview()
        
        with tabs[1]:
            display_lane_analysis()
        
        with tabs[2]:
            display_route_optimizer()
        
        with tabs[3]:
            display_ai_assistant()
        
        with tabs[4]:
            display_analytics()
    else:
        # Show only data overview when no data is loaded
        display_data_overview()

if __name__ == "__main__":
    main()
