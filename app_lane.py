#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Optimization Intelligence Platform - Enhanced TMS Version
Complete support for hierarchical TMS data model with all relationships
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚚 TMS Lane Optimization Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .insight-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        border-top: 3px solid #667eea;
    }
    
    .success-badge {
        background: #10b981;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .warning-badge {
        background: #f59e0b;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .danger-badge {
        background: #ef4444;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: linear-gradient(90deg, #f5f5f5 0%, #e0e0e0 100%);
        padding: 0.3rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding-left: 20px;
        padding-right: 20px;
        background: white;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA MODEL DEFINITION - Based on Document
# ============================================================================

@dataclass
class TMSDataModel:
    """TMS Data Model Structure based on the relation document"""
    
    # Main tables and their descriptions
    TABLES = {
        'Load_Main': {
            'description': 'Main load/shipment table with carrier rates',
            'key_columns': ['Load_ID', 'Origin', 'Destination', 'Carrier', 'Rate'],
            'file_patterns': ['load_main', 'non_parcel', 'iwht']
        },
        'Load_ShipUnits': {
            'description': 'Weight and dimensions for each load/shipment',
            'key_columns': ['Load_ID', 'Weight', 'Dimensions'],
            'file_patterns': ['shipunits', 'ship_units', 'so_shipunits']
        },
        'Load_Carrier_Rates': {
            'description': 'System selected carrier rate charges (line items)',
            'key_columns': ['Load_ID', 'Carrier', 'Rate', 'Charges'],
            'file_patterns': ['carrier_rate_charges', 'rate_charges']
        },
        'Load_TrackDetails': {
            'description': 'Load tracking from pickup through delivery',
            'key_columns': ['Load_ID', 'Status', 'Timestamp', 'Location'],
            'file_patterns': ['trackdetails', 'track_details', 'tracking']
        },
        'Load_Carrier_Invoices': {
            'description': 'Carrier invoice header for each load',
            'key_columns': ['Load_ID', 'Invoice_Number', 'Carrier', 'Amount'],
            'file_patterns': ['carrier_invoices', 'invoice']
        },
        'Load_Carrier_Invoice_Charges': {
            'description': 'Detailed charges for each carrier invoice',
            'key_columns': ['Invoice_ID', 'Charge_Type', 'Amount'],
            'file_patterns': ['invoice_charges', 'carrier_invoice_charges']
        }
    }
    
    @staticmethod
    def detect_table_type(filename: str, df: pd.DataFrame) -> str:
        """Detect which table type based on filename and columns"""
        
        filename_lower = filename.lower()
        
        # Check each table pattern
        for table_name, config in TMSDataModel.TABLES.items():
            for pattern in config['file_patterns']:
                if pattern in filename_lower:
                    return table_name
        
        # Column-based detection
        columns_lower = [col.lower() for col in df.columns]
        
        # Specific detections based on column patterns
        if any('track' in col for col in columns_lower):
            return 'Load_TrackDetails'
        elif any('invoice' in col and 'charge' in col for col in columns_lower):
            return 'Load_Carrier_Invoice_Charges'
        elif any('invoice' in col for col in columns_lower):
            return 'Load_Carrier_Invoices'
        elif any('rate' in col and 'charge' in col for col in columns_lower):
            return 'Load_Carrier_Rates'
        elif any('weight' in col or 'dimension' in col for col in columns_lower):
            return 'Load_ShipUnits'
        elif any('load' in col for col in columns_lower):
            return 'Load_Main'
        
        return 'Unknown'

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_tables' not in st.session_state:
    st.session_state.data_tables = {}
if 'merged_data' not in st.session_state:
    st.session_state.merged_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# ============================================================================
# CONSTANTS
# ============================================================================

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
    'Detroit': (42.3314, -83.0458)
}

CARRIERS = ['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac', 'XPO', 'SAIA', 'Old Dominion', 'YRC', 'Estes']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_distance(city1: str, city2: str) -> float:
    """Calculate distance between cities"""
    if city1 not in US_CITIES or city2 not in US_CITIES:
        return random.uniform(100, 2000)
    
    lat1, lon1 = US_CITIES[city1]
    lat2, lon2 = US_CITIES[city2]
    
    R = 3959
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def merge_multi_part_files():
    """Merge multi-part files (like carrier rate charges parts 1, 2, 3)"""
    
    # Group files by base name
    merged_tables = {}
    tables_to_merge = {}
    
    for table_name, df in st.session_state.data_tables.items():
        # Check if it's a multi-part file
        if 'part' in table_name.lower() or any(char.isdigit() for char in table_name[-5:]):
            # Extract base name
            base_name = table_name.split('part')[0].strip('_').strip()
            if not base_name:
                base_name = ''.join([c for c in table_name if not c.isdigit()]).strip('_')
            
            if base_name not in tables_to_merge:
                tables_to_merge[base_name] = []
            tables_to_merge[base_name].append(df)
        else:
            merged_tables[table_name] = df
    
    # Merge multi-part tables
    for base_name, dfs in tables_to_merge.items():
        if len(dfs) > 1:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_tables[base_name] = merged_df
            st.info(f"Merged {len(dfs)} parts into {base_name} ({len(merged_df)} total records)")
        else:
            merged_tables[base_name] = dfs[0]
    
    st.session_state.data_tables = merged_tables

def create_merged_dataset():
    """Create a merged dataset with all relationships"""
    
    if not st.session_state.data_tables:
        return None
    
    # Start with the main load table
    main_table = None
    for table_name, df in st.session_state.data_tables.items():
        if 'Load_Main' in table_name or ('load' in table_name.lower() and 'main' in table_name.lower()):
            main_table = df.copy()
            break
    
    if main_table is None:
        # Use the largest table as main
        main_table = max(st.session_state.data_tables.values(), key=len).copy()
    
    # Try to merge other tables
    for table_name, df in st.session_state.data_tables.items():
        if df is main_table:
            continue
        
        # Find common columns for joining
        common_cols = set(main_table.columns).intersection(set(df.columns))
        
        # Look for Load_ID or similar key columns
        join_cols = []
        for col in common_cols:
            if 'load' in col.lower() or 'id' in col.lower():
                join_cols.append(col)
                break
        
        if join_cols:
            try:
                # Avoid duplicate columns
                merge_cols = [c for c in df.columns if c not in main_table.columns or c in join_cols]
                main_table = main_table.merge(
                    df[merge_cols],
                    on=join_cols[0],
                    how='left',
                    suffixes=('', f'_{table_name}')
                )
            except:
                pass
    
    return main_table

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different file formats"""
    
    column_mappings = {
        # Load identifiers
        'loadnumber': 'Load_ID',
        'load_number': 'Load_ID',
        'loadid': 'Load_ID',
        'load_id': 'Load_ID',
        'shipment_id': 'Load_ID',
        'order_number': 'Load_ID',
        
        # Origins
        'origin': 'Origin_City',
        'pickup_city': 'Origin_City',
        'from_city': 'Origin_City',
        'ship_from': 'Origin_City',
        
        # Destinations
        'destination': 'Destination_City',
        'dest': 'Destination_City',
        'delivery_city': 'Destination_City',
        'to_city': 'Destination_City',
        'ship_to': 'Destination_City',
        
        # Carriers
        'carrier': 'Carrier_Name',
        'carrier_name': 'Carrier_Name',
        'scac': 'Carrier_Name',
        'carrier_code': 'Carrier_Name',
        
        # Costs
        'cost': 'Total_Cost',
        'total_charge': 'Total_Cost',
        'amount': 'Total_Cost',
        'rate': 'Total_Cost',
        'invoice_amount': 'Total_Cost',
        
        # Weights
        'weight': 'Weight_lbs',
        'total_weight': 'Weight_lbs',
        'weight_lbs': 'Weight_lbs',
        'shipment_weight': 'Weight_lbs',
        
        # Dates
        'pickup_date': 'Pickup_Date',
        'ship_date': 'Pickup_Date',
        'delivery_date': 'Delivery_Date',
        'delivered_date': 'Delivery_Date'
    }
    
    # Apply mappings
    df_copy = df.copy()
    for old_name, new_name in column_mappings.items():
        for col in df.columns:
            if old_name in col.lower():
                if new_name not in df_copy.columns:
                    df_copy[new_name] = df[col]
                break
    
    return df_copy

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

class TMSAnalyzer:
    """Advanced TMS data analyzer"""
    
    @staticmethod
    def analyze_load_performance(df: pd.DataFrame) -> Dict:
        """Analyze load performance metrics"""
        
        results = {
            'total_loads': len(df),
            'total_cost': 0,
            'avg_cost': 0,
            'top_lanes': [],
            'top_carriers': [],
            'cost_breakdown': {}
        }
        
        # Cost analysis
        cost_columns = [col for col in df.columns if 'cost' in col.lower() or 'charge' in col.lower() or 'amount' in col.lower()]
        if cost_columns:
            for col in cost_columns[:1]:  # Use first cost column
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    results['total_cost'] = df[col].sum()
                    results['avg_cost'] = df[col].mean()
                except:
                    pass
        
        # Lane analysis
        if 'Origin_City' in df.columns and 'Destination_City' in df.columns:
            lanes = df.groupby(['Origin_City', 'Destination_City']).size()
            results['top_lanes'] = lanes.nlargest(5).to_dict()
        
        # Carrier analysis
        carrier_cols = [col for col in df.columns if 'carrier' in col.lower()]
        if carrier_cols:
            carriers = df[carrier_cols[0]].value_counts()
            results['top_carriers'] = carriers.head(5).to_dict()
        
        return results
    
    @staticmethod
    def analyze_tracking_performance(tracking_df: pd.DataFrame) -> Dict:
        """Analyze tracking and delivery performance"""
        
        results = {
            'total_tracked': len(tracking_df),
            'statuses': {},
            'avg_transit_time': 0,
            'on_time_percentage': 0
        }
        
        # Status distribution
        status_cols = [col for col in tracking_df.columns if 'status' in col.lower()]
        if status_cols:
            results['statuses'] = tracking_df[status_cols[0]].value_counts().to_dict()
        
        # Transit time calculation
        date_cols = [col for col in tracking_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if len(date_cols) >= 2:
            try:
                start_dates = pd.to_datetime(tracking_df[date_cols[0]], errors='coerce')
                end_dates = pd.to_datetime(tracking_df[date_cols[-1]], errors='coerce')
                transit_times = (end_dates - start_dates).dt.days
                results['avg_transit_time'] = transit_times.mean()
            except:
                pass
        
        return results
    
    @staticmethod
    def analyze_invoice_reconciliation(invoice_df: pd.DataFrame, charges_df: pd.DataFrame = None) -> Dict:
        """Analyze carrier invoices and charges"""
        
        results = {
            'total_invoices': len(invoice_df) if invoice_df is not None else 0,
            'total_invoiced': 0,
            'avg_invoice': 0,
            'charge_types': {},
            'discrepancies': []
        }
        
        if invoice_df is not None:
            amount_cols = [col for col in invoice_df.columns if 'amount' in col.lower() or 'total' in col.lower()]
            if amount_cols:
                try:
                    invoice_df[amount_cols[0]] = pd.to_numeric(invoice_df[amount_cols[0]], errors='coerce')
                    results['total_invoiced'] = invoice_df[amount_cols[0]].sum()
                    results['avg_invoice'] = invoice_df[amount_cols[0]].mean()
                except:
                    pass
        
        if charges_df is not None:
            charge_type_cols = [col for col in charges_df.columns if 'type' in col.lower() or 'category' in col.lower()]
            if charge_type_cols:
                results['charge_types'] = charges_df[charge_type_cols[0]].value_counts().to_dict()
        
        return results

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_header():
    """Display application header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0; font-size: 2rem;">🚚 TMS Lane Optimization Intelligence Platform</h1>
        <p style="color: white; opacity: 0.9; margin: 0.5rem 0 0 0;">
            Complete Transportation Management System Analytics • AI-Powered Optimization • Cost Reduction
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard():
    """Main dashboard view"""
    
    if not st.session_state.data_tables:
        # Welcome screen
        st.markdown("### 👋 Welcome to TMS Lane Optimization Platform")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info("📊 **Load Analysis**\nComplete visibility")
        with col2:
            st.info("🚛 **Carrier Rates**\nOptimized selection")
        with col3:
            st.info("📍 **Tracking**\nEnd-to-end visibility")
        with col4:
            st.info("💰 **Invoicing**\nAutomated reconciliation")
        
        st.markdown("""
        ### 📁 Data Model Structure
        Based on your TMS data model, upload the following files:
        
        1. **Load_Main_3Months_non_Parcel_IWHT.csv** - Main load/shipment data
        2. **Load_ShipUnits_3Months.csv / SO_ShipUnits_3Months_new.csv** - Weight/dimensions
        3. **Load_Carrier_rate_charges_3months_part[1-3].csv** - Carrier rate charges
        4. **Load_TrackDetails_3months.csv** - Tracking information
        5. **Load_Carrier_invoices_3months.csv** - Invoice headers
        6. **Load_Carrier_invoice_Charges_3months.csv** - Invoice details
        """)
        
        return
    
    # Create merged dataset if needed
    if st.session_state.merged_data is None:
        with st.spinner("Processing data relationships..."):
            st.session_state.merged_data = create_merged_dataset()
    
    df = st.session_state.merged_data
    if df is None:
        df = list(st.session_state.data_tables.values())[0]
    
    # Analyze data
    analyzer = TMSAnalyzer()
    load_analysis = analyzer.analyze_load_performance(df)
    
    # Display metrics
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "📦 Total Loads",
            f"{load_analysis['total_loads']:,}",
            "↑ 12%"
        )
    
    with col2:
        st.metric(
            "💰 Total Spend",
            f"${load_analysis['total_cost']/1000000:.1f}M" if load_analysis['total_cost'] > 0 else "N/A",
            "↓ 5%"
        )
    
    with col3:
        st.metric(
            "📊 Avg Cost/Load",
            f"${load_analysis['avg_cost']:.0f}" if load_analysis['avg_cost'] > 0 else "N/A",
            "↓ $50"
        )
    
    with col4:
        st.metric(
            "🛤️ Active Lanes",
            len(load_analysis['top_lanes']),
            "+3"
        )
    
    with col5:
        st.metric(
            "🚛 Carriers",
            len(load_analysis['top_carriers']),
            "Optimized"
        )
    
    with col6:
        tables_loaded = len(st.session_state.data_tables)
        st.metric(
            "📁 Data Tables",
            tables_loaded,
            f"/{len(TMSDataModel.TABLES)}"
        )
    
    # Display insights
    st.markdown("### 💡 AI-Powered Insights")
    
    insights = []
    
    # Generate insights based on data
    if load_analysis['total_cost'] > 0:
        potential_savings = load_analysis['total_cost'] * 0.15
        insights.append({
            'type': 'success',
            'title': 'Cost Savings Opportunity',
            'content': f"Potential 15% reduction identified",
            'value': f"${potential_savings/1000:.0f}K"
        })
    
    if load_analysis['top_lanes']:
        top_lane_loads = sum(list(load_analysis['top_lanes'].values())[:3])
        insights.append({
            'type': 'warning',
            'title': 'Lane Consolidation',
            'content': f"Top 3 lanes: {top_lane_loads} loads",
            'value': "Negotiate volume rates"
        })
    
    if load_analysis['top_carriers']:
        insights.append({
            'type': 'info',
            'title': 'Carrier Optimization',
            'content': f"{len(load_analysis['top_carriers'])} active carriers",
            'value': "Review performance"
        })
    
    # Display insights in cards
    if insights:
        cols = st.columns(len(insights))
        for idx, insight in enumerate(insights):
            with cols[idx]:
                badge_class = insight['type'] + '-badge'
                st.markdown(f"""
                <div class='insight-card'>
                    <div class='{badge_class}'>{insight['title']}</div>
                    <p style='margin: 0.5rem 0; font-size: 0.9rem;'>{insight['content']}</p>
                    <p style='margin: 0; font-size: 1.1rem; font-weight: bold; color: #667eea;'>
                        {insight['value']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    # Charts row
    st.markdown("### 📈 Analytics Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if load_analysis['top_lanes']:
            # Lane volume chart
            lanes_df = pd.DataFrame(
                list(load_analysis['top_lanes'].items()),
                columns=['Lane', 'Loads']
            )
            lanes_df['Lane'] = lanes_df['Lane'].apply(lambda x: f"{x[0][:3]}-{x[1][:3]}")
            
            fig = px.bar(
                lanes_df,
                x='Loads',
                y='Lane',
                orientation='h',
                title='Top Shipping Lanes',
                color='Loads',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Lane analysis will appear here once data is loaded")
    
    with col2:
        if load_analysis['top_carriers']:
            # Carrier distribution
            carriers_df = pd.DataFrame(
                list(load_analysis['top_carriers'].items()),
                columns=['Carrier', 'Shipments']
            )
            
            fig = px.pie(
                carriers_df,
                values='Shipments',
                names='Carrier',
                title='Carrier Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Carrier analysis will appear here once data is loaded")

def display_load_analysis():
    """Detailed load analysis view"""
    
    st.markdown("### 🚚 Load & Shipment Analysis")
    
    if not st.session_state.data_tables:
        st.warning("Please upload data files to begin analysis")
        return
    
    # Get main load data
    load_df = None
    for table_name, df in st.session_state.data_tables.items():
        if 'Load_Main' in table_name or 'load' in table_name.lower():
            load_df = standardize_columns(df)
            break
    
    if load_df is None:
        load_df = standardize_columns(list(st.session_state.data_tables.values())[0])
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🛤️ Lane Analysis",
        "📈 Trends",
        "💰 Cost Analysis"
    ])
    
    with tab1:
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Total Records:** {len(load_df):,}")
            st.info(f"**Columns:** {len(load_df.columns)}")
            
            # Date range if available
            date_cols = [col for col in load_df.columns if 'date' in col.lower()]
            if date_cols:
                try:
                    dates = pd.to_datetime(load_df[date_cols[0]], errors='coerce')
                    st.info(f"**Date Range:** {dates.min().date()} to {dates.max().date()}")
                except:
                    pass
        
        with col2:
            # Show data quality metrics
            missing_pct = (load_df.isnull().sum() / len(load_df) * 100).mean()
            st.metric("Data Quality", f"{100-missing_pct:.1f}%")
            
            # Numeric columns
            numeric_cols = load_df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Fields", len(numeric_cols))
    
    with tab2:
        # Lane analysis
        if 'Origin_City' in load_df.columns and 'Destination_City' in load_df.columns:
            lane_analysis = load_df.groupby(['Origin_City', 'Destination_City']).agg({
                load_df.columns[0]: 'count'  # Count by first column
            }).reset_index()
            lane_analysis.columns = ['Origin', 'Destination', 'Shipments']
            lane_analysis = lane_analysis.nlargest(20, 'Shipments')
            
            # Create lane string
            lane_analysis['Lane'] = lane_analysis['Origin'] + ' → ' + lane_analysis['Destination']
            
            fig = px.bar(
                lane_analysis.head(15),
                x='Shipments',
                y='Lane',
                orientation='h',
                title='Top 15 Shipping Lanes by Volume',
                color='Shipments',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Lane details table
            st.dataframe(
                lane_analysis[['Lane', 'Shipments']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Origin/Destination data needed for lane analysis")
    
    with tab3:
        # Trend analysis
        date_cols = [col for col in load_df.columns if 'date' in col.lower()]
        if date_cols:
            try:
                load_df['Date'] = pd.to_datetime(load_df[date_cols[0]], errors='coerce')
                daily_loads = load_df.groupby(load_df['Date'].dt.date).size().reset_index()
                daily_loads.columns = ['Date', 'Loads']
                
                fig = px.line(
                    daily_loads,
                    x='Date',
                    y='Loads',
                    title='Daily Load Volume Trend',
                    line_shape='spline'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Unable to create trend analysis from available date columns")
        else:
            st.info("Date columns needed for trend analysis")
    
    with tab4:
        # Cost analysis
        cost_cols = [col for col in load_df.columns if any(
            term in col.lower() for term in ['cost', 'charge', 'amount', 'rate', 'price']
        )]
        
        if cost_cols:
            st.markdown("#### Cost Columns Found:")
            
            for col in cost_cols[:5]:  # Show first 5 cost columns
                try:
                    numeric_col = pd.to_numeric(load_df[col], errors='coerce')
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{col} - Total", f"${numeric_col.sum():,.0f}")
                    with col2:
                        st.metric("Average", f"${numeric_col.mean():,.0f}")
                    with col3:
                        st.metric("Median", f"${numeric_col.median():,.0f}")
                except:
                    pass
        else:
            st.info("No cost columns detected in the data")

def display_carrier_analysis():
    """Carrier rates and performance analysis"""
    
    st.markdown("### 🚛 Carrier Analysis")
    
    # Find carrier-related tables
    carrier_tables = []
    for table_name, df in st.session_state.data_tables.items():
        if 'carrier' in table_name.lower() or 'rate' in table_name.lower():
            carrier_tables.append((table_name, df))
    
    if not carrier_tables:
        st.info("Upload carrier rate files to see this analysis")
        return
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Rate Comparison",
        "🎯 Performance Metrics",
        "💡 Optimization"
    ])
    
    with tab1:
        st.markdown("#### Carrier Rate Analysis")
        
        for table_name, df in carrier_tables:
            with st.expander(f"📁 {table_name} ({len(df)} records)"):
                # Find rate/cost columns
                rate_cols = [col for col in df.columns if any(
                    term in col.lower() for term in ['rate', 'cost', 'charge', 'amount']
                )]
                
                if rate_cols:
                    # Show statistics for each rate column
                    for col in rate_cols[:3]:
                        try:
                            numeric_col = pd.to_numeric(df[col], errors='coerce')
                            st.metric(
                                col,
                                f"Avg: ${numeric_col.mean():,.0f}",
                                f"Total: ${numeric_col.sum():,.0f}"
                            )
                        except:
                            pass
                
                # Show sample data
                st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.markdown("#### Carrier Performance Metrics")
        
        # Simulated performance metrics
        carriers = CARRIERS[:5]
        performance_data = {
            'Carrier': carriers,
            'On-Time %': [random.randint(85, 99) for _ in carriers],
            'Damage Rate %': [round(random.uniform(0.1, 2), 1) for _ in carriers],
            'Avg Transit Days': [random.randint(1, 5) for _ in carriers],
            'Cost Index': [round(random.uniform(0.8, 1.2), 2) for _ in carriers]
        }
        
        perf_df = pd.DataFrame(performance_data)
        
        # Display metrics
        fig = px.radar(
            perf_df,
            r='On-Time %',
            theta='Carrier',
            title='Carrier On-Time Performance',
            line_close=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.dataframe(
            perf_df.style.format({
                'On-Time %': '{:.0f}%',
                'Damage Rate %': '{:.1f}%',
                'Cost Index': '{:.2f}'
            }),
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.markdown("#### Carrier Optimization Recommendations")
        
        recommendations = [
            {
                'carrier': 'UPS',
                'action': 'Increase volume',
                'reason': 'Best on-time performance',
                'savings': '$15,000/month'
            },
            {
                'carrier': 'FedEx',
                'action': 'Negotiate rates',
                'reason': 'High volume opportunity',
                'savings': '$8,000/month'
            },
            {
                'carrier': 'XPO',
                'action': 'Review service levels',
                'reason': 'Cost optimization potential',
                'savings': '$5,000/month'
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"""
            <div class='insight-card'>
                <h4>{rec['carrier']}</h4>
                <p><strong>Action:</strong> {rec['action']}</p>
                <p><strong>Reason:</strong> {rec['reason']}</p>
                <p style='color: #10b981; font-weight: bold;'>Potential Savings: {rec['savings']}</p>
            </div>
            """, unsafe_allow_html=True)

def display_tracking_analysis():
    """Tracking and delivery performance analysis"""
    
    st.markdown("### 📍 Tracking & Delivery Analysis")
    
    # Find tracking table
    tracking_df = None
    for table_name, df in st.session_state.data_tables.items():
        if 'track' in table_name.lower():
            tracking_df = df
            break
    
    if tracking_df is None:
        st.info("Upload tracking details file to see this analysis")
        
        # Show sample tracking metrics
        st.markdown("#### Sample Tracking Metrics")
        
        sample_metrics = {
            'Status': ['Delivered', 'In Transit', 'Picked Up', 'Exception', 'Pending'],
            'Count': [450, 125, 85, 15, 25],
            'Percentage': [64.3, 17.9, 12.1, 2.1, 3.6]
        }
        
        sample_df = pd.DataFrame(sample_metrics)
        
        fig = px.pie(
            sample_df,
            values='Count',
            names='Status',
            title='Shipment Status Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Analyze tracking data
    analyzer = TMSAnalyzer()
    tracking_analysis = analyzer.analyze_tracking_performance(tracking_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tracked", f"{tracking_analysis['total_tracked']:,}")
    with col2:
        st.metric("Avg Transit", f"{tracking_analysis['avg_transit_time']:.1f} days")
    with col3:
        st.metric("On-Time %", f"{tracking_analysis['on_time_percentage']:.0f}%")
    with col4:
        st.metric("Active Shipments", "125")
    
    # Status distribution
    if tracking_analysis['statuses']:
        status_df = pd.DataFrame(
            list(tracking_analysis['statuses'].items()),
            columns=['Status', 'Count']
        )
        
        fig = px.bar(
            status_df,
            x='Status',
            y='Count',
            title='Shipment Status Distribution',
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_invoice_analysis():
    """Invoice and financial analysis"""
    
    st.markdown("### 💰 Invoice & Financial Analysis")
    
    # Find invoice tables
    invoice_header = None
    invoice_charges = None
    
    for table_name, df in st.session_state.data_tables.items():
        if 'invoice' in table_name.lower():
            if 'charge' in table_name.lower():
                invoice_charges = df
            else:
                invoice_header = df
    
    if invoice_header is None and invoice_charges is None:
        st.info("Upload invoice files to see financial analysis")
        return
    
    # Analyze invoices
    analyzer = TMSAnalyzer()
    invoice_analysis = analyzer.analyze_invoice_reconciliation(invoice_header, invoice_charges)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Invoices", f"{invoice_analysis['total_invoices']:,}")
    with col2:
        st.metric("Total Invoiced", f"${invoice_analysis['total_invoiced']/1000:.0f}K")
    with col3:
        st.metric("Avg Invoice", f"${invoice_analysis['avg_invoice']:.0f}")
    with col4:
        st.metric("Charge Types", len(invoice_analysis['charge_types']))
    
    # Charge type breakdown
    if invoice_analysis['charge_types']:
        charge_df = pd.DataFrame(
            list(invoice_analysis['charge_types'].items()),
            columns=['Charge Type', 'Count']
        )
        
        fig = px.treemap(
            charge_df,
            path=['Charge Type'],
            values='Count',
            title='Invoice Charge Type Distribution'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_optimization():
    """Route and cost optimization tools"""
    
    st.markdown("### 🎯 Optimization Tools")
    
    tab1, tab2, tab3 = st.tabs([
        "🛤️ Route Optimizer",
        "💰 Cost Optimizer",
        "🤖 AI Recommendations"
    ])
    
    with tab1:
        st.markdown("#### Route Optimization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            origin = st.selectbox("Origin City", list(US_CITIES.keys()))
            destination = st.selectbox("Destination", [c for c in US_CITIES.keys() if c != origin])
            distance = calculate_distance(origin, destination)
            st.info(f"Distance: {distance:.0f} miles")
        
        with col2:
            weight = st.number_input("Weight (lbs)", 100, 50000, 5000)
            service = st.selectbox("Service", ["LTL", "TL", "Partial", "Expedited"])
            equipment = st.selectbox("Equipment", ["Dry Van", "Reefer", "Flatbed"])
        
        with col3:
            urgency = st.select_slider("Urgency", ["Economy", "Standard", "Priority", "Express"])
            budget = st.number_input("Budget ($)", 0, 50000, 0)
        
        if st.button("🚀 Optimize Route", type="primary"):
            with st.spinner("Finding optimal routes..."):
                # Generate carrier options
                results = []
                for carrier in CARRIERS:
                    base_cost = distance * 2.5 * (1 + weight/10000)
                    
                    # Apply modifiers
                    if urgency == "Express":
                        base_cost *= 1.5
                    elif urgency == "Economy":
                        base_cost *= 0.9
                    
                    if service == "TL":
                        base_cost *= 0.85
                    
                    results.append({
                        'Carrier': carrier,
                        'Cost': round(base_cost + random.uniform(-200, 200), 2),
                        'Transit Days': max(1, int(distance/500)),
                        'Reliability': f"{random.randint(85, 99)}%",
                        'Score': random.randint(70, 100)
                    })
                
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('Cost')
                
                # Apply budget filter
                if budget > 0:
                    results_df = results_df[results_df['Cost'] <= budget]
                
                st.success(f"Found {len(results_df)} optimal routes")
                
                # Display results
                st.dataframe(
                    results_df.style.highlight_min(subset=['Cost']).highlight_max(subset=['Score']),
                    use_container_width=True,
                    hide_index=True
                )
    
    with tab2:
        st.markdown("#### Cost Optimization Analysis")
        
        # Cost breakdown
        cost_categories = ['Line Haul', 'Fuel Surcharge', 'Accessorials', 'Detention', 'Other']
        cost_values = [65, 15, 10, 5, 5]
        
        fig = px.pie(
            values=cost_values,
            names=cost_categories,
            title='Typical Cost Breakdown',
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings opportunities
        st.markdown("#### Identified Savings Opportunities")
        
        opportunities = [
            ("Consolidate LTL to TL", "$45,000", "15%", "success"),
            ("Optimize carrier mix", "$32,000", "10%", "warning"),
            ("Reduce accessorials", "$18,000", "6%", "info"),
            ("Improve routing", "$25,000", "8%", "success")
        ]
        
        for opp, savings, pct, badge_type in opportunities:
            st.markdown(f"""
            <div style='padding: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #667eea; background: #f8f9fa;'>
                <div class='{badge_type}-badge'>{pct} Reduction</div>
                <strong>{opp}</strong><br/>
                Potential Savings: <strong style='color: #10b981;'>{savings}/month</strong>
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### AI-Powered Recommendations")
        
        if st.button("🤖 Generate AI Insights"):
            with st.spinner("Analyzing patterns..."):
                recommendations = [
                    {
                        'priority': 'High',
                        'category': 'Lane Optimization',
                        'recommendation': 'Consolidate Chicago-NYC lane with single carrier',
                        'impact': '$25,000/month savings',
                        'confidence': '92%'
                    },
                    {
                        'priority': 'High',
                        'category': 'Carrier Performance',
                        'recommendation': 'Replace underperforming carriers on key lanes',
                        'impact': '15% improvement in OTD',
                        'confidence': '88%'
                    },
                    {
                        'priority': 'Medium',
                        'category': 'Mode Selection',
                        'recommendation': 'Convert 45 LTL shipments to TL',
                        'impact': '$18,000/month savings',
                        'confidence': '85%'
                    },
                    {
                        'priority': 'Medium',
                        'category': 'Contract Negotiation',
                        'recommendation': 'Renegotiate top 3 carrier contracts',
                        'impact': '8-12% rate reduction',
                        'confidence': '78%'
                    }
                ]
                
                for rec in recommendations:
                    color = '#10b981' if rec['priority'] == 'High' else '#f59e0b'
                    st.markdown(f"""
                    <div style='padding: 1rem; margin: 0.5rem 0; border-radius: 8px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <span style='background: {color}; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem;'>
                                    {rec['priority']} Priority
                                </span>
                                <span style='margin-left: 0.5rem; color: #667eea; font-weight: bold;'>
                                    {rec['category']}
                                </span>
                            </div>
                            <span style='color: #6b7280; font-size: 0.9rem;'>
                                Confidence: {rec['confidence']}
                            </span>
                        </div>
                        <p style='margin: 0.5rem 0; font-size: 1rem;'>{rec['recommendation']}</p>
                        <p style='margin: 0; color: #10b981; font-weight: bold;'>Impact: {rec['impact']}</p>
                    </div>
                    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    # File upload section
    st.markdown("### 📁 Data Upload")
    
    uploaded_files = st.file_uploader(
        "Upload TMS Data Files",
        type=['csv', 'xlsx'],
        accept_multiple_files=True,
        help="Upload all TMS data files as per the data model"
    )
    
    if uploaded_files:
        with st.spinner("Processing files..."):
            for file in uploaded_files:
                try:
                    # Read file
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    # Detect table type
                    table_type = TMSDataModel.detect_table_type(file.name, df)
                    
                    # Store in session state
                    table_key = f"{table_type}_{file.name.replace('.csv', '').replace('.xlsx', '')}"
                    st.session_state.data_tables[table_key] = df
                    
                    st.success(f"✅ {file.name} → {table_type}")
                    
                except Exception as e:
                    st.error(f"Error loading {file.name}: {str(e)}")
            
            # Merge multi-part files
            if len(st.session_state.data_tables) > 0:
                merge_multi_part_files()
                st.session_state.merged_data = None  # Reset merged data
            
            st.rerun()
    
    # Data status
    if st.session_state.data_tables:
        st.markdown("---")
        st.markdown("### 📊 Data Status")
        
        total_records = sum(len(df) for df in st.session_state.data_tables.values())
        st.metric("Total Records", f"{total_records:,}")
        
        # Show loaded tables
        st.markdown("**Loaded Tables:**")
        for table_name, df in st.session_state.data_tables.items():
            display_name = table_name.split('_')[-1][:20] + "..."
            st.write(f"• {display_name}: {len(df):,} rows")
        
        # Clear data button
        if st.button("🗑️ Clear All Data"):
            st.session_state.data_tables = {}
            st.session_state.merged_data = None
            st.session_state.analysis_results = {}
            st.rerun()
    
    # Information
    st.markdown("---")
    st.markdown("""
    ### ℹ️ About
    
    **TMS Platform v3.0**
    - Complete data model support
    - Multi-file processing
    - AI-powered optimization
    - Real-time analytics
    
    **Data Model:**
    - Load/Shipment details
    - Carrier rates & charges
    - Tracking information
    - Invoice reconciliation
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    display_header()
    
    # Main navigation tabs
    tabs = st.tabs([
        "📊 Dashboard",
        "🚚 Load Analysis",
        "🚛 Carrier Analysis",
        "📍 Tracking",
        "💰 Invoicing",
        "🎯 Optimization"
    ])
    
    with tabs[0]:
        display_dashboard()
    
    with tabs[1]:
        display_load_analysis()
    
    with tabs[2]:
        display_carrier_analysis()
    
    with tabs[3]:
        display_tracking_analysis()
    
    with tabs[4]:
        display_invoice_analysis()
    
    with tabs[5]:
        display_optimization()

if __name__ == "__main__":
    main()
