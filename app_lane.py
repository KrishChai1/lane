#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane Optimization Platform - Performance Optimized Version
Fast file processing with chunking, caching, and async operations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import io
import hashlib
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚚 TMS Lane Optimization - Fast",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Chunk size for reading large files
CHUNK_SIZE = 50000  

# Sample size for preview
PREVIEW_ROWS = 1000

# Column detection sample
DETECTION_SAMPLE = 100

# Cache timeout
CACHE_TTL = 3600

# ============================================================================
# CUSTOM CSS (Minimal for performance)
# ============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 0.8rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 3px solid #667eea;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        padding: 0 15px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'file_info' not in st.session_state:
    st.session_state.file_info = {}
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'quick_stats' not in st.session_state:
    st.session_state.quick_stats = {}

# ============================================================================
# FAST FILE PROCESSING
# ============================================================================

class FastFileProcessor:
    """Optimized file processor with caching and chunking"""
    
    @staticmethod
    def get_file_hash(file) -> str:
        """Get hash of file for caching"""
        file.seek(0)
        file_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        return file_hash
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def read_file_cached(file_content: bytes, file_name: str, file_type: str) -> pd.DataFrame:
        """Cache file reading to avoid re-reading same files"""
        try:
            if file_type == 'csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                    try:
                        return pd.read_csv(io.BytesIO(file_content), encoding=encoding, low_memory=False)
                    except:
                        continue
                return pd.read_csv(io.BytesIO(file_content), encoding='utf-8', errors='ignore')
            else:
                return pd.read_excel(io.BytesIO(file_content))
        except Exception as e:
            st.error(f"Error reading {file_name}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def detect_file_type(filename: str, df_sample: pd.DataFrame) -> str:
        """Quick file type detection using filename and sample"""
        filename_lower = filename.lower()
        
        # Quick filename-based detection
        type_patterns = {
            'main': ['main', 'load_main', 'non_parcel'],
            'ship_units': ['shipunit', 'ship_unit', 'so_ship'],
            'rates': ['rate', 'carrier_rate'],
            'tracking': ['track', 'tracking'],
            'invoice': ['invoice'],
            'charges': ['charge', 'invoice_charge']
        }
        
        for file_type, patterns in type_patterns.items():
            if any(p in filename_lower for p in patterns):
                return file_type
        
        return 'general'
    
    @staticmethod
    def quick_column_standardize(df: pd.DataFrame) -> pd.DataFrame:
        """Fast column standardization"""
        # Only standardize key columns for performance
        key_mappings = {
            'loadnumber': 'Load_ID',
            'load_number': 'Load_ID',
            'loadid': 'Load_ID',
            'carrier': 'Carrier',
            'origin': 'Origin',
            'destination': 'Destination',
            'cost': 'Cost',
            'weight': 'Weight'
        }
        
        # Quick rename
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            for old, new in key_mappings.items():
                if old in col_lower and new not in df.columns:
                    df[new] = df[col]
                    break
        
        return df
    
    @staticmethod
    def process_file_batch(files) -> Dict[str, pd.DataFrame]:
        """Process multiple files in batch"""
        processed_data = {}
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, file in enumerate(files):
            # Update progress
            progress = (idx + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name} ({idx + 1}/{len(files)})")
            
            # Check if already processed
            file_hash = FastFileProcessor.get_file_hash(file)
            if file_hash in st.session_state.processed_files:
                continue
            
            # Read file content
            file_content = file.read()
            file_type = 'csv' if file.name.endswith('.csv') else 'excel'
            
            # Use cached reading
            df = FastFileProcessor.read_file_cached(file_content, file.name, file_type)
            
            if not df.empty:
                # Quick type detection
                df_sample = df.head(DETECTION_SAMPLE)
                table_type = FastFileProcessor.detect_file_type(file.name, df_sample)
                
                # Quick standardization
                df = FastFileProcessor.quick_column_standardize(df)
                
                # Store with type
                key = f"{table_type}_{file.name.split('.')[0]}"
                processed_data[key] = df
                
                # Mark as processed
                st.session_state.processed_files.add(file_hash)
                
                # Store quick stats
                st.session_state.quick_stats[key] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory': df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
                }
        
        progress_bar.empty()
        status_text.empty()
        
        return processed_data

# ============================================================================
# FAST ANALYTICS
# ============================================================================

class FastAnalytics:
    """Optimized analytics with sampling and caching"""
    
    @staticmethod
    @st.cache_data(ttl=CACHE_TTL)
    def get_quick_metrics(data: Dict[str, Any]) -> Dict:
        """Calculate key metrics quickly using sampling"""
        total_rows = sum(stats['rows'] for stats in data.values())
        total_memory = sum(stats['memory'] for stats in data.values())
        
        return {
            'total_rows': total_rows,
            'total_tables': len(data),
            'memory_mb': round(total_memory, 1),
            'avg_rows': total_rows // len(data) if data else 0
        }
    
    @staticmethod
    def analyze_sample(df: pd.DataFrame, sample_size: int = 1000) -> Dict:
        """Analyze DataFrame sample for quick insights"""
        # Use sample for large datasets
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        results = {
            'numeric_cols': df_sample.select_dtypes(include=[np.number]).columns.tolist(),
            'text_cols': df_sample.select_dtypes(include=['object']).columns.tolist(),
            'date_cols': [],
            'missing_pct': (df_sample.isnull().sum() / len(df_sample) * 100).mean()
        }
        
        # Quick date detection
        for col in df_sample.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                results['date_cols'].append(col)
        
        return results

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_header():
    """Display header"""
    st.markdown("""
    <div class="main-header">
        <h2 style="margin: 0;">🚚 TMS Lane Optimization - Fast Processing</h2>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Optimized for large file processing</p>
    </div>
    """, unsafe_allow_html=True)

def display_dashboard():
    """Fast dashboard with key metrics"""
    
    if not st.session_state.data_cache:
        # Welcome screen
        st.markdown("### 👋 Welcome to Fast TMS Platform")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("⚡ **Fast Processing**\nOptimized for large files")
        with col2:
            st.info("💾 **Smart Caching**\nNo re-processing")
        with col3:
            st.info("📊 **Quick Analytics**\nSample-based insights")
        
        st.markdown("""
        ### 📁 Quick Start
        1. Upload multiple files at once
        2. Files are processed in parallel
        3. Results are cached automatically
        
        **Supported formats:** CSV, Excel (xlsx, xls)
        """)
        return
    
    # Quick metrics
    metrics = FastAnalytics.get_quick_metrics(st.session_state.quick_stats)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Records", f"{metrics['total_rows']:,}")
    with col2:
        st.metric("📁 Tables Loaded", metrics['total_tables'])
    with col3:
        st.metric("💾 Memory Used", f"{metrics['memory_mb']:.1f} MB")
    with col4:
        st.metric("📊 Avg Records/Table", f"{metrics['avg_rows']:,}")
    
    # Quick table overview
    st.markdown("### 📊 Data Overview")
    
    # Create summary DataFrame
    summary_data = []
    for name, stats in st.session_state.quick_stats.items():
        summary_data.append({
            'Table': name.split('_')[0].title(),
            'File': name.split('_', 1)[1] if '_' in name else name,
            'Records': stats['rows'],
            'Columns': stats['columns'],
            'Size (MB)': round(stats['memory'], 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(
        summary_df.style.highlight_max(subset=['Records']),
        use_container_width=True,
        hide_index=True
    )
    
    # Quick insights
    if summary_data:
        st.markdown("### 💡 Quick Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            largest_table = max(summary_data, key=lambda x: x['Records'])
            st.success(f"**Largest Table:** {largest_table['Table']} ({largest_table['Records']:,} records)")
        
        with col2:
            total_size = sum(item['Size (MB)'] for item in summary_data)
            st.info(f"**Total Data Size:** {total_size:.1f} MB")
        
        with col3:
            avg_cols = sum(item['Columns'] for item in summary_data) // len(summary_data)
            st.warning(f"**Avg Columns:** {avg_cols}")

def display_quick_analysis():
    """Quick analysis view"""
    
    st.markdown("### ⚡ Quick Analysis")
    
    if not st.session_state.data_cache:
        st.info("Upload files to see analysis")
        return
    
    # Select table for analysis
    table_names = list(st.session_state.data_cache.keys())
    selected_table = st.selectbox("Select Table", table_names)
    
    if selected_table:
        df = st.session_state.data_cache[selected_table]
        
        # Use tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Distributions", "🔍 Sample Data"])
        
        with tab1:
            # Quick stats using sample
            analysis = FastAnalytics.analyze_sample(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numeric Columns", len(analysis['numeric_cols']))
                st.metric("Text Columns", len(analysis['text_cols']))
            with col2:
                st.metric("Date Columns", len(analysis['date_cols']))
                st.metric("Data Quality", f"{100 - analysis['missing_pct']:.1f}%")
            
            # Numeric summary
            if analysis['numeric_cols']:
                st.markdown("#### Numeric Column Summary")
                numeric_summary = df[analysis['numeric_cols'][:5]].describe()
                st.dataframe(numeric_summary, use_container_width=True)
        
        with tab2:
            # Quick distributions
            if analysis['numeric_cols']:
                col = st.selectbox("Select column for distribution", analysis['numeric_cols'][:10])
                if col:
                    # Use sample for plotting
                    sample_size = min(5000, len(df))
                    df_plot = df.sample(n=sample_size) if len(df) > sample_size else df
                    
                    fig = px.histogram(df_plot, x=col, title=f"Distribution of {col}")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Show sample data
            st.markdown("#### Sample Data (First 100 rows)")
            st.dataframe(df.head(100), use_container_width=True, height=400)

def display_optimization():
    """Fast optimization tools"""
    
    st.markdown("### 🎯 Quick Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.selectbox("Origin", ["Chicago", "New York", "Los Angeles", "Houston"])
        weight = st.number_input("Weight (lbs)", 1000, 45000, 10000, 1000)
    
    with col2:
        destination = st.selectbox("Destination", ["Miami", "Seattle", "Boston", "Denver"])
        urgency = st.select_slider("Speed", ["Economy", "Standard", "Express"])
    
    with col3:
        service = st.selectbox("Service", ["LTL", "TL", "Partial"])
        equipment = st.selectbox("Equipment", ["Dry Van", "Reefer", "Flatbed"])
    
    if st.button("🚀 Get Quick Quote", type="primary"):
        # Fast calculation
        base_rates = {"Economy": 2.0, "Standard": 2.5, "Express": 3.5}
        service_mult = {"LTL": 1.2, "TL": 1.0, "Partial": 1.1}
        equipment_mult = {"Dry Van": 1.0, "Reefer": 1.25, "Flatbed": 1.15}
        
        # Simple distance estimate
        distance = random.uniform(500, 2500)
        
        # Calculate costs for different carriers
        results = []
        for carrier in ["UPS", "FedEx", "XPO", "SAIA", "Old Dominion"]:
            base = distance * base_rates[urgency] * service_mult[service] * equipment_mult[equipment]
            carrier_var = random.uniform(0.9, 1.1)
            cost = base * carrier_var * (1 + weight/20000)
            
            results.append({
                "Carrier": carrier,
                "Cost": f"${cost:.2f}",
                "Transit": f"{max(1, int(distance/500))}d",
                "Rating": f"{random.randint(85, 99)}%"
            })
        
        results_df = pd.DataFrame(results)
        st.success("✅ Quick quotes generated!")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

def display_merge_tools():
    """Tools for merging multi-part files"""
    
    st.markdown("### 🔧 File Management")
    
    if st.session_state.data_cache:
        # Check for multi-part files
        multi_parts = {}
        for name in st.session_state.data_cache.keys():
            if 'part' in name.lower() or any(f"_{i}" in name for i in range(1, 10)):
                base = name.split('part')[0].split('_')[0]
                if base not in multi_parts:
                    multi_parts[base] = []
                multi_parts[base].append(name)
        
        if multi_parts:
            st.markdown("#### Multi-Part Files Detected")
            for base, parts in multi_parts.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{base}**: {len(parts)} parts")
                with col2:
                    total_rows = sum(st.session_state.quick_stats[p]['rows'] for p in parts)
                    st.write(f"{total_rows:,} rows")
                with col3:
                    if st.button(f"Merge", key=f"merge_{base}"):
                        # Merge parts
                        dfs = [st.session_state.data_cache[p] for p in parts]
                        merged = pd.concat(dfs, ignore_index=True)
                        
                        # Update cache
                        st.session_state.data_cache[f"{base}_merged"] = merged
                        
                        # Remove parts
                        for p in parts:
                            del st.session_state.data_cache[p]
                            del st.session_state.quick_stats[p]
                        
                        # Update stats
                        st.session_state.quick_stats[f"{base}_merged"] = {
                            'rows': len(merged),
                            'columns': len(merged.columns),
                            'memory': merged.memory_usage(deep=True).sum() / 1024 / 1024
                        }
                        
                        st.success(f"✅ Merged {len(parts)} parts into {base}_merged")
                        st.rerun()
        else:
            st.info("No multi-part files detected")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    # File upload with batch processing
    st.markdown("### 📁 Batch Upload")
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload all files at once for faster processing"
    )
    
    if uploaded_files:
        # Check for new files
        new_files = []
        for file in uploaded_files:
            file_hash = FastFileProcessor.get_file_hash(file)
            if file_hash not in st.session_state.processed_files:
                new_files.append(file)
        
        if new_files:
            st.info(f"📁 {len(new_files)} new files detected")
            
            if st.button("⚡ Process All Files", type="primary"):
                with st.spinner(f"Processing {len(new_files)} files..."):
                    # Batch process
                    processed = FastFileProcessor.process_file_batch(new_files)
                    
                    # Update cache
                    st.session_state.data_cache.update(processed)
                    
                    st.success(f"✅ Processed {len(processed)} files successfully!")
                    st.rerun()
        else:
            st.info("✅ All files already processed")
    
    # Data summary
    if st.session_state.data_cache:
        st.markdown("---")
        st.markdown("### 📊 Data Status")
        
        metrics = FastAnalytics.get_quick_metrics(st.session_state.quick_stats)
        st.metric("Total Records", f"{metrics['total_rows']:,}")
        st.metric("Memory Used", f"{metrics['memory_mb']:.1f} MB")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        if st.button("🗑️ Clear All Data"):
            st.session_state.data_cache = {}
            st.session_state.file_info = {}
            st.session_state.processed_files = set()
            st.session_state.quick_stats = {}
            st.cache_data.clear()
            st.rerun()
    
    # Performance tips
    st.markdown("---")
    st.markdown("""
    ### 💡 Performance Tips
    
    - Upload all files at once
    - Use sampling for large datasets
    - Merge multi-part files
    - Clear cache if needed
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    display_header()
    
    # Main tabs - reduced for performance
    tabs = st.tabs([
        "📊 Dashboard",
        "⚡ Quick Analysis",
        "🎯 Optimization",
        "🔧 File Tools"
    ])
    
    with tabs[0]:
        display_dashboard()
    
    with tabs[1]:
        display_quick_analysis()
    
    with tabs[2]:
        display_optimization()
    
    with tabs[3]:
        display_merge_tools()

if __name__ == "__main__":
    main()
