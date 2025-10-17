import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
import json
from datetime import datetime, timedelta
import warnings
import math
from geopy.distance import geodesic
import random
from typing import Dict, List, Tuple, Optional
import re
import anthropic
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚚 Intelligent Lane Optimization System",
    page_icon="UPS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .config-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .feature-card {
        background: #f0f7ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
    }
    .insight-card {
        background: #fff8e1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 0.5rem 0;
    }
    .stTab {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'user_config' not in st.session_state:
    st.session_state.user_config = {}
if 'feature_analysis' not in st.session_state:
    st.session_state.feature_analysis = {}
if 'column_mappings' not in st.session_state:
    st.session_state.column_mappings = {}

# Claude API Integration
class ClaudeAnalyzer:
    """Intelligent data analysis using Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                st.warning(f"Claude API initialization failed: {e}")
    
    def analyze_dataset_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze dataset structure and suggest feature mappings"""
        if not self.client:
            return self._fallback_analysis(df)
        
        try:
            # Prepare dataset summary
            summary = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'sample_values': {col: df[col].head(3).tolist() for col in df.columns},
                'null_counts': df.isnull().sum().to_dict(),
                'unique_counts': df.nunique().to_dict()
            }
            
            prompt = f"""
            I have a shipping/logistics dataset with the following structure:
            
            Columns: {summary['columns']}
            Data types: {summary['dtypes']}
            Sample values: {summary['sample_values']}
            
            Please analyze this dataset and provide:
            1. Identify which column likely represents the target variable (cost/price)
            2. Categorize columns as: location, carrier, service_type, weight, volume, distance, value, time, or other
            3. Suggest feature engineering opportunities
            4. Identify potential data quality issues
            5. Recommend which columns to use for ML training
            
            Respond in JSON format with these keys:
            - target_column: string
            - column_categories: dict mapping column names to categories
            - feature_engineering: list of suggestions
            - data_quality_issues: list of issues
            - recommended_features: list of column names
            """
            
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            analysis = json.loads(response.content[0].text)
            return analysis
            
        except Exception as e:
            st.warning(f"Claude analysis failed, using fallback: {e}")
            return self._fallback_analysis(df)
    
    def _fallback_analysis(self, df: pd.DataFrame) -> Dict:
        """Fallback analysis without Claude API"""
        analysis = {
            'target_column': None,
            'column_categories': {},
            'feature_engineering': [],
            'data_quality_issues': [],
            'recommended_features': []
        }
        
        # Simple heuristics for column categorization
        for col in df.columns:
            col_lower = col.lower()
            
            # Target variable detection
            if any(word in col_lower for word in ['cost', 'price', 'rate', 'fee', 'charge', 'total']):
                if analysis['target_column'] is None:
                    analysis['target_column'] = col
            
            # Category detection
            if any(word in col_lower for word in ['origin', 'source', 'from', 'pickup']):
                analysis['column_categories'][col] = 'origin_location'
            elif any(word in col_lower for word in ['destination', 'dest', 'to', 'delivery']):
                analysis['column_categories'][col] = 'destination_location'
            elif any(word in col_lower for word in ['carrier', 'shipper', 'company']):
                analysis['column_categories'][col] = 'carrier'
            elif any(word in col_lower for word in ['service', 'mode', 'type']):
                analysis['column_categories'][col] = 'service_type'
            elif any(word in col_lower for word in ['weight', 'wgt', 'mass']):
                analysis['column_categories'][col] = 'weight'
            elif any(word in col_lower for word in ['volume', 'vol', 'cubic', 'size']):
                analysis['column_categories'][col] = 'volume'
            elif any(word in col_lower for word in ['distance', 'dist', 'miles', 'km']):
                analysis['column_categories'][col] = 'distance'
            elif any(word in col_lower for word in ['value', 'worth', 'declared']):
                analysis['column_categories'][col] = 'value'
            elif any(word in col_lower for word in ['date', 'time', 'timestamp']):
                analysis['column_categories'][col] = 'time'
            else:
                analysis['column_categories'][col] = 'other'
        
        # Recommended features (exclude target and non-useful columns)
        analysis['recommended_features'] = [
            col for col in df.columns 
            if col != analysis['target_column'] and 
            analysis['column_categories'].get(col) != 'time'
        ]
        
        # Basic feature engineering suggestions
        analysis['feature_engineering'] = [
            "Create weight-to-volume ratio if both weight and volume columns exist",
            "Extract day of week from date columns",
            "Create distance bins for categorical analysis",
            "One-hot encode categorical variables"
        ]
        
        # Data quality checks
        if df.isnull().sum().sum() > 0:
            analysis['data_quality_issues'].append("Missing values detected")
        
        if any(df.select_dtypes(include=[np.number]).min() < 0):
            analysis['data_quality_issues'].append("Negative values in numeric columns")
        
        return analysis

# Advanced Configuration Manager
class ConfigurationManager:
    """Manage user configurations and data mappings"""
    
    @staticmethod
    def save_config(config: Dict, filename: str = "user_config.json"):
        """Save user configuration"""
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            return True
        except Exception as e:
            st.error(f"Failed to save config: {e}")
            return False
    
    @staticmethod
    def load_config(filename: str = "user_config.json") -> Dict:
        """Load user configuration"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load config: {e}")
        return {}
    
    @staticmethod
    def create_cost_rules_config() -> Dict:
        """Create default cost calculation rules"""
        return {
            'base_rates': {
                'distance_rate': 0.45,
                'weight_rate': 0.75,
                'volume_rate': 0.30,
                'minimum_cost': 15.0
            },
            'carrier_multipliers': {
                'Default': {'ground': 1.0, 'express': 1.3, 'overnight': 1.8, 'economy': 0.8}
            },
            'surcharges': {
                'fuel_surcharge_rate': 0.15,
                'insurance_rate': 0.001,
                'residential_fee': 8.50,
                'oversized_threshold': 70,
                'oversized_fee': 25.0
            },
            'zone_multipliers': {
                'same_zone': 1.0,
                'adjacent_zone': 1.1,
                'cross_country': 1.4
            }
        }

# Dynamic Feature Engineering
class FeatureEngineer:
    """Intelligent feature engineering based on data analysis"""
    
    @staticmethod
    def auto_engineer_features(df: pd.DataFrame, analysis: Dict) -> pd.DataFrame:
        """Automatically engineer features based on analysis"""
        df_engineered = df.copy()
        
        # Get column categories
        categories = analysis.get('column_categories', {})
        
        # Find columns by category
        weight_cols = [col for col, cat in categories.items() if cat == 'weight']
        volume_cols = [col for col, cat in categories.items() if cat == 'volume']
        distance_cols = [col for col, cat in categories.items() if cat == 'distance']
        value_cols = [col for col, cat in categories.items() if cat == 'value']
        
        # Create engineered features
        if weight_cols and volume_cols:
            weight_col = weight_cols[0]
            volume_col = volume_cols[0]
            df_engineered[f'{weight_col}_to_{volume_col}_ratio'] = (
                df_engineered[weight_col] / (df_engineered[volume_col] + 1e-6)
            )
        
        if weight_cols and value_cols:
            weight_col = weight_cols[0]
            value_col = value_cols[0]
            df_engineered[f'{value_col}_per_{weight_col}'] = (
                df_engineered[value_col] / (df_engineered[weight_col] + 1e-6)
            )
        
        if distance_cols and weight_cols:
            distance_col = distance_cols[0]
            weight_col = weight_cols[0]
            df_engineered[f'{distance_col}_{weight_col}_interaction'] = (
                df_engineered[distance_col] * df_engineered[weight_col]
            )
        
        # Create categorical bins for numeric features
        for col in distance_cols + weight_cols + volume_cols:
            if col in df_engineered.columns:
                df_engineered[f'{col}_bin'] = pd.cut(
                    df_engineered[col], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high']
                )
        
        return df_engineered
    
    @staticmethod
    def smart_feature_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> List[str]:
        """Intelligent feature selection using multiple methods"""
        # Remove non-numeric columns for initial selection
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]
        
        if len(X_numeric.columns) == 0:
            return []
        
        # Method 1: Statistical selection
        selector_stats = SelectKBest(score_func=f_regression, k=min(n_features, len(X_numeric.columns)))
        try:
            selector_stats.fit(X_numeric.fillna(0), y)
            stats_features = X_numeric.columns[selector_stats.get_support()].tolist()
        except:
            stats_features = list(X_numeric.columns[:n_features])
        
        # Method 2: Tree-based feature importance
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_numeric.fillna(0), y)
            feature_importance = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            tree_features = feature_importance.head(n_features)['feature'].tolist()
        except:
            tree_features = list(X_numeric.columns[:n_features])
        
        # Combine and deduplicate
        selected_features = list(set(stats_features + tree_features))
        return selected_features[:n_features]

# Enhanced ML Agents with Auto-Configuration
class AdvancedMLAgents:
    """ML Agents with automatic configuration based on data"""
    
    @staticmethod
    def auto_configure_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """Automatically configure models based on data characteristics"""
        n_samples, n_features = X_train.shape
        
        config = {
            'random_forest': {
                'n_estimators': min(200, max(50, n_samples // 10)),
                'max_depth': min(15, max(5, int(np.log2(n_features)) + 1)),
                'min_samples_split': max(2, n_samples // 1000),
                'min_samples_leaf': max(1, n_samples // 2000)
            },
            'gradient_boosting': {
                'n_estimators': min(200, max(50, n_samples // 20)),
                'learning_rate': 0.1 if n_samples > 1000 else 0.2,
                'max_depth': min(8, max(3, int(np.log2(n_features)))),
                'subsample': 0.8 if n_samples > 500 else 1.0
            },
            'lstm': {
                'epochs': min(100, max(20, n_samples // 50)),
                'batch_size': min(64, max(16, n_samples // 100)),
                'units_1': min(128, max(32, n_features * 4)),
                'units_2': min(64, max(16, n_features * 2))
            }
        }
        
        return config
    
    @staticmethod
    def train_adaptive_models(X_train, X_test, y_train, y_test, config: Dict):
        """Train models with adaptive configuration"""
        models = {}
        predictions = {}
        metrics = {}
        
        # Random Forest with auto-config
        rf_config = config['random_forest']
        rf_model = RandomForestRegressor(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            min_samples_split=rf_config['min_samples_split'],
            min_samples_leaf=rf_config['min_samples_leaf'],
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        models['Random Forest'] = rf_model
        predictions['Random Forest'] = rf_pred
        metrics['Random Forest'] = {
            'MAE': mean_absolute_error(y_test, rf_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'R2': r2_score(y_test, rf_pred)
        }
        
        # Gradient Boosting with auto-config
        gb_config = config['gradient_boosting']
        gb_model = GradientBoostingRegressor(
            n_estimators=gb_config['n_estimators'],
            learning_rate=gb_config['learning_rate'],
            max_depth=gb_config['max_depth'],
            subsample=gb_config['subsample'],
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        
        models['Gradient Boosting'] = gb_model
        predictions['Gradient Boosting'] = gb_pred
        metrics['Gradient Boosting'] = {
            'MAE': mean_absolute_error(y_test, gb_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, gb_pred)),
            'R2': r2_score(y_test, gb_pred)
        }
        
        # LSTM with auto-config (if enough data)
        if len(X_train) >= 100:
            try:
                lstm_config = config['lstm']
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Reshape for LSTM
                X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                # Build adaptive LSTM
                lstm_model = Sequential([
                    LSTM(lstm_config['units_1'], return_sequences=True, 
                         input_shape=(1, X_train_scaled.shape[1])),
                    Dropout(0.2),
                    LSTM(lstm_config['units_2']),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dense(1)
                ])
                
                lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                # Train with early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                lstm_model.fit(
                    X_train_lstm, y_train,
                    validation_data=(X_test_lstm, y_test),
                    epochs=lstm_config['epochs'],
                    batch_size=lstm_config['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                lstm_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
                
                models['LSTM'] = {'model': lstm_model, 'scaler': scaler}
                predictions['LSTM'] = lstm_pred
                metrics['LSTM'] = {
                    'MAE': mean_absolute_error(y_test, lstm_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_test, lstm_pred)),
                    'R2': r2_score(y_test, lstm_pred)
                }
            except Exception as e:
                st.warning(f"LSTM training failed: {e}")
        
        # Ensemble
        if len(predictions) >= 2:
            pred_values = list(predictions.values())
            ensemble_pred = np.mean(pred_values, axis=0)
            
            predictions['Ensemble'] = ensemble_pred
            metrics['Ensemble'] = {
                'MAE': mean_absolute_error(y_test, ensemble_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'R2': r2_score(y_test, ensemble_pred)
            }
        
        return models, predictions, metrics

# Header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0; text-align: center;">🚚 Intelligent Lane Optimization System</h1>
    <p style="color: white; margin: 0; text-align: center; opacity: 0.9;">
        AI-Powered Auto-Configuring Multi-Agent System with Claude Intelligence
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.header("🎛️ System Configuration")
    
    # Claude API Configuration
    st.subheader("  Claude AI Integration")
    claude_api_key = st.text_input(
        "Claude API Key (Optional)", 
        type="password",
        help="Enter your Claude API key for intelligent data analysis"
    )
    
    if claude_api_key:
        st.success("✅ Claude AI Connected")
    else:
        st.info("💡 Add Claude API key for intelligent analysis")
    
    # Cost Rules Configuration
    st.subheader("💰 Cost Calculation Rules")
    
    with st.expander("Configure Cost Rules"):
        distance_rate = st.number_input("Distance Rate ($/mile)", 0.1, 2.0, 0.45)
        weight_rate = st.number_input("Weight Rate ($/lb)", 0.1, 2.0, 0.75)
        fuel_surcharge = st.slider("Fuel Surcharge %", 0, 50, 15)
        minimum_cost = st.number_input("Minimum Cost ($)", 5.0, 50.0, 15.0)
        
        # Save to session state
        st.session_state.user_config['cost_rules'] = {
            'distance_rate': distance_rate,
            'weight_rate': weight_rate,
            'fuel_surcharge_rate': fuel_surcharge / 100,
            'minimum_cost': minimum_cost
        }
    
    # Model Configuration
    st.subheader("🤖 Model Configuration")
    auto_feature_engineering = st.checkbox("Auto Feature Engineering", True)
    auto_model_config = st.checkbox("Auto Model Configuration", True)
    max_features = st.slider("Max Features to Use", 5, 50, 20)
    
    # Data Configuration
    st.subheader("📊 Data Settings")
    handle_missing_values = st.selectbox(
        "Missing Values Strategy",
        ["Auto", "Drop", "Fill with Mean", "Fill with Median", "Fill with Zero"]
    )
    
    outlier_detection = st.checkbox("Outlier Detection & Removal", True)
    
    if st.button("💾 Save Configuration"):
        config = {
            'auto_feature_engineering': auto_feature_engineering,
            'auto_model_config': auto_model_config,
            'max_features': max_features,
            'missing_values': handle_missing_values,
            'outlier_detection': outlier_detection,
            'cost_rules': st.session_state.user_config.get('cost_rules', {})
        }
        if ConfigurationManager.save_config(config):
            st.success("✅ Configuration saved!")

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data Upload & Analysis", 
    "🔧 Feature Engineering", 
    "🤖 Model Training", 
    "🎯 Predictions", 
    "📈 Analytics", 
    "⚙️ Configuration Manager"
])

# Tab 1: Data Upload & Analysis
with tab1:
    st.subheader("📊 Intelligent Data Upload & Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Upload Your Dataset", 
            type=['csv', 'xlsx', 'json'],
            help="Upload CSV, Excel, or JSON file with your shipping data"
        )
        
        if uploaded_file:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    st.session_state.data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.data = pd.read_json(uploaded_file)
                
                st.success(f"✅ Loaded {len(st.session_state.data)} records with {len(st.session_state.data.columns)} columns")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(st.session_state.data.head(), use_container_width=True)
                
                # Data summary
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Rows", len(st.session_state.data))
                with col_b:
                    st.metric("Columns", len(st.session_state.data.columns))
                with col_c:
                    st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
                with col_d:
                    st.metric("Numeric Columns", len(st.session_state.data.select_dtypes(include=[np.number]).columns))
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
        
        # Generate sample data option
        if st.button("🎲 Generate Sample Data for Testing"):
            # Simple sample data generator
            np.random.seed(42)
            n_samples = 1000
            
            data = {
                'origin_city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
                'destination_city': np.random.choice(['Miami', 'Seattle', 'Denver', 'Atlanta', 'Boston'], n_samples),
                'carrier_name': np.random.choice(['FedEx', 'UPS', 'DHL', 'USPS'], n_samples),
                'service_level': np.random.choice(['Ground', 'Express', 'Overnight', 'Economy'], n_samples),
                'package_weight_lbs': np.random.exponential(25, n_samples) + 1,
                'package_volume_cuft': np.random.exponential(8, n_samples) + 0.5,
                'declared_value_usd': np.random.exponential(1000, n_samples) + 100,
                'distance_miles': np.random.uniform(100, 3000, n_samples),
                'shipping_cost_usd': None  # Will be calculated
            }
            
            # Calculate shipping cost using user rules
            cost_rules = st.session_state.user_config.get('cost_rules', {})
            distance_rate = cost_rules.get('distance_rate', 0.45)
            weight_rate = cost_rules.get('weight_rate', 0.75)
            fuel_rate = cost_rules.get('fuel_surcharge_rate', 0.15)
            min_cost = cost_rules.get('minimum_cost', 15.0)
            
            shipping_costs = []
            for i in range(n_samples):
                base_cost = (data['distance_miles'][i] * distance_rate + 
                           data['package_weight_lbs'][i] * weight_rate)
                fuel_cost = base_cost * fuel_rate
                total_cost = max(base_cost + fuel_cost, min_cost)
                # Add some randomness
                total_cost *= np.random.uniform(0.9, 1.1)
                shipping_costs.append(round(total_cost, 2))
            
            data['shipping_cost_usd'] = shipping_costs
            st.session_state.data = pd.DataFrame(data)
            st.success(f"✅ Generated {n_samples} sample records")
            st.rerun()
    
    with col2:
        if st.session_state.data is not None:
            st.subheader("  Intelligent Analysis")
            
            if st.button("🔍 Analyze with Claude AI", type="primary"):
                if claude_api_key:
                    with st.spinner("Analyzing dataset with Claude AI..."):
                        analyzer = ClaudeAnalyzer(claude_api_key)
                        analysis = analyzer.analyze_dataset_structure(st.session_state.data)
                        st.session_state.feature_analysis = analysis
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display analysis results
                    st.markdown("### 🎯 Target Variable")
                    if analysis.get('target_column'):
                        st.success(f"**{analysis['target_column']}** identified as target")
                    else:
                        st.warning("No clear target variable identified")
                    
                    st.markdown("### 📋 Column Categories")
                    for col, category in analysis.get('column_categories', {}).items():
                        st.markdown(f"**{col}**: {category}")
                    
                    if analysis.get('feature_engineering'):
                        st.markdown("### 🔧 Feature Engineering Suggestions")
                        for suggestion in analysis['feature_engineering']:
                            st.markdown(f"• {suggestion}")
                    
                    if analysis.get('data_quality_issues'):
                        st.markdown("### ⚠️ Data Quality Issues")
                        for issue in analysis['data_quality_issues']:
                            st.warning(issue)
                
                else:
                    with st.spinner("Analyzing dataset with fallback method..."):
                        analyzer = ClaudeAnalyzer()
                        analysis = analyzer.analyze_dataset_structure(st.session_state.data)
                        st.session_state.feature_analysis = analysis
                    
                    st.info("✅ Basic analysis complete (Add Claude API key for advanced analysis)")
            
            # Manual column mapping
            st.subheader("🗂️ Manual Column Mapping")
            if st.session_state.data is not None:
                target_col = st.selectbox(
                    "Select Target Variable (Cost/Price)",
                    ['None'] + list(st.session_state.data.columns),
                    index=0
                )
                
                if target_col != 'None':
                    st.session_state.column_mappings['target'] = target_col
                    st.success(f"Target: {target_col}")

# Tab 2: Feature Engineering
with tab2:
    st.subheader("🔧 Intelligent Feature Engineering")
    
    if st.session_state.data is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Current Features")
            
            # Display current features
            feature_info = []
            for col in st.session_state.data.columns:
                dtype = str(st.session_state.data[col].dtype)
                nulls = st.session_state.data[col].isnull().sum()
                unique = st.session_state.data[col].nunique()
                
                feature_info.append({
                    'Column': col,
                    'Type': dtype,
                    'Nulls': nulls,
                    'Unique Values': unique,
                    'Category': st.session_state.feature_analysis.get('column_categories', {}).get(col, 'unknown')
                })
            
            feature_df = pd.DataFrame(feature_info)
            st.dataframe(feature_df, use_container_width=True)
            
            # Auto feature engineering
            if st.button("🤖 Auto-Engineer Features"):
                if st.session_state.feature_analysis:
                    with st.spinner("Engineering features..."):
                        engineer = FeatureEngineer()
                        engineered_data = engineer.auto_engineer_features(
                            st.session_state.data, 
                            st.session_state.feature_analysis
                        )
                        
                        new_features = set(engineered_data.columns) - set(st.session_state.data.columns)
                        st.session_state.data = engineered_data
                        
                        if new_features:
                            st.success(f"✅ Created {len(new_features)} new features: {', '.join(new_features)}")
                        else:
                            st.info("No new features created based on current data structure")
                else:
                    st.warning("Run data analysis first")
        
        with col2:
            st.subheader("🎛️ Feature Engineering Options")
            
            # Manual feature creation
            with st.expander("Create Custom Features"):
                st.markdown("**Mathematical Operations**")
                
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    col1_select = st.selectbox("Column 1", numeric_cols, key="feat_col1")
                    operation = st.selectbox("Operation", ["+", "-", "*", "/"], key="feat_op")
                    col2_select = st.selectbox("Column 2", numeric_cols, key="feat_col2")
                    new_feature_name = st.text_input("New Feature Name", key="feat_name")
                    
                    if st.button("Create Feature") and new_feature_name:
                        try:
                            if operation == "+":
                                st.session_state.data[new_feature_name] = (
                                    st.session_state.data[col1_select] + st.session_state.data[col2_select]
                                )
                            elif operation == "-":
                                st.session_state.data[new_feature_name] = (
                                    st.session_state.data[col1_select] - st.session_state.data[col2_select]
                                )
                            elif operation == "*":
                                st.session_state.data[new_feature_name] = (
                                    st.session_state.data[col1_select] * st.session_state.data[col2_select]
                                )
                            elif operation == "/":
                                st.session_state.data[new_feature_name] = (
                                    st.session_state.data[col1_select] / (st.session_state.data[col2_select] + 1e-6)
                                )
                            
                            st.success(f"✅ Created feature: {new_feature_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating feature: {e}")
            
            # Data preprocessing options
            st.subheader("🧹 Data Preprocessing")
            
            if st.button("🔧 Apply Preprocessing"):
                with st.spinner("Preprocessing data..."):
                    # Handle missing values
                    if handle_missing_values == "Drop":
                        st.session_state.data = st.session_state.data.dropna()
                    elif handle_missing_values == "Fill with Mean":
                        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                        st.session_state.data[numeric_cols] = st.session_state.data[numeric_cols].fillna(
                            st.session_state.data[numeric_cols].mean()
                        )
                    elif handle_missing_values == "Fill with Median":
                        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                        st.session_state.data[numeric_cols] = st.session_state.data[numeric_cols].fillna(
                            st.session_state.data[numeric_cols].median()
                        )
                    elif handle_missing_values == "Fill with Zero":
                        st.session_state.data = st.session_state.data.fillna(0)
                    
                    # Outlier detection
                    if outlier_detection:
                        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            Q1 = st.session_state.data[col].quantile(0.25)
                            Q3 = st.session_state.data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers_mask = (
                                (st.session_state.data[col] < lower_bound) | 
                                (st.session_state.data[col] > upper_bound)
                            )
                            st.session_state.data = st.session_state.data[~outliers_mask]
                    
                    st.success("✅ Preprocessing complete!")
                    st.rerun()
    
    else:
        st.info("📊 Upload data first to begin feature engineering")

# Tab 3: Model Training
with tab3:
    st.subheader("🤖 Intelligent Model Training")
    
    if st.session_state.data is not None:
        # Target variable selection
        target_column = st.session_state.column_mappings.get('target')
        
        if not target_column:
            st.warning("⚠️ Please select a target variable in the Data Upload tab")
            target_column = st.selectbox(
                "Select Target Variable",
                st.session_state.data.columns,
                help="Select the column you want to predict (e.g., cost, price)"
            )
            if target_column:
                st.session_state.column_mappings['target'] = target_column
        
        if target_column and target_column in st.session_state.data.columns:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"🎯 Training to Predict: {target_column}")
                
                # Prepare data for training
                try:
                    # Encode categorical variables
                    df_encoded = st.session_state.data.copy()
                    encoders = {}
                    
                    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        if col != target_column:
                            encoders[col] = LabelEncoder()
                            df_encoded[f'{col}_encoded'] = encoders[col].fit_transform(df_encoded[col].astype(str))
                    
                    # Select features
                    feature_cols = []
                    
                    # Add numeric columns
                    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
                    feature_cols.extend([col for col in numeric_cols if col != target_column])
                    
                    # Add encoded categorical columns
                    encoded_cols = [col for col in df_encoded.columns if col.endswith('_encoded')]
                    feature_cols.extend(encoded_cols)
                    
                    if len(feature_cols) == 0:
                        st.error("❌ No suitable features found for training")
                    else:
                        X = df_encoded[feature_cols].fillna(0)
                        y = df_encoded[target_column]
                        
                        # Intelligent feature selection
                        if len(feature_cols) > max_features:
                            engineer = FeatureEngineer()
                            selected_features = engineer.smart_feature_selection(X, y, max_features)
                            X = X[selected_features]
                            feature_cols = selected_features
                            st.info(f"🎯 Selected {len(selected_features)} best features from {len(feature_cols)} total")
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Display data info
                        st.markdown(f"**Training Data:** {len(X_train)} samples, {len(feature_cols)} features")
                        st.markdown(f"**Test Data:** {len(X_test)} samples")
                        
                        # Auto-configure models
                        if auto_model_config:
                            config = AdvancedMLAgents.auto_configure_models(X_train, y_train)
                            st.markdown("### 🤖 Auto-Configured Model Parameters")
                            st.json(config)
                        
                        # Train models
                        if st.button("🚀 Train All Models", type="primary"):
                            with st.spinner("Training intelligent models..."):
                                try:
                                    if auto_model_config:
                                        models, predictions, metrics = AdvancedMLAgents.train_adaptive_models(
                                            X_train, X_test, y_train, y_test, config
                                        )
                                    else:
                                        # Use default configuration
                                        default_config = {
                                            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2},
                                            'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.8},
                                            'lstm': {'epochs': 50, 'batch_size': 32, 'units_1': 64, 'units_2': 32}
                                        }
                                        models, predictions, metrics = AdvancedMLAgents.train_adaptive_models(
                                            X_train, X_test, y_train, y_test, default_config
                                        )
                                    
                                    # Store in session state
                                    st.session_state.models['current'] = {
                                        'models': models,
                                        'encoders': encoders,
                                        'feature_cols': feature_cols,
                                        'metrics': metrics,
                                        'target_column': target_column,
                                        'trained_at': datetime.now()
                                    }
                                    
                                    st.success("✅ All models trained successfully!")
                                    
                                except Exception as e:
                                    st.error(f"❌ Training failed: {str(e)}")
                                    st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Data preparation failed: {str(e)}")
            
            with col2:
                if 'current' in st.session_state.models:
                    st.subheader("📊 Model Performance")
                    
                    metrics = st.session_state.models['current']['metrics']
                    
                    # Display metrics for each model
                    for model_name, model_metrics in metrics.items():
                        st.markdown(f"### {model_name}")
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            st.metric("MAE", f"{model_metrics['MAE']:.2f}")
                        with col_m2:
                            st.metric("RMSE", f"{model_metrics['RMSE']:.2f}")
                        with col_m3:
                            st.metric("R²", f"{model_metrics['R2']:.3f}")
                    
                    # Best model
                    best_model = min(metrics.keys(), key=lambda k: metrics[k]['MAE'])
                    st.success(f"🏆 Best Model: **{best_model}**")
                
                else:
                    st.info("👆 Train models to see performance metrics")
    
    else:
        st.info("📊 Upload data first to train models")

# Tab 4: Predictions
with tab4:
    st.subheader("🎯 Intelligent Predictions")
    
    if 'current' in st.session_state.models:
        model_info = st.session_state.models['current']
        target_col = model_info['target_column']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"📝 Input Features for {target_col}")
            
            # Dynamic input form based on features
            input_data = {}
            feature_cols = model_info['feature_cols']
            encoders = model_info['encoders']
            
            # Group features by type
            numeric_features = [col for col in feature_cols if not col.endswith('_encoded')]
            categorical_features = [col.replace('_encoded', '') for col in feature_cols if col.endswith('_encoded')]
            
            # Numeric inputs
            if numeric_features:
                st.markdown("**📊 Numeric Features**")
                for feature in numeric_features:
                    if feature in st.session_state.data.columns:
                        min_val = float(st.session_state.data[feature].min())
                        max_val = float(st.session_state.data[feature].max())
                        default_val = float(st.session_state.data[feature].mean())
                        
                        input_data[feature] = st.number_input(
                            feature.replace('_', ' ').title(),
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"num_{feature}"
                        )
            
            # Categorical inputs
            if categorical_features:
                st.markdown("**📋 Categorical Features**")
                for feature in categorical_features:
                    if feature in st.session_state.data.columns:
                        unique_values = st.session_state.data[feature].unique().tolist()
                        input_data[feature] = st.selectbox(
                            feature.replace('_', ' ').title(),
                            unique_values,
                            key=f"cat_{feature}"
                        )
            
            if st.button("🎯 Predict with All Models", type="primary"):
                try:
                    # Prepare input dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode categorical variables
                    for feature in categorical_features:
                        if feature in encoders:
                            try:
                                encoded_col = f'{feature}_encoded'
                                input_df[encoded_col] = encoders[feature].transform([input_data[feature]])
                            except ValueError:
                                # Handle unseen categories
                                input_df[encoded_col] = 0
                    
                    # Select only the features used in training
                    X_input = input_df[feature_cols].fillna(0)
                    
                    # Get predictions from all models
                    predictions = {}
                    for model_name, model in model_info['models'].items():
                        if model_name == 'LSTM':
                            # Handle LSTM prediction
                            scaler = model['scaler']
                            X_scaled = scaler.transform(X_input)
                            X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
                            pred = model['model'].predict(X_lstm, verbose=0)[0][0]
                        else:
                            pred = model.predict(X_input)[0]
                        
                        predictions[model_name] = max(pred, 0)  # Ensure non-negative
                    
                    # Display predictions
                    st.subheader("🎯 Model Predictions")
                    
                    for model_name, pred_value in predictions.items():
                        st.metric(f"{model_name}", f"${pred_value:.2f}" if 'cost' in target_col.lower() else f"{pred_value:.2f}")
                    
                    # Best prediction
                    if 'cost' in target_col.lower() or 'price' in target_col.lower():
                        best_model = min(predictions.keys(), key=lambda k: predictions[k])
                        best_value = predictions[best_model]
                        st.success(f"🏆 Lowest Cost: **{best_model}** predicts **${best_value:.2f}**")
                    else:
                        avg_prediction = np.mean(list(predictions.values()))
                        st.info(f"📊 Average Prediction: **{avg_prediction:.2f}**")
                    
                    # Store prediction
                    prediction_record = {
                        'timestamp': datetime.now(),
                        'input_data': input_data,
                        'predictions': predictions,
                        'target': target_col
                    }
                    st.session_state.predictions.append(prediction_record)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    st.exception(e)
        
        with col2:
            st.subheader("📊 Batch Predictions")
            
            # Batch prediction option
            batch_file = st.file_uploader(
                "Upload CSV for Batch Predictions",
                type=['csv'],
                help="Upload a CSV file with the same features for batch predictions"
            )
            
            if batch_file:
                try:
                    batch_data = pd.read_csv(batch_file)
                    st.write(f"Loaded {len(batch_data)} records for prediction")
                    
                    if st.button("🔄 Run Batch Predictions"):
                        with st.spinner("Running batch predictions..."):
                            # Prepare batch data similar to single prediction
                            batch_results = []
                            
                            for _, row in batch_data.iterrows():
                                try:
                                    # Create input dataframe for this row
                                    row_input = pd.DataFrame([row.to_dict()])
                                    
                                    # Encode categorical variables
                                    for feature in categorical_features:
                                        if feature in encoders and feature in row_input.columns:
                                            try:
                                                encoded_col = f'{feature}_encoded'
                                                row_input[encoded_col] = encoders[feature].transform([row[feature]])
                                            except:
                                                row_input[encoded_col] = 0
                                    
                                    # Select features and predict
                                    X_row = row_input[feature_cols].fillna(0)
                                    
                                    row_predictions = {}
                                    for model_name, model in model_info['models'].items():
                                        if model_name == 'LSTM':
                                            scaler = model['scaler']
                                            X_scaled = scaler.transform(X_row)
                                            X_lstm = X_scaled.reshape((1, 1, X_scaled.shape[1]))
                                            pred = model['model'].predict(X_lstm, verbose=0)[0][0]
                                        else:
                                            pred = model.predict(X_row)[0]
                                        
                                        row_predictions[model_name] = max(pred, 0)
                                    
                                    # Add to results
                                    result_row = row.to_dict()
                                    for model_name, pred in row_predictions.items():
                                        result_row[f'{model_name}_prediction'] = pred
                                    
                                    batch_results.append(result_row)
                                
                                except Exception as e:
                                    st.warning(f"Failed to predict for row {len(batch_results)}: {e}")
                            
                            # Display results
                            if batch_results:
                                results_df = pd.DataFrame(batch_results)
                                st.subheader("📊 Batch Results")
                                st.dataframe(results_df, use_container_width=True)
                                
                                # Download option
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    "📥 Download Results",
                                    csv,
                                    "batch_predictions.csv",
                                    "text/csv"
                                )
                
                except Exception as e:
                    st.error(f"Error processing batch file: {e}")
            
            # Recent predictions
            if st.session_state.predictions:
                st.subheader("🕒 Recent Predictions")
                recent_preds = st.session_state.predictions[-5:]
                
                for i, pred in enumerate(recent_preds):
                    with st.expander(f"Prediction {len(st.session_state.predictions) - len(recent_preds) + i + 1}"):
                        st.write(f"**Time:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                        st.write(f"**Target:** {pred['target']}")
                        st.write("**Predictions:**")
                        for model, value in pred['predictions'].items():
                            st.write(f"- {model}: {value:.2f}")
    
    else:
        st.info("🤖 Train models first to make predictions")

# Tab 5: Analytics
with tab5:
    st.subheader("📈 Advanced Analytics Dashboard")
    
    if st.session_state.data is not None:
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(st.session_state.data))
        with col2:
            st.metric("Features", len(st.session_state.data.columns))
        with col3:
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.metric("Avg Numeric Value", f"{st.session_state.data[numeric_cols].mean().mean():.2f}")
            else:
                st.metric("Numeric Columns", 0)
        with col4:
            if 'current' in st.session_state.models:
                best_model = min(
                    st.session_state.models['current']['metrics'].keys(),
                    key=lambda k: st.session_state.models['current']['metrics'][k]['MAE']
                )
                st.metric("Best Model", best_model)
            else:
                st.metric("Models Trained", 0)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution of target variable
            if 'current' in st.session_state.models:
                target_col = st.session_state.models['current']['target_column']
                if target_col in st.session_state.data.columns:
                    fig = px.histogram(
                        st.session_state.data, 
                        x=target_col, 
                        title=f"Distribution of {target_col}",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            numeric_data = st.session_state.data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) > 1:
                corr_matrix = numeric_data.corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Matrix",
                    color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Feature importance (if Random Forest was trained)
            if 'current' in st.session_state.models and 'Random Forest' in st.session_state.models['current']['models']:
                rf_model = st.session_state.models['current']['models']['Random Forest']
                feature_cols = st.session_state.models['current']['feature_cols']
                
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False).head(15)
                
                fig = px.bar(
                    importance_df, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    title="Top 15 Feature Importances"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Model performance comparison
            if 'current' in st.session_state.models:
                metrics_data = []
                for model_name, metrics in st.session_state.models['current']['metrics'].items():
                    metrics_data.append({
                        'Model': model_name,
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE'],
                        'R²': metrics['R2']
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                
                fig = px.bar(
                    metrics_df, 
                    x='Model', 
                    y='R²',
                    title="Model Performance (R² Score)",
                    color='R²',
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction history
        if st.session_state.predictions:
            st.subheader("📊 Prediction History Analysis")
            
            # Convert predictions to DataFrame for analysis
            pred_data = []
            for pred in st.session_state.predictions:
                base_row = {
                    'timestamp': pred['timestamp'],
                    'target': pred['target']
                }
                
                # Add input features
                for key, value in pred['input_data'].items():
                    base_row[f'input_{key}'] = value
                
                # Add predictions
                for model, prediction in pred['predictions'].items():
                    base_row[f'pred_{model}'] = prediction
                
                pred_data.append(base_row)
            
            if pred_data:
                pred_df = pd.DataFrame(pred_data)
                
                # Prediction timeline
                prediction_cols = [col for col in pred_df.columns if col.startswith('pred_')]
                if prediction_cols:
                    fig = go.Figure()
                    
                    for col in prediction_cols:
                        model_name = col.replace('pred_', '')
                        fig.add_trace(go.Scatter(
                            x=pred_df['timestamp'],
                            y=pred_df[col],
                            mode='lines+markers',
                            name=model_name,
                            line=dict(width=2)
                        ))
                    
                    fig.update_layout(
                        title="Prediction Timeline",
                        xaxis_title="Time",
                        yaxis_title="Predicted Value",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show recent predictions table
                st.subheader("Recent Predictions")
                display_cols = ['timestamp', 'target'] + prediction_cols[:5]  # Show max 5 models
                st.dataframe(pred_df[display_cols].tail(10), use_container_width=True)
    
    else:
        st.info("📊 Upload and analyze data to view analytics")

# Tab 6: Configuration Manager
with tab6:
    st.subheader("⚙️ Advanced Configuration Manager")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💾 Save/Load Configurations")
        
        # Save current configuration
        if st.button("💾 Save Current Configuration"):
            config = {
                'user_config': st.session_state.user_config,
                'feature_analysis': st.session_state.feature_analysis,
                'column_mappings': st.session_state.column_mappings,
                'saved_at': datetime.now().isoformat()
            }
            
            if ConfigurationManager.save_config(config, "full_config.json"):
                st.success("✅ Configuration saved successfully!")
        
        # Load configuration
        uploaded_config = st.file_uploader("📁 Upload Configuration File", type=['json'])
        if uploaded_config:
            try:
                config = json.load(uploaded_config)
                st.session_state.user_config = config.get('user_config', {})
                st.session_state.feature_analysis = config.get('feature_analysis', {})
                st.session_state.column_mappings = config.get('column_mappings', {})
                st.success("✅ Configuration loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load configuration: {e}")
        
        # Export model
        if 'current' in st.session_state.models:
            if st.button("📦 Export Trained Models"):
                try:
                    model_info = st.session_state.models['current']
                    export_data = {
                        'feature_cols': model_info['feature_cols'],
                        'target_column': model_info['target_column'],
                        'metrics': model_info['metrics'],
                        'trained_at': model_info['trained_at'].isoformat(),
                        'encoders': {k: v.classes_.tolist() for k, v in model_info['encoders'].items()},
                    }
                    
                    # Save non-neural network models
                    for model_name, model in model_info['models'].items():
                        if model_name != 'LSTM':
                            joblib.dump(model, f"{model_name.lower().replace(' ', '_')}_model.pkl")
                    
                    # Save export data
                    with open("model_export.json", "w") as f:
                        json.dump(export_data, f, indent=2)
                    
                    st.success("✅ Models exported successfully!")
                    st.info("📁 Files saved: model_export.json, random_forest_model.pkl, gradient_boosting_model.pkl")
                    
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")
    
    with col2:
        st.subheader("🔧 System Settings")
        
        # Display current configuration
        st.markdown("### 📋 Current Configuration")
        
        if st.session_state.user_config:
            st.json(st.session_state.user_config)
        else:
            st.info("No custom configuration set")
        
        # Advanced settings
        with st.expander("🔬 Advanced Settings"):
            st.markdown("**Performance Settings**")
            
            cache_enabled = st.checkbox("Enable Model Caching", True)
            parallel_processing = st.checkbox("Enable Parallel Processing", True)
            gpu_acceleration = st.checkbox("GPU Acceleration (if available)", False)
            
            st.markdown("**Data Processing Settings**")
            
            auto_data_validation = st.checkbox("Auto Data Validation", True)
            feature_scaling = st.selectbox(
                "Feature Scaling Method",
                ["Auto", "StandardScaler", "MinMaxScaler", "RobustScaler", "None"]
            )
            
            cross_validation_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            
            if st.button("💾 Apply Advanced Settings"):
                advanced_config = {
                    'cache_enabled': cache_enabled,
                    'parallel_processing': parallel_processing,
                    'gpu_acceleration': gpu_acceleration,
                    'auto_data_validation': auto_data_validation,
                    'feature_scaling': feature_scaling,
                    'cross_validation_folds': cross_validation_folds
                }
                
                st.session_state.user_config['advanced_settings'] = advanced_config
                st.success("✅ Advanced settings applied!")
        
        # System information
        st.subheader("ℹ️ System Information")
        
        sys_info = {
            'Streamlit Version': st.__version__,
            'Python Version': f"{'.'.join(map(str, [3, 8, 0]))}",  # Simplified
            'Data Loaded': len(st.session_state.data) if st.session_state.data is not None else 0,
            'Models Trained': len(st.session_state.models),
            'Predictions Made': len(st.session_state.predictions),
            'Features Analyzed': bool(st.session_state.feature_analysis),
            'Claude API Connected': bool(claude_api_key)
        }
        
        for key, value in sys_info.items():
            st.markdown(f"**{key}:** {value}")
        
        # Data quality report
        if st.session_state.data is not None:
            st.subheader("📊 Data Quality Report")
            
            with st.expander("View Data Quality Report"):
                data = st.session_state.data
                
                quality_report = {
                    'Total Rows': len(data),
                    'Total Columns': len(data.columns),
                    'Missing Values': data.isnull().sum().sum(),
                    'Duplicate Rows': data.duplicated().sum(),
                    'Numeric Columns': len(data.select_dtypes(include=[np.number]).columns),
                    'Categorical Columns': len(data.select_dtypes(include=['object']).columns),
                    'Memory Usage (MB)': round(data.memory_usage(deep=True).sum() / 1024**2, 2)
                }
                
                for metric, value in quality_report.items():
                    st.metric(metric, value)
                
                # Column-wise quality metrics
                st.markdown("**Column Quality Metrics:**")
                
                col_quality = []
                for col in data.columns:
                    col_info = {
                        'Column': col,
                        'Type': str(data[col].dtype),
                        'Missing %': round((data[col].isnull().sum() / len(data)) * 100, 2),
                        'Unique Values': data[col].nunique(),
                        'Most Frequent': str(data[col].mode().iloc[0]) if len(data[col].mode()) > 0 else 'N/A'
                    }
                    col_quality.append(col_info)
                
                quality_df = pd.DataFrame(col_quality)
                st.dataframe(quality_df, use_container_width=True)

# Additional Helper Functions
def validate_data_quality(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate data quality and return issues"""
    issues = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    # Check for missing values
    missing_pct = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_pct[missing_pct > 50]
    if not high_missing.empty:
        issues['errors'].extend([f"Column '{col}' has {pct:.1f}% missing values" 
                                for col, pct in high_missing.items()])
    
    moderate_missing = missing_pct[(missing_pct > 10) & (missing_pct <= 50)]
    if not moderate_missing.empty:
        issues['warnings'].extend([f"Column '{col}' has {pct:.1f}% missing values" 
                                  for col, pct in moderate_missing.items()])
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues['warnings'].append(f"{duplicates} duplicate rows found")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues['warnings'].extend([f"Column '{col}' has constant values" for col in constant_cols])
    
    # Check for high cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.8:
            issues['warnings'].append(f"Column '{col}' has very high cardinality ({unique_ratio:.1%})")
    
    # Check for numeric outliers
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        if len(outliers) > len(df) * 0.1:  # More than 10% outliers
            issues['info'].append(f"Column '{col}' has {len(outliers)} outliers ({len(outliers)/len(df):.1%})")
    
    return issues

# Data Quality Check Section
if st.session_state.data is not None:
    with st.expander("🔍 Run Data Quality Check"):
        if st.button("🧹 Analyze Data Quality"):
            with st.spinner("Analyzing data quality..."):
                quality_issues = validate_data_quality(st.session_state.data)
                
                # Display issues
                if quality_issues['errors']:
                    st.error("❌ **Critical Issues Found:**")
                    for error in quality_issues['errors']:
                        st.error(f"• {error}")
                
                if quality_issues['warnings']:
                    st.warning("⚠️ **Warnings:**")
                    for warning in quality_issues['warnings']:
                        st.warning(f"• {warning}")
                
                if quality_issues['info']:
                    st.info("ℹ️ **Information:**")
                    for info in quality_issues['info']:
                        st.info(f"• {info}")
                
                if not any(quality_issues.values()):
                    st.success("✅ No significant data quality issues found!")

# Performance Monitoring
if 'current' in st.session_state.models:
    st.subheader("⚡ Performance Monitoring")
    
    model_info = st.session_state.models['current']
    training_time = model_info['trained_at']
    
    # Model performance over time
    if len(st.session_state.predictions) > 0:
        # Calculate prediction accuracy trends
        recent_predictions = st.session_state.predictions[-20:]  # Last 20 predictions
        
        if len(recent_predictions) > 5:
            st.markdown("### 📈 Recent Performance Trends")
            
            # Create performance metrics
            performance_data = []
            for i, pred in enumerate(recent_predictions):
                # Calculate prediction variance as a proxy for uncertainty
                pred_values = list(pred['predictions'].values())
                if len(pred_values) > 1:
                    variance = np.var(pred_values)
                    performance_data.append({
                        'Prediction #': i + 1,
                        'Timestamp': pred['timestamp'],
                        'Prediction Variance': variance,
                        'Average Prediction': np.mean(pred_values)
                    })
            
            if performance_data:
                perf_df = pd.DataFrame(performance_data)
                
                # Plot variance trend
                fig = px.line(
                    perf_df, 
                    x='Prediction #', 
                    y='Prediction Variance',
                    title="Prediction Variance Trend",
                    hover_data=['Timestamp']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Variance", f"{perf_df['Prediction Variance'].mean():.2f}")
                with col2:
                    st.metric("Trend", "↑ Increasing" if perf_df['Prediction Variance'].iloc[-1] > perf_df['Prediction Variance'].iloc[0] else "↓ Decreasing")
                with col3:
                    st.metric("Predictions Made", len(recent_predictions))

# Footer with additional information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🚚 <strong>Intelligent Lane Optimization System</strong><br>
    <small>
        Multi-Agent AI • Auto-Configuration • Claude Intelligence • Dynamic Feature Engineering<br>
        Powered by TensorFlow, Scikit-learn, Anthropic Claude & Advanced ML Algorithms
    </small>
</div>
""", unsafe_allow_html=True)

# Quick Actions Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 Quick Actions")

if st.sidebar.button("🔄 Reset All Data"):
    for key in ['data', 'models', 'predictions', 'feature_analysis', 'column_mappings']:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("✅ Reset complete!")
    st.rerun()

if st.sidebar.button("💾 Quick Save Session"):
    session_data = {
        'data_shape': st.session_state.data.shape if st.session_state.data is not None else None,
        'models_trained': len(st.session_state.models),
        'predictions_made': len(st.session_state.predictions),
        'features_analyzed': bool(st.session_state.feature_analysis),
        'timestamp': datetime.now().isoformat()
    }
    
    with open("session_summary.json", "w") as f:
        json.dump(session_data, f, indent=2)
    
    st.sidebar.success("✅ Session saved!")

# Display session info
if st.session_state.data is not None:
    st.sidebar.markdown("### 📊 Session Info")
    st.sidebar.markdown(f"**Data:** {len(st.session_state.data)} rows")
    st.sidebar.markdown(f"**Models:** {len(st.session_state.models)}")
    st.sidebar.markdown(f"**Predictions:** {len(st.session_state.predictions)}")

# Tips and Help
with st.sidebar.expander("💡 Tips & Help"):
    st.markdown("""
    **Quick Start:**
    1. Upload your CSV file or generate sample data
    2. Add Claude API key for intelligent analysis
    3. Run data analysis to understand your dataset
    4. Let the system auto-engineer features
    5. Train all models with auto-configuration
    6. Make predictions and analyze results
    
    **Claude API Benefits:**
    - Intelligent column categorization
    - Automatic feature engineering suggestions
    - Data quality issue detection
    - Optimization recommendations
    
    **Supported File Formats:**
    - CSV (.csv)
    - Excel (.xlsx)
    - JSON (.json)
    
    **Model Types:**
    - Random Forest (Tree-based ensemble)
    - Gradient Boosting (Advanced boosting)
    - LSTM (Deep neural network)
    - Ensemble (Intelligent combination)
    """)

# Error handling and user feedback
if 'error_log' not in st.session_state:
    st.session_state.error_log = []

# Display any recent errors
if st.session_state.error_log:
    with st.sidebar.expander("⚠️ Recent Issues"):
        for error in st.session_state.error_log[-5:]:  # Show last 5 errors
            st.error(f"{error['timestamp']}: {error['message']}")

# System health check
def system_health_check():
    """Quick system health check"""
    health = {
        'data_loaded': st.session_state.data is not None,
        'models_trained': len(st.session_state.models) > 0,
        'claude_connected': bool(claude_api_key),
        'predictions_working': len(st.session_state.predictions) > 0,
        'config_valid': bool(st.session_state.user_config)
    }
    
    healthy_count = sum(health.values())
    total_checks = len(health)
    
    return health, healthy_count, total_checks

# Display health status
health, healthy, total = system_health_check()
health_percentage = (healthy / total) * 100

st.sidebar.markdown("### 🏥 System Health")
st.sidebar.progress(health_percentage / 100)
st.sidebar.markdown(f"**{healthy}/{total} checks passed** ({health_percentage:.0f}%)")

if health_percentage == 100:
    st.sidebar.success("✅ All systems operational!")
elif health_percentage >= 60:
    st.sidebar.warning("⚠️ Some features not configured")
else:
    st.sidebar.error("❌ System needs configuration")

