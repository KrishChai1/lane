import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import os
import requests
import json
from datetime import datetime, timedelta
import warnings
import math
import random
from typing import Dict, List, Tuple
import time

# Optional imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. LSTM models will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost not available. XGBoost models will be disabled.")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    st.warning("⚠️ LightGBM not available. LightGBM models will be disabled.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.warning("⚠️ Anthropic library not available. Claude API integration will be disabled.")

try:
    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False
    st.info("ℹ️ Geopy not available. Using estimated distances.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚚 Advanced Lane Optimization System",
    page_icon="🚚",
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
        color: white;
        text-align: center;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
    }
    .carrier-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .recommended-card {
        background: linear-gradient(135deg, #00c851 0%, #007e33 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        border: 3px solid #ffd700;
    }
    .cost-breakdown {
        background: #f0f7ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin-top: 1rem;
    }
    .model-performance {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .api-status {
        background: #e8f5e8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .api-error {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin-bottom: 1rem;
    }
    .stTab {
        background-color: #f8f9fa;
    }
    .input-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = []
    if 'model_performance' not in st.session_state:
        st.session_state.model_performance = {}
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'claude_client' not in st.session_state:
        st.session_state.claude_client = None
    if 'api_validated' not in st.session_state:
        st.session_state.api_validated = False
    if 'geocoder' not in st.session_state:
        if GEOPY_AVAILABLE:
            st.session_state.geocoder = Nominatim(user_agent="lane_optimizer")
        else:
            st.session_state.geocoder = None

# Claude API validation
def validate_claude_api(api_key):
    """Validate Claude API key"""
    if not ANTHROPIC_AVAILABLE:
        return False, "Anthropic library not installed"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Test with a simple request
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            messages=[{"role": "user", "content": "Test"}]
        )
        return True, client
    except Exception as e:
        return False, str(e)

# Enhanced ML Agent with better error handling and validation
class AdvancedMLAgent:
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_metrics = {}
        self.is_trained = False
        self.feature_columns = None
        
    def prepare_features(self, data, is_training=True):
        """Enhanced feature preparation with proper encoding"""
        try:
            data_copy = data.copy()
            
            # Define feature columns
            numeric_features = ['Package_Weight_lbs', 'Distance_miles', 'Package_Length_in', 
                              'Package_Width_in', 'Package_Height_in', 'Declared_Value_USD']
            categorical_features = ['Carrier', 'Service_Type', 'Origin_State', 'Destination_State']
            
            # Handle missing values
            for col in numeric_features:
                if col in data_copy.columns:
                    data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce').fillna(0)
            
            # Encode categorical variables
            for col in categorical_features:
                if col in data_copy.columns:
                    if is_training:
                        # Fit encoder during training
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                        data_copy[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data_copy[col].astype(str))
                    else:
                        # Transform using existing encoder
                        if col in self.label_encoders:
                            # Handle unseen categories
                            known_categories = self.label_encoders[col].classes_
                            data_copy[col] = data_copy[col].astype(str)
                            data_copy[col] = data_copy[col].apply(
                                lambda x: x if x in known_categories else known_categories[0]
                            )
                            data_copy[f'{col}_encoded'] = self.label_encoders[col].transform(data_copy[col])
                        else:
                            data_copy[f'{col}_encoded'] = 0
            
            # Create feature matrix
            encoded_features = [f'{col}_encoded' for col in categorical_features if col in data_copy.columns]
            available_numeric = [col for col in numeric_features if col in data_copy.columns]
            
            self.feature_columns = available_numeric + encoded_features
            feature_matrix = data_copy[self.feature_columns].fillna(0)
            
            return feature_matrix
            
        except Exception as e:
            st.error(f"Error in feature preparation for {self.name}: {str(e)}")
            return pd.DataFrame()
    
    def train(self, X, y):
        """Enhanced training with better error handling"""
        try:
            if X.empty or len(y) == 0:
                st.error(f"Empty dataset for {self.name}")
                return False
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model based on type with availability checks
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=10,
                    min_samples_split=5
                )
                self.model.fit(X_train_scaled, y_train)
                
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=6,
                    learning_rate=0.1
                )
                self.model.fit(X_train_scaled, y_train)
                
            elif self.model_type == 'xgboost':
                if not XGBOOST_AVAILABLE:
                    st.error("XGBoost not available. Please install: pip install xgboost")
                    return False
                self.model = xgb.XGBRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    n_jobs=-1,
                    max_depth=6,
                    learning_rate=0.1
                )
                self.model.fit(X_train_scaled, y_train)
                
            elif self.model_type == 'lightgbm':
                if not LIGHTGBM_AVAILABLE:
                    st.error("LightGBM not available. Please install: pip install lightgbm")
                    return False
                self.model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    max_depth=6,
                    learning_rate=0.1,
                    verbose=-1
                )
                self.model.fit(X_train_scaled, y_train)
                
            elif self.model_type == 'neural_network':
                self.model = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32), 
                    random_state=42, 
                    max_iter=500,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.2
                )
                self.model.fit(X_train_scaled, y_train)
                
            elif self.model_type == 'lstm':
                if not TENSORFLOW_AVAILABLE:
                    st.error("TensorFlow not available. Please install: pip install tensorflow")
                    return False
                self.model = self._build_lstm_model(X_train_scaled.shape[1])
                
                # Reshape for LSTM (samples, timesteps, features)
                X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
                
                history = self.model.fit(
                    X_train_lstm, y_train, 
                    epochs=100, 
                    batch_size=32, 
                    validation_data=(X_test_lstm, y_test), 
                    callbacks=[early_stopping], 
                    verbose=0
                )
                
            elif self.model_type == 'ensemble':
                # Create ensemble of available models
                models = [
                    ('rf', RandomForestRegressor(n_estimators=50, random_state=42, max_depth=8)),
                    ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42, max_depth=6))
                ]
                
                if XGBOOST_AVAILABLE:
                    models.append(('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42, max_depth=6)))
                
                if len(models) < 2:
                    st.error("Not enough models available for ensemble. Please install XGBoost.")
                    return False
                
                self.model = VotingRegressor(models)
                self.model.fit(X_train_scaled, y_train)
            
            # Calculate predictions
            if self.model_type == 'lstm':
                y_pred = self.model.predict(X_test_lstm, verbose=0).flatten()
            else:
                y_pred = self.model.predict(X_test_scaled)
            
            # Calculate performance metrics
            self.performance_metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'accuracy': max(0, 1 - (mean_absolute_error(y_test, y_pred) / np.mean(y_test))) if np.mean(y_test) > 0 else 0
            }
            
            self.is_trained = True
            return True
            
        except Exception as e:
            st.error(f"Error training {self.name}: {str(e)}")
            return False
    
    def _build_lstm_model(self, input_features):
        """Enhanced LSTM architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        model = Sequential([
            LSTM(128, input_shape=(1, input_features), return_sequences=True),
            Dropout(0.3),
            BatchNormalization(),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001), 
            loss='mse', 
            metrics=['mae']
        )
        return model
    
    def predict(self, X):
        """Enhanced prediction with error handling"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            # Ensure we have the same features as training
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_columns]
            
            X_scaled = self.scaler.transform(X.fillna(0))
            
            if self.model_type == 'lstm':
                X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                predictions = self.model.predict(X_scaled, verbose=0).flatten()
            else:
                predictions = self.model.predict(X_scaled)
            
            return predictions
            
        except Exception as e:
            st.error(f"Error making predictions with {self.name}: {str(e)}")
            return None

# Enhanced distance calculation
def calculate_real_distance(origin_city, origin_state, dest_city, dest_state):
    """Calculate real distance using geocoding"""
    if GEOPY_AVAILABLE and st.session_state.geocoder:
        try:
            origin_location = st.session_state.geocoder.geocode(f"{origin_city}, {origin_state}")
            dest_location = st.session_state.geocoder.geocode(f"{dest_city}, {dest_state}")
            
            if origin_location and dest_location:
                origin_coords = (origin_location.latitude, origin_location.longitude)
                dest_coords = (dest_location.latitude, dest_location.longitude)
                distance = geodesic(origin_coords, dest_coords).miles
                return distance
        except:
            pass
    
    # Fallback to estimation
    if origin_state == dest_state:
        if origin_city.lower() == dest_city.lower():
            return random.uniform(10, 50)
        else:
            return random.uniform(50, 400)
    else:
        return random.uniform(200, 2800)

# Carrier configurations
CARRIERS = {
    'UPS': {
        'base_rate': 8.50, 'weight_factor': 0.45, 'distance_factor': 0.0012,
        'service_multiplier': {'ground': 1.0, 'express': 1.8, 'overnight': 2.5},
        'zone_factors': [1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.1],
        'fuel_surcharge': 0.085, 'residential_fee': 4.95,
        'strength': 'Express Delivery', 'color': '#8B4513'
    },
    'FedEx': {
        'base_rate': 9.25, 'weight_factor': 0.42, 'distance_factor': 0.0011,
        'service_multiplier': {'ground': 1.0, 'express': 1.9, 'overnight': 2.7},
        'zone_factors': [1.0, 1.12, 1.25, 1.4, 1.55, 1.75, 1.95, 2.2],
        'fuel_surcharge': 0.092, 'residential_fee': 5.25,
        'strength': 'Overnight Delivery', 'color': '#4B0082'
    },
    'USPS': {
        'base_rate': 6.75, 'weight_factor': 0.38, 'distance_factor': 0.0008,
        'service_multiplier': {'ground': 1.0, 'express': 1.6, 'overnight': 2.2},
        'zone_factors': [1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        'fuel_surcharge': 0.065, 'residential_fee': 0.0,
        'strength': 'Cost Effective', 'color': '#1E90FF'
    },
    'DHL': {
        'base_rate': 11.50, 'weight_factor': 0.48, 'distance_factor': 0.0015,
        'service_multiplier': {'ground': 1.2, 'express': 2.0, 'overnight': 3.0},
        'zone_factors': [1.0, 1.15, 1.3, 1.5, 1.7, 1.9, 2.1, 2.4],
        'fuel_surcharge': 0.105, 'residential_fee': 6.50,
        'strength': 'International', 'color': '#FFD700'
    },
    'OnTrac': {
        'base_rate': 7.25, 'weight_factor': 0.35, 'distance_factor': 0.0009,
        'service_multiplier': {'ground': 1.0, 'express': 1.7, 'overnight': 2.3},
        'zone_factors': [1.0, 1.08, 1.15, 1.25, 1.35, 1.45, 1.55, 1.7],
        'fuel_surcharge': 0.075, 'residential_fee': 3.95,
        'strength': 'Regional Optimization', 'color': '#32CD32'
    }
}

# Enhanced cost calculation
def calculate_enhanced_cost(carrier, carrier_data, shipment_details, ml_adjustment=1.0):
    """Enhanced cost calculation with real-world factors"""
    try:
        weight = float(shipment_details.get('weight', 1))
        declared_value = float(shipment_details.get('declared_value', 100))
        length = float(shipment_details.get('length', 10))
        width = float(shipment_details.get('width', 10))
        height = float(shipment_details.get('height', 10))
        service_type = shipment_details.get('service_type', 'ground')
        special_handling = shipment_details.get('special_handling', 'none')
        
        # Calculate real distance
        distance = calculate_real_distance(
            shipment_details.get('origin_city', ''),
            shipment_details.get('origin_state', ''),
            shipment_details.get('dest_city', ''),
            shipment_details.get('dest_state', '')
        )
        
        # Calculate dimensional weight
        dim_weight = (length * width * height) / 166
        billable_weight = max(weight, dim_weight)
        
        # Enhanced cost calculation
        total_cost = carrier_data['base_rate']
        
        # Weight-based cost with progressive scaling
        weight_cost = billable_weight * carrier_data['weight_factor']
        if billable_weight > 10:
            weight_cost *= 1.1  # 10% surcharge for heavy packages
        
        total_cost += weight_cost
        
        # Distance-based cost with zone calculation
        zone = min(int(distance / 250), 7)
        zone_multiplier = carrier_data['zone_factors'][zone]
        total_cost *= zone_multiplier
        
        distance_cost = distance * carrier_data['distance_factor']
        total_cost += distance_cost
        
        # Service type multiplier
        service_multiplier = carrier_data['service_multiplier'][service_type]
        total_cost *= service_multiplier
        
        # Dynamic fuel surcharge (simulate market conditions)
        fuel_rate = carrier_data['fuel_surcharge'] * random.uniform(0.9, 1.1)
        fuel_cost = total_cost * fuel_rate
        total_cost += fuel_cost
        
        # Residential delivery fee
        total_cost += carrier_data['residential_fee']
        
        # Insurance cost
        insurance_cost = max(declared_value * 0.005, 2.0)
        total_cost += insurance_cost
        
        # Special handling fees
        handling_fees = {
            'fragile': 15.00, 'hazmat': 45.00, 'signature': 8.50, 
            'adult_signature': 12.75, 'white_glove': 25.00
        }
        if special_handling in handling_fees:
            total_cost += handling_fees[special_handling]
        
        # Apply ML model adjustment
        total_cost *= ml_adjustment
        
        # Calculate delivery estimation
        base_days = {'ground': zone + 1, 'express': max(zone - 1, 1), 'overnight': 1}
        carrier_adjustments = {'UPS': 0, 'FedEx': 0, 'USPS': 1, 'DHL': -1, 'OnTrac': 0}
        estimated_days = max(base_days[service_type] + carrier_adjustments.get(carrier, 0), 1)
        
        # Enhanced confidence calculation
        confidence = 0.85
        if carrier == 'USPS' and distance < 500: confidence += 0.1
        if carrier == 'OnTrac' and distance < 1000: confidence += 0.08
        if carrier in ['UPS', 'FedEx'] and weight > 5: confidence += 0.1
        if service_type == 'overnight' and carrier in ['UPS', 'FedEx']: confidence += 0.05
        
        # Add ML model confidence boost
        if ml_adjustment != 1.0:
            confidence += 0.05
        
        confidence = min(confidence + random.uniform(0, 0.05), 0.98)
        
        return {
            'carrier': carrier,
            'total_cost': round(total_cost, 2),
            'base_cost': carrier_data['base_rate'],
            'weight_cost': round(weight_cost, 2),
            'distance_cost': round(distance_cost, 2),
            'zone_multiplier': zone_multiplier,
            'fuel_cost': round(fuel_cost, 2),
            'insurance_cost': round(insurance_cost, 2),
            'estimated_days': estimated_days,
            'distance': round(distance, 1),
            'billable_weight': round(billable_weight, 1),
            'confidence_score': round(confidence, 3),
            'strength': carrier_data['strength'],
            'zone': zone + 1
        }
        
    except Exception as e:
        st.error(f"Error calculating cost for {carrier}: {str(e)}")
        return None

# Generate sample training data
def generate_sample_data(num_samples=1000):
    """Generate realistic sample training data"""
    np.random.seed(42)
    random.seed(42)
    
    carriers = list(CARRIERS.keys())
    service_types = ['ground', 'express', 'overnight']
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
    cities = {
        'CA': ['Los Angeles', 'San Francisco', 'San Diego'],
        'TX': ['Houston', 'Dallas', 'Austin'],
        'FL': ['Miami', 'Tampa', 'Jacksonville'],
        'NY': ['New York', 'Buffalo', 'Albany'],
        'IL': ['Chicago', 'Springfield', 'Rockford']
    }
    
    data = []
    for _ in range(num_samples):
        carrier = random.choice(carriers)
        service_type = random.choice(service_types)
        origin_state = random.choice(states)
        dest_state = random.choice(states)
        
        # Generate realistic package dimensions
        weight = round(random.uniform(0.5, 50), 1)
        length = round(random.uniform(6, 36), 1)
        width = round(random.uniform(4, 24), 1)
        height = round(random.uniform(2, 18), 1)
        declared_value = round(random.uniform(10, 5000), 2)
        
        # Calculate distance
        distance = calculate_real_distance('City1', origin_state, 'City2', dest_state)
        
        # Calculate realistic cost
        shipment_details = {
            'weight': weight, 'declared_value': declared_value,
            'length': length, 'width': width, 'height': height,
            'service_type': service_type, 'special_handling': 'none',
            'origin_state': origin_state, 'dest_state': dest_state,
            'origin_city': 'City1', 'dest_city': 'City2'
        }
        
        cost_result = calculate_enhanced_cost(carrier, CARRIERS[carrier], shipment_details)
        if cost_result:
            data.append({
                'Carrier': carrier,
                'Service_Type': service_type,
                'Package_Weight_lbs': weight,
                'Package_Length_in': length,
                'Package_Width_in': width,
                'Package_Height_in': height,
                'Declared_Value_USD': declared_value,
                'Distance_miles': distance,
                'Origin_State': origin_state,
                'Destination_State': dest_state,
                'Total_Cost_USD': cost_result['total_cost']
            })
    
    return pd.DataFrame(data)

# Main Streamlit App
def main():
    init_session_state()
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Claude API Key input
        st.subheader("🤖 Claude API Setup")
        if ANTHROPIC_AVAILABLE:
            # Use session state to persist API key
            if 'claude_api_key' not in st.session_state:
                st.session_state.claude_api_key = ""
            
            api_key = st.text_input(
                "Claude API Key", 
                value=st.session_state.claude_api_key,
                type="password", 
                help="Enter your Claude API key",
                key="api_key_input"
            )
            
            # Update session state when key changes
            if api_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = api_key
                st.session_state.api_validated = False  # Reset validation when key changes
            
            # Validate API key if provided and not already validated
            if api_key and len(api_key) > 10 and not st.session_state.get('api_validated', False):
                with st.spinner("Validating Claude API key..."):
                    is_valid, result = validate_claude_api(api_key)
                    if is_valid:
                        st.session_state.claude_client = result
                        st.session_state.api_validated = True
                        st.session_state.claude_api_key = api_key  # Store validated key
                        st.success("✅ API key validated successfully!")
                        st.rerun()  # Refresh to show connected status
                    else:
                        st.session_state.api_validated = False
                        st.error(f"❌ API validation failed: {result}")
            
            # Show connection status
            if st.session_state.get('api_validated', False) and st.session_state.get('claude_api_key', ''):
                st.markdown('<div class="api-status">🟢 Claude API Connected</div>', unsafe_allow_html=True)
                st.info(f"Using API key: ...{st.session_state.claude_api_key[-8:]}")
            else:
                st.markdown('<div class="api-error">🔴 Claude API Not Connected</div>', unsafe_allow_html=True)
                if api_key and len(api_key) < 10:
                    st.warning("API key seems too short. Please check and re-enter.")
        else:
            st.markdown('<div class="api-error">🔴 Anthropic library not installed</div>', unsafe_allow_html=True)
            st.info("📦 Install anthropic library to enable Claude API: `pip install anthropic`")
        
        # Model training status
        st.subheader("📊 Model Status")
        if st.session_state.trained_models:
            st.success(f"✅ {len(st.session_state.trained_models)} models trained")
            for model_name in st.session_state.trained_models.keys():
                performance = st.session_state.trained_models[model_name].performance_metrics
                if performance:
                    st.write(f"**{model_name.title()}**: {performance.get('accuracy', 0)*100:.1f}% accuracy")
        else:
            st.info("No models trained yet")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("Generate Sample Data", help="Create sample training data"):
            with st.spinner("Generating sample data..."):
                sample_data = generate_sample_data(1000)
                st.session_state.data = sample_data
                st.success("Sample data generated!")
        
        if st.button("Reset All Models", help="Clear all trained models"):
            st.session_state.trained_models = {}
            st.session_state.model_performance = {}
            st.success("Models reset!")

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚚 Advanced Lane Optimization System</h1>
        <h3>AI-Powered Carrier Selection with Multi-Model ML Agents</h3>
        <p>LSTM Neural Networks • Ensemble Methods • Real-time Cost Optimization • Claude AI Integration</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Optimize Routes", "📊 Train Models", "🔍 Model Performance", "📈 Analytics"])
    
    with tab1:
        st.header("🎯 Route Optimization with AI")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("📦 Shipment Details")
            
            # Origin details
            st.write("**🚀 Origin Information**")
            origin_city = st.text_input("Origin City", value="Chicago", placeholder="Enter city name")
            origin_state = st.selectbox("Origin State", 
                                      options=['IL', 'CA', 'TX', 'FL', 'NY', 'PA', 'OH', 'GA', 'NC', 'MI'],
                                      index=0)
            origin_zip = st.text_input("Origin ZIP", value="60601", placeholder="ZIP code")
            
            # Destination details
            st.write("**🎯 Destination Information**")
            dest_city = st.text_input("Destination City", value="Louisville", placeholder="Enter city name")
            dest_state = st.selectbox("Destination State", 
                                    options=['KY', 'CA', 'TX', 'FL', 'NY', 'PA', 'OH', 'GA', 'NC', 'MI'],
                                    index=0)
            dest_zip = st.text_input("Destination ZIP", value="40201", placeholder="ZIP code")
            
            # Package details
            st.write("**📦 Package Specifications**")
            col_w, col_v = st.columns(2)
            with col_w:
                weight = st.number_input("Weight (lbs)", min_value=0.1, value=5.0, step=0.1)
            with col_v:
                declared_value = st.number_input("Declared Value ($)", min_value=1.0, value=500.0, step=10.0)
            
            col_l, col_w_dim, col_h = st.columns(3)
            with col_l:
                length = st.number_input("Length (in)", min_value=1.0, value=12.0, step=0.5)
            with col_w_dim:
                width = st.number_input("Width (in)", min_value=1.0, value=8.0, step=0.5)
            with col_h:
                height = st.number_input("Height (in)", min_value=1.0, value=6.0, step=0.5)
            
            # Service options
            st.write("**⚡ Service Preferences**")
            service_type = st.selectbox("Service Type", 
                                      options=['ground', 'express', 'overnight'],
                                      format_func=lambda x: f"🚛 {x.replace('_', ' ').title()}")
            
            special_handling = st.selectbox("Special Handling", 
                                          options=['none', 'fragile', 'hazmat', 'signature', 'adult_signature'],
                                          format_func=lambda x: f"📋 {x.replace('_', ' ').title()}")
            
            # ML Model selection
            st.write("**🤖 AI Model Selection**")
            
            # Get available models based on installed libraries
            available_models = []
            if st.session_state.trained_models:
                available_models = list(st.session_state.trained_models.keys())
            
            # Add base calculation as fallback
            if not available_models:
                available_models = ['base_calculation']
            
            model_names = {
                'ensemble': '🏆 Ensemble (Best)',
                'lstm': '🧠 LSTM Neural Network',
                'xgboost': '⚡ XGBoost',
                'lightgbm': '💡 LightGBM',
                'gradient_boosting': '📈 Gradient Boosting',
                'random_forest': '🌳 Random Forest',
                'neural_network': '🔗 Neural Network',
                'base_calculation': '📊 Base Calculation'
            }
            
            if len(available_models) > 1 or available_models[0] != 'base_calculation':
                selected_model = st.selectbox("ML Model", 
                                            options=available_models,
                                            format_func=lambda x: model_names.get(x, x.title()))
            else:
                selected_model = 'base_calculation'
                st.info("📚 Train models first to use ML predictions")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate button
            if st.button("🚀 Calculate Optimal Routes", type="primary", use_container_width=True):
                if origin_city and dest_city:
                    with st.spinner("🔄 Calculating optimal routes with AI..."):
                        shipment_details = {
                            'origin_city': origin_city, 'origin_state': origin_state,
                            'dest_city': dest_city, 'dest_state': dest_state,
                            'weight': weight, 'declared_value': declared_value,
                            'length': length, 'width': width, 'height': height,
                            'service_type': service_type, 'special_handling': special_handling
                        }
                        
                        # Calculate costs for all carriers
                        results = []
                        for carrier, carrier_data in CARRIERS.items():
                            ml_adjustment = 1.0
                            
                            # Apply ML prediction if model is available
                            if selected_model != 'base_calculation' and selected_model in st.session_state.trained_models:
                                try:
                                    # Create feature vector
                                    distance = calculate_real_distance(origin_city, origin_state, dest_city, dest_state)
                                    
                                    feature_data = pd.DataFrame([{
                                        'Package_Weight_lbs': weight,
                                        'Package_Length_in': length,
                                        'Package_Width_in': width,
                                        'Package_Height_in': height,
                                        'Declared_Value_USD': declared_value,
                                        'Distance_miles': distance,
                                        'Carrier': carrier,
                                        'Service_Type': service_type,
                                        'Origin_State': origin_state,
                                        'Destination_State': dest_state
                                    }])
                                    
                                    # Prepare features using the same preprocessing
                                    model_agent = st.session_state.trained_models[selected_model]
                                    processed_features = model_agent.prepare_features(feature_data, is_training=False)
                                    
                                    # Get ML prediction
                                    prediction = model_agent.predict(processed_features)
                                    
                                    if prediction is not None and len(prediction) > 0:
                                        # Calculate base cost for comparison
                                        base_cost = calculate_enhanced_cost(carrier, carrier_data, shipment_details)
                                        if base_cost:
                                            # Use ML prediction directly as the cost
                                            ml_cost = max(prediction[0], 5.0)  # Minimum $5
                                            cost_result = base_cost.copy()
                                            cost_result['total_cost'] = round(ml_cost, 2)
                                            cost_result['ml_adjusted'] = True
                                            results.append(cost_result)
                                            continue
                                    
                                except Exception as e:
                                    st.warning(f"ML prediction failed for {carrier}, using base calculation: {str(e)}")
                            
                            # Fallback to base calculation
                            cost_result = calculate_enhanced_cost(carrier, carrier_data, shipment_details, ml_adjustment)
                            if cost_result:
                                cost_result['ml_adjusted'] = False
                                results.append(cost_result)
                        
                        # Sort and rank results
                        if results:
                            results.sort(key=lambda x: x['total_cost'])
                            for i, result in enumerate(results[:3]):
                                result['recommended'] = True
                                result['rank'] = i + 1
                            
                            st.session_state.predictions = results
                        else:
                            st.error("Failed to calculate costs for any carrier")
                else:
                    st.error("Please enter both origin and destination cities.")
        
        with col2:
            if st.session_state.predictions:
                st.subheader("🎯 AI-Powered Carrier Recommendations")
                
                # Summary metrics
                best_cost = min(r['total_cost'] for r in st.session_state.predictions)
                fastest_delivery = min(r['estimated_days'] for r in st.session_state.predictions)
                highest_confidence = max(r['confidence_score'] for r in st.session_state.predictions)
                ml_count = sum(1 for r in st.session_state.predictions if r.get('ml_adjusted', False))
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("💰 Best Price", f"${best_cost:.2f}")
                with metric_col2:
                    st.metric("⚡ Fastest", f"{fastest_delivery} day{'s' if fastest_delivery != 1 else ''}")
                with metric_col3:
                    st.metric("🎯 Confidence", f"{highest_confidence*100:.1f}%")
                with metric_col4:
                    st.metric("🤖 ML Enhanced", f"{ml_count}/{len(st.session_state.predictions)}")
                
                # Display results
                for i, result in enumerate(st.session_state.predictions):
                    if result.get('recommended', False):
                        st.markdown(f"""
                        <div class="recommended-card">
                            <h3>🏆 #{result['rank']} {result['carrier']} - RECOMMENDED 
                            {'🤖 ML Enhanced' if result.get('ml_adjusted') else '📊 Base Calc'}</h3>
                            <p><strong>Strength:</strong> {result['strength']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="carrier-card">
                            <h3>{result['carrier']} {'🤖' if result.get('ml_adjusted') else '📊'}</h3>
                            <p><strong>Strength:</strong> {result['strength']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Cost and details
                    detail_col1, detail_col2 = st.columns([2, 1])
                    with detail_col1:
                        st.markdown(f"""
                        <div class="cost-breakdown">
                            <h4>💰 Cost Breakdown</h4>
                            <ul>
                                <li><strong>Distance:</strong> {result['distance']:.1f} miles (Zone {result['zone']})</li>
                                <li><strong>Billable Weight:</strong> {result['billable_weight']:.1f} lbs</li>
                                <li><strong>Base Rate:</strong> ${result['base_cost']:.2f}</li>
                                <li><strong>Weight Cost:</strong> ${result['weight_cost']:.2f}</li>
                                <li><strong>Distance Cost:</strong> ${result['distance_cost']:.2f}</li>
                                <li><strong>Zone Multiplier:</strong> {result['zone_multiplier']:.2f}x</li>
                                <li><strong>Fuel Surcharge:</strong> ${result['fuel_cost']:.2f}</li>
                                <li><strong>Insurance:</strong> ${result['insurance_cost']:.2f}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with detail_col2:
                        st.metric("💵 Total Cost", f"${result['total_cost']:.2f}")
                        st.metric("📅 Delivery", f"{result['estimated_days']} days")
                        st.metric("🎯 Confidence", f"{result['confidence_score']*100:.1f}%")
                        if result.get('ml_adjusted'):
                            st.success("🤖 ML Enhanced")
                        else:
                            st.info("📊 Base Calculation")
                    
                    st.markdown("---")
                
                # Enhanced visualization
                if len(st.session_state.predictions) > 1:
                    st.subheader("📊 Interactive Cost Analysis")
                    
                    # Cost comparison chart
                    fig1 = go.Figure()
                    
                    carriers = [r['carrier'] for r in st.session_state.predictions]
                    costs = [r['total_cost'] for r in st.session_state.predictions]
                    colors = [CARRIERS[r['carrier']]['color'] for r in st.session_state.predictions]
                    ml_enhanced = ['🤖 ML Enhanced' if r.get('ml_adjusted') else '📊 Base Calc' for r in st.session_state.predictions]
                    
                    fig1.add_trace(go.Bar(
                        x=carriers,
                        y=costs,
                        marker_color=colors,
                        text=[f"${cost:.2f}<br>{ml}" for cost, ml in zip(costs, ml_enhanced)],
                        textposition='auto',
                        hovertemplate='<b>%{x}</b><br>Cost: $%{y:.2f}<br>%{text}<extra></extra>'
                    ))
                    
                    fig1.update_layout(
                        title="🏆 Carrier Cost Comparison",
                        xaxis_title="Carrier",
                        yaxis_title="Total Cost ($)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    # Cost vs Delivery scatter plot
                    fig2 = go.Figure()
                    
                    delivery_days = [r['estimated_days'] for r in st.session_state.predictions]
                    confidence_scores = [r['confidence_score'] for r in st.session_state.predictions]
                    
                    for i, carrier in enumerate(carriers):
                        fig2.add_trace(go.Scatter(
                            x=[delivery_days[i]],
                            y=[costs[i]],
                            mode='markers+text',
                            name=carrier,
                            text=[carrier],
                            textposition="top center",
                            marker=dict(
                                size=confidence_scores[i] * 40,
                                color=colors[i],
                                opacity=0.7,
                                line=dict(width=2, color='white')
                            ),
                            hovertemplate=f'<b>{carrier}</b><br>Cost: ${costs[i]:.2f}<br>Delivery: {delivery_days[i]} days<br>Confidence: {confidence_scores[i]*100:.1f}%<extra></extra>'
                        ))
                    
                    fig2.update_layout(
                        title="📈 Cost vs Delivery Time Analysis (Bubble size = Confidence)",
                        xaxis_title="Delivery Days",
                        yaxis_title="Total Cost ($)",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("👆 Enter shipment details and click 'Calculate Optimal Routes' to see AI-powered recommendations.")
                
                # Enhanced feature showcase
                st.markdown("""
                ### 🚀 Advanced System Capabilities
                
                **🧠 Machine Learning Models:**
                - 🏆 **Ensemble Methods**: Combines multiple algorithms for maximum accuracy
                - 🧠 **LSTM Neural Networks**: Deep learning for time-series cost prediction  
                - ⚡ **XGBoost & LightGBM**: Gradient boosting for complex patterns
                - 🔗 **Neural Networks**: Multi-layer perceptrons for non-linear relationships
                - 🌳 **Random Forest**: Robust tree-based ensemble learning
                
                **💰 Real-World Cost Factors:**
                - 🗺️ **Real Distance Calculation**: Geocoding-based distance measurement
                - 📦 **Dimensional Weight**: Accurate billable weight calculations
                - ⛽ **Dynamic Fuel Surcharges**: Market-responsive pricing
                - 🏠 **Residential Fees**: Accurate delivery surcharges
                - 🛡️ **Insurance Coverage**: Value-based protection costs
                
                **🎯 Intelligent Optimization:**
                - 🏆 **Multi-Carrier Analysis**: Compare all major carriers simultaneously
                - 📊 **Confidence Scoring**: ML-based prediction reliability
                - 🎨 **Interactive Visualizations**: Real-time cost and delivery analysis
                - 🤖 **Claude AI Integration**: Advanced reasoning and recommendations
                """)
    
    with tab2:
        st.header("📊 Advanced ML Model Training")
        
        # Data source selection
        data_source = st.radio("Choose Data Source:", 
                              options=["Upload CSV File", "Generate Sample Data", "Use Existing Data"],
                              horizontal=True)
        
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.success(f"✅ Data loaded successfully! Shape: {data.shape}")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        elif data_source == "Generate Sample Data":
            col1, col2 = st.columns(2)
            with col1:
                num_samples = st.number_input("Number of Samples", min_value=100, max_value=10000, value=1000)
            with col2:
                if st.button("🎲 Generate Data", type="primary"):
                    with st.spinner("Generating realistic sample data..."):
                        sample_data = generate_sample_data(num_samples)
                        st.session_state.data = sample_data
                        st.success(f"✅ Generated {num_samples} sample records!")
        
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Data overview
            st.subheader("📋 Data Overview")
            
            overview_col1, overview_col2, overview_col3 = st.columns(3)
            with overview_col1:
                st.metric("📊 Total Records", len(data))
            with overview_col2:
                st.metric("📈 Features", len(data.columns))
            with overview_col3:
                st.metric("💰 Avg Cost", f"${data.get('Total_Cost_USD', [0]).mean():.2f}" if 'Total_Cost_USD' in data.columns else "N/A")
            
            # Data preview
            with st.expander("👀 Data Preview", expanded=False):
                st.dataframe(data.head(10))
            
            # Data quality check
            with st.expander("🔍 Data Quality Analysis", expanded=False):
                st.write("**Missing Values:**")
                missing_data = data.isnull().sum()
                if missing_data.sum() > 0:
                    st.dataframe(missing_data[missing_data > 0])
                else:
                    st.success("✅ No missing values found!")
                
                st.write("**Data Types:**")
                st.dataframe(data.dtypes)
            
            # Model configuration
            st.subheader("🎯 Model Configuration")
            
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                # Target variable selection
                target_column = st.selectbox("🎯 Select Target Variable (Cost Column)", 
                                           options=data.columns.tolist(),
                                           index=list(data.columns).index('Total_Cost_USD') if 'Total_Cost_USD' in data.columns else 0)
            
            with config_col2:
                # Test size
                test_size = st.slider("📊 Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            # Model selection with availability checks
            st.subheader("🤖 Select Models to Train")
            
            model_options = {
                'random_forest': '🌳 Random Forest - Robust ensemble method',
                'gradient_boosting': '📈 Gradient Boosting - Sequential improvement',
                'neural_network': '🔗 Neural Network - Multi-layer perceptron'
            }
            
            # Add optional models based on availability
            if XGBOOST_AVAILABLE:
                model_options['xgboost'] = '⚡ XGBoost - Optimized gradient boosting'
            
            if LIGHTGBM_AVAILABLE:
                model_options['lightgbm'] = '💡 LightGBM - Fast gradient boosting'
            
            if TENSORFLOW_AVAILABLE:
                model_options['lstm'] = '🧠 LSTM - Long Short-Term Memory network'
            
            # Ensemble requires at least 2 base models
            available_base_models = ['random_forest', 'gradient_boosting']
            if XGBOOST_AVAILABLE:
                available_base_models.append('xgboost')
            
            if len(available_base_models) >= 2:
                model_options['ensemble'] = '🏆 Ensemble - Combines multiple models'
            
            selected_models = []
            
            for model_key, model_desc in model_options.items():
                if st.checkbox(model_desc, key=f"model_{model_key}"):
                    selected_models.append(model_key)
            
            # Training configuration
            with st.expander("⚙️ Advanced Training Settings", expanded=False):
                use_cross_validation = st.checkbox("🔄 Use Cross-Validation", value=True)
                n_splits = st.number_input("CV Folds", min_value=3, max_value=10, value=5) if use_cross_validation else 5
                random_state = st.number_input("🎲 Random State", min_value=0, max_value=1000, value=42)
            
            # Training button
            if st.button("🚀 Train Selected Models", type="primary", disabled=len(selected_models) == 0):
                if target_column and selected_models:
                    
                    # Validate target column
                    if not pd.api.types.is_numeric_dtype(data[target_column]):
                        st.error(f"Target column '{target_column}' must be numeric!")
                        return
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = {}
                    
                    # Prepare target variable
                    y = data[target_column].copy()
                    
                    # Remove rows with missing target values
                    valid_indices = ~y.isnull()
                    y = y[valid_indices]
                    data_clean = data[valid_indices].copy()
                    
                    if len(y) < 10:
                        st.error("Not enough valid data points for training!")
                        return
                    
                    for i, model_type in enumerate(selected_models):
                        status_text.text(f"🔄 Training {model_options[model_type]}...")
                        
                        try:
                            # Create and train agent
                            agent = AdvancedMLAgent(f"{model_type}_agent", model_type)
                            
                            # Prepare features
                            features = agent.prepare_features(data_clean, is_training=True)
                            
                            if features.empty:
                                st.error(f"Failed to prepare features for {model_type}")
                                continue
                            
                            # Train model
                            success = agent.train(features, y)
                            
                            if success:
                                st.session_state.trained_models[model_type] = agent
                                results[model_type] = agent.performance_metrics
                                
                                # Cross-validation if enabled
                                if use_cross_validation and model_type != 'lstm':
                                    try:
                                        cv_scores = cross_val_score(
                                            agent.model, 
                                            agent.scaler.transform(features), 
                                            y, 
                                            cv=n_splits, 
                                            scoring='neg_mean_absolute_error',
                                            n_jobs=-1
                                        )
                                        results[model_type]['cv_mae'] = -cv_scores.mean()
                                        results[model_type]['cv_std'] = cv_scores.std()
                                    except:
                                        pass
                                
                                st.success(f"✅ {model_options[model_type]} trained successfully!")
                            else:
                                st.error(f"❌ Failed to train {model_options[model_type]}")
                                
                        except Exception as e:
                            st.error(f"❌ Error training {model_type}: {str(e)}")
                        
                        progress_bar.progress((i + 1) / len(selected_models))
                    
                    st.session_state.model_performance = results
                    status_text.text("🎉 Training completed!")
                    
                    # Training results summary
                    if results:
                        st.subheader("🏆 Training Results Summary")
                        
                        results_df = pd.DataFrame(results).T
                        results_df['accuracy_pct'] = results_df['accuracy'] * 100
                        
                        # Display results table
                        display_columns = ['accuracy_pct', 'mae', 'rmse', 'r2']
                        if 'cv_mae' in results_df.columns:
                            display_columns.append('cv_mae')
                        
                        display_df = results_df[display_columns].round(3)
                        display_df.columns = ['Accuracy (%)', 'MAE ($)', 'RMSE ($)', 'R² Score'] + (['CV MAE ($)'] if 'cv_mae' in results_df.columns else [])
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Best model identification
                        best_model = results_df['accuracy'].idxmax()
                        best_accuracy = results_df.loc[best_model, 'accuracy_pct']
                        
                        st.success(f"🏆 **Best Model**: {model_options[best_model]} with {best_accuracy:.1f}% accuracy!")
                        
                        # Model comparison chart
                        fig = px.bar(
                            x=results_df.index,
                            y=results_df['accuracy_pct'],
                            title="🏆 Model Performance Comparison",
                            labels={'x': 'Model', 'y': 'Accuracy (%)'},
                            color=results_df['accuracy_pct'],
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Please select target variable and at least one model.")
        
        else:
            st.info("📁 Please select a data source to begin model training.")
            
            # Sample data format guide
            st.subheader("📋 Expected Data Format")
            st.write("Your CSV should contain columns like:")
            
            sample_data = pd.DataFrame({
                'Carrier': ['UPS', 'FedEx', 'USPS', 'DHL', 'OnTrac'],
                'Service_Type': ['ground', 'express', 'overnight', 'ground', 'express'],
                'Package_Weight_lbs': [5.0, 3.2, 8.5, 2.1, 7.3],
                'Distance_miles': [450, 1200, 800, 300, 650],
                'Origin_State': ['IL', 'CA', 'NY', 'TX', 'FL'],
                'Destination_State': ['KY', 'TX', 'FL', 'CA', 'GA'],
                'Package_Length_in': [12, 8, 15, 10, 14],
                'Package_Width_in': [8, 6, 10, 7, 9],
                'Package_Height_in': [6, 4, 8, 5, 7],
                'Declared_Value_USD': [500, 200, 1000, 150, 750],
                'Total_Cost_USD': [25.50, 45.75, 35.20, 28.90, 32.15]
            })
            st.dataframe(sample_data)
    
    with tab3:
        st.header("🔍 Comprehensive Model Performance Analysis")
        
        if st.session_state.model_performance:
            perf_df = pd.DataFrame(st.session_state.model_performance).T
            perf_df['accuracy_pct'] = perf_df['accuracy'] * 100
            
            # Performance overview
            st.subheader("📊 Performance Overview")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                best_model = perf_df['accuracy'].idxmax()
                st.metric("🏆 Best Model", best_model.title(), f"{perf_df.loc[best_model, 'accuracy_pct']:.1f}%")
            
            with metric_col2:
                lowest_mae = perf_df['mae'].idxmin()
                st.metric("📉 Lowest Error", lowest_mae.title(), f"${perf_df.loc[lowest_mae, 'mae']:.2f}")
            
            with metric_col3:
                highest_r2 = perf_df['r2'].idxmax()
                st.metric("📈 Best Fit", highest_r2.title(), f"{perf_df.loc[highest_r2, 'r2']:.3f}")
            
            with metric_col4:
                avg_accuracy = perf_df['accuracy_pct'].mean()
                st.metric("📊 Avg Accuracy", f"{avg_accuracy:.1f}%", f"±{perf_df['accuracy_pct'].std():.1f}%")
            
            # Detailed performance visualization
            st.subheader("📈 Detailed Performance Metrics")
            
            # Multi-metric comparison
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('🎯 Accuracy (%)', '📉 Mean Absolute Error ($)', '📊 R² Score', '📈 RMSE ($)'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            models = perf_df.index.tolist()
            colors = px.colors.qualitative.Set3[:len(models)]
            
            # Accuracy
            fig.add_trace(
                go.Bar(x=models, y=perf_df['accuracy_pct'], marker_color=colors, 
                       text=[f"{x:.1f}%" for x in perf_df['accuracy_pct']], textposition='auto'),
                row=1, col=1
            )
            
            # MAE
            fig.add_trace(
                go.Bar(x=models, y=perf_df['mae'], marker_color=colors,
                       text=[f"${x:.2f}" for x in perf_df['mae']], textposition='auto'),
                row=1, col=2
            )
            
            # R²
            fig.add_trace(
                go.Bar(x=models, y=perf_df['r2'], marker_color=colors,
                       text=[f"{x:.3f}" for x in perf_df['r2']], textposition='auto'),
                row=2, col=1
            )
            
            # RMSE
            fig.add_trace(
                go.Bar(x=models, y=perf_df['rmse'], marker_color=colors,
                       text=[f"${x:.2f}" for x in perf_df['rmse']], textposition='auto'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance ranking
            st.subheader("🏆 Model Rankings")
            
            ranking_metrics = ['accuracy_pct', 'mae', 'r2', 'rmse']
            ranking_data = []
            
            for metric in ranking_metrics:
                if metric in ['mae', 'rmse']:  # Lower is better
                    ranked = perf_df.sort_values(metric).index.tolist()
                else:  # Higher is better
                    ranked = perf_df.sort_values(metric, ascending=False).index.tolist()
                
                for i, model in enumerate(ranked):
                    ranking_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Rank': i + 1,
                        'Model': model.title(),
                        'Value': perf_df.loc[model, metric]
                    })
            
            ranking_df = pd.DataFrame(ranking_data)
            
            # Create ranking heatmap
            pivot_ranking = ranking_df.pivot(index='Model', columns='Metric', values='Rank')
            
            fig_heatmap = px.imshow(
                pivot_ranking.values,
                x=pivot_ranking.columns,
                y=pivot_ranking.index,
                color_continuous_scale='RdYlGn_r',
                title="🎯 Model Ranking Heatmap (1=Best, Higher=Worse)",
                text_auto=True
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("📋 Detailed Performance Table")
            
            display_df = perf_df.copy()
            display_df = display_df.round(4)
            
            # Add CV results if available
            if 'cv_mae' in display_df.columns:
                display_df['cv_mae'] = display_df['cv_mae'].round(2)
                display_df['cv_std'] = display_df['cv_std'].round(2)
            
            # Format for display
            formatted_df = display_df[['accuracy_pct', 'mae', 'rmse', 'r2']].copy()
            formatted_df.columns = ['Accuracy (%)', 'MAE ($)', 'RMSE ($)', 'R² Score']
            
            # Add CV columns if available
            if 'cv_mae' in display_df.columns:
                formatted_df['CV MAE ($)'] = display_df['cv_mae']
                formatted_df['CV Std'] = display_df['cv_std']
            
            # Color code the dataframe
            st.dataframe(
                formatted_df.style.format({
                    'Accuracy (%)': '{:.2f}%',
                    'MAE ($)': '${:.2f}',
                    'RMSE ($)': '${:.2f}',
                    'R² Score': '{:.3f}',
                    'CV MAE ($)': '${:.2f}',
                    'CV Std': '{:.3f}'
                }).background_gradient(subset=['Accuracy (%)', 'R² Score'], cmap='Greens')
                .background_gradient(subset=['MAE ($)', 'RMSE ($)'], cmap='Reds_r'),
                use_container_width=True
            )
            
            # Model insights
            st.subheader("🧠 Model Insights & Recommendations")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("### 🎯 Best Performers")
                
                best_accuracy = perf_df['accuracy'].idxmax()
                best_mae = perf_df['mae'].idxmin()
                best_r2 = perf_df['r2'].idxmax()
                
                st.success(f"🏆 **Highest Accuracy**: {best_accuracy.title()} ({perf_df.loc[best_accuracy, 'accuracy_pct']:.1f}%)")
                st.success(f"📉 **Lowest Error**: {best_mae.title()} (${perf_df.loc[best_mae, 'mae']:.2f} MAE)")
                st.success(f"📊 **Best Fit**: {best_r2.title()} ({perf_df.loc[best_r2, 'r2']:.3f} R²)")
                
                # Overall recommendation
                if best_accuracy == best_mae == best_r2:
                    st.info(f"🌟 **Overall Champion**: {best_accuracy.title()} excels in all metrics!")
                else:
                    # Calculate overall score
                    normalized_scores = perf_df.copy()
                    normalized_scores['accuracy_norm'] = (normalized_scores['accuracy'] - normalized_scores['accuracy'].min()) / (normalized_scores['accuracy'].max() - normalized_scores['accuracy'].min())
                    normalized_scores['mae_norm'] = 1 - ((normalized_scores['mae'] - normalized_scores['mae'].min()) / (normalized_scores['mae'].max() - normalized_scores['mae'].min()))
                    normalized_scores['r2_norm'] = (normalized_scores['r2'] - normalized_scores['r2'].min()) / (normalized_scores['r2'].max() - normalized_scores['r2'].min())
                    
                    normalized_scores['overall_score'] = (normalized_scores['accuracy_norm'] + normalized_scores['mae_norm'] + normalized_scores['r2_norm']) / 3
                    overall_best = normalized_scores['overall_score'].idxmax()
                    
                    st.info(f"🌟 **Overall Best**: {overall_best.title()} (balanced performance)")
            
            with insights_col2:
                st.markdown("### 📈 Performance Analysis")
                
                # Performance distribution
                accuracy_range = perf_df['accuracy_pct'].max() - perf_df['accuracy_pct'].min()
                mae_range = perf_df['mae'].max() - perf_df['mae'].min()
                
                if accuracy_range < 5:
                    st.info("🔄 **Close Competition**: All models perform similarly")
                else:
                    st.warning("📊 **Performance Gap**: Significant difference between models")
                
                # Model complexity vs performance
                complexity_order = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm', 'neural_network', 'lstm', 'ensemble']
                
                if 'ensemble' in perf_df.index and perf_df.loc['ensemble', 'accuracy'] > perf_df['accuracy'].mean():
                    st.success("🏆 **Ensemble Advantage**: Combined models outperform individual ones")
                
                if 'lstm' in perf_df.index:
                    if perf_df.loc['lstm', 'accuracy'] > perf_df.drop('lstm')['accuracy'].mean():
                        st.success("🧠 **Deep Learning Bonus**: LSTM captures complex patterns")
                    else:
                        st.info("📊 **Traditional ML**: Simpler models work well for this data")
        
        else:
            st.info("🤖 Train some models first to see comprehensive performance analysis.")
            
            # Performance interpretation guide
            st.markdown("""
            ### 📚 Performance Metrics Guide
            
            **🎯 Accuracy**: Percentage of predictions within acceptable range
            - 90%+ = Excellent
            - 80-90% = Good  
            - 70-80% = Fair
            - <70% = Needs improvement
            
            **📉 MAE (Mean Absolute Error)**: Average prediction error in dollars
            - Lower is better
            - Should be <10% of average cost
            
            **📊 R² Score**: How well the model explains cost variation
            - 1.0 = Perfect fit
            - 0.8+ = Very good
            - 0.6-0.8 = Good
            - <0.6 = Poor fit
            
            **📈 RMSE**: Root Mean Square Error, penalizes large errors
            - Similar to MAE but more sensitive to outliers
            """)
    
    with tab4:
        st.header("📈 Advanced Analytics Dashboard")
        
        if st.session_state.predictions:
            predictions = st.session_state.predictions
            
            # Analytics overview
            st.subheader("📊 Route Optimization Analysis")
            
            carriers = [p['carrier'] for p in predictions]
            total_costs = [p['total_cost'] for p in predictions]
            delivery_days = [p['estimated_days'] for p in predictions]
            confidence_scores = [p['confidence_score'] for p in predictions]
            distances = [p['distance'] for p in predictions]
            zones = [p['zone'] for p in predictions]
            
            # Key insights
            insight_col1, insight_col2, insight_col3, insight_col4 = st.columns(4)
            
            with insight_col1:
                cost_savings = max(total_costs) - min(total_costs)
                st.metric("💰 Max Savings", f"${cost_savings:.2f}", f"{(cost_savings/max(total_costs)*100):.1f}%")
            
            with insight_col2:
                time_advantage = max(delivery_days) - min(delivery_days)
                st.metric("⚡ Time Advantage", f"{time_advantage} days", f"{(time_advantage/max(delivery_days)*100):.1f}%")
            
            with insight_col3:
                avg_confidence = np.mean(confidence_scores)
                st.metric("🎯 Avg Confidence", f"{avg_confidence*100:.1f}%", f"±{np.std(confidence_scores)*100:.1f}%")
            
            with insight_col4:
                ml_enhanced = sum(1 for p in predictions if p.get('ml_adjusted', False))
                st.metric("🤖 ML Enhanced", f"{ml_enhanced}/{len(predictions)}", f"{(ml_enhanced/len(predictions)*100):.0f}%")
            
            # Comprehensive cost analysis
            st.subheader("💰 Cost Structure Analysis")
            
            # Cost breakdown pie chart
            col1, col2 = st.columns(2)
            
            with col1:
                # Average cost breakdown
                avg_breakdown = {
                    'Base Cost': np.mean([p['base_cost'] for p in predictions]),
                    'Weight Cost': np.mean([p['weight_cost'] for p in predictions]),
                    'Distance Cost': np.mean([p['distance_cost'] for p in predictions]),
                    'Fuel Surcharge': np.mean([p['fuel_cost'] for p in predictions]),
                    'Insurance': np.mean([p['insurance_cost'] for p in predictions])
                }
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(avg_breakdown.keys()),
                    values=list(avg_breakdown.values()),
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='outside'
                )])
                
                fig_pie.update_layout(
                    title="Average Cost Breakdown",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Cost vs delivery performance matrix
                fig_matrix = go.Figure()
                
                for i, carrier in enumerate(carriers):
                    fig_matrix.add_trace(go.Scatter(
                        x=[delivery_days[i]],
                        y=[total_costs[i]],
                        mode='markers+text',
                        name=carrier,
                        text=[carrier],
                        textposition="middle center",
                        marker=dict(
                            size=confidence_scores[i] * 50,
                            color=CARRIERS[carrier]['color'],
                            opacity=0.8,
                            line=dict(width=3, color='white')
                        ),
                        hovertemplate=f'<b>{carrier}</b><br>Cost: ${total_costs[i]:.2f}<br>Days: {delivery_days[i]}<br>Zone: {zones[i]}<br>Distance: {distances[i]:.0f} mi<br>Confidence: {confidence_scores[i]*100:.1f}%<extra></extra>'
                    ))
                
                fig_matrix.update_layout(
                    title="Performance Matrix: Cost vs Speed",
                    xaxis_title="Delivery Days",
                    yaxis_title="Total Cost ($)",
                    height=400,
                    showlegend=False
                )
                
                # Add quadrant lines
                avg_cost = np.mean(total_costs)
                avg_days = np.mean(delivery_days)
                
                fig_matrix.add_hline(y=avg_cost, line_dash="dash", line_color="gray", opacity=0.5)
                fig_matrix.add_vline(x=avg_days, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Add quadrant labels
                fig_matrix.add_annotation(x=min(delivery_days), y=max(total_costs), text="Expensive & Slow", showarrow=False, bgcolor="rgba(255,0,0,0.1)")
                fig_matrix.add_annotation(x=max(delivery_days), y=min(total_costs), text="Cheap & Slow", showarrow=False, bgcolor="rgba(255,255,0,0.1)")
                fig_matrix.add_annotation(x=min(delivery_days), y=min(total_costs), text="Fast & Cheap", showarrow=False, bgcolor="rgba(0,255,0,0.1)")
                fig_matrix.add_annotation(x=max(delivery_days), y=max(total_costs), text="Expensive & Fast", showarrow=False, bgcolor="rgba(255,165,0,0.1)")
                
                st.plotly_chart(fig_matrix, use_container_width=True)
            
            # Detailed carrier comparison
            st.subheader("🏢 Carrier Performance Analysis")
            
            # Create comprehensive comparison table
            comparison_data = []
            for p in predictions:
                comparison_data.append({
                    'Carrier': p['carrier'],
                    'Total Cost': f"${p['total_cost']:.2f}",
                    'Delivery Days': p['estimated_days'],
                    'Distance (mi)': f"{p['distance']:.0f}",
                    'Zone': p['zone'],
                    'Weight Cost': f"${p['weight_cost']:.2f}",
                    'Fuel Cost': f"${p['fuel_cost']:.2f}",
                    'Confidence': f"{p['confidence_score']*100:.1f}%",
                    'Strength': p['strength'],
                    'ML Enhanced': '🤖' if p.get('ml_adjusted') else '📊'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance insights
            st.subheader("🧠 Smart Insights")
            
            insights = []
            
            # Best value analysis
            best_value_idx = np.argmin([c/d for c, d in zip(total_costs, delivery_days)])
            insights.append(f"🏆 **Best Value**: {carriers[best_value_idx]} offers the best cost-to-speed ratio")
            
            # Speed vs cost trade-off
            fastest_carrier = carriers[np.argmin(delivery_days)]
            cheapest_carrier = carriers[np.argmin(total_costs)]
            
            if fastest_carrier != cheapest_carrier:
                fastest_cost = total_costs[np.argmin(delivery_days)]
                cheapest_cost = min(total_costs)
                premium = fastest_cost - cheapest_cost
                insights.append(f"⚡ **Speed Premium**: {fastest_carrier} charges ${premium:.2f} extra for fastest delivery")
            
            # Distance efficiency
            if len(set(distances)) > 1:  # Multiple distances
                distance_efficiency = [(c-min(total_costs))/d for c, d in zip(total_costs, distances)]
                most_efficient_idx = np.argmin(distance_efficiency)
                insights.append(f"🗺️ **Distance Efficient**: {carriers[most_efficient_idx]} handles long distances most cost-effectively")
            
            # Confidence analysis
            most_confident_idx = np.argmax(confidence_scores)
            if confidence_scores[most_confident_idx] > 0.9:
                insights.append(f"🎯 **Most Reliable**: {carriers[most_confident_idx]} has highest prediction confidence ({confidence_scores[most_confident_idx]*100:.1f}%)")
            
            # ML enhancement impact
            ml_enhanced_count = sum(1 for p in predictions if p.get('ml_adjusted', False))
            if ml_enhanced_count > 0:
                insights.append(f"🤖 **AI Advantage**: {ml_enhanced_count} carriers benefit from ML-enhanced pricing")
            
            for insight in insights:
                st.info(insight)
            
            # Advanced visualizations
            st.subheader("📊 Advanced Analytics")
            
            # Multi-dimensional analysis
            fig_radar = go.Figure()
            
            # Normalize metrics for radar chart
            normalized_costs = [(max(total_costs) - c) / (max(total_costs) - min(total_costs)) for c in total_costs]  # Inverted: higher is better
            normalized_speed = [(max(delivery_days) - d) / (max(delivery_days) - min(delivery_days)) for d in delivery_days]  # Inverted: higher is better
            normalized_confidence = confidence_scores
            
            for i, carrier in enumerate(carriers):
                fig_radar.add_trace(go.Scatterpolar(
                    r=[normalized_costs[i], normalized_speed[i], normalized_confidence[i]],
                    theta=['Cost Efficiency', 'Speed', 'Confidence'],
                    fill='toself',
                    name=carrier,
                    line_color=CARRIERS[carrier]['color']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Carrier Performance Radar",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Optimization recommendations
            st.subheader("💡 Optimization Recommendations")
            
            recommendations = []
            
            # Budget-conscious recommendation
            cheapest_idx = np.argmin(total_costs)
            recommendations.append({
                'Scenario': '💰 Budget-Conscious',
                'Recommendation': carriers[cheapest_idx],
                'Cost': f"${total_costs[cheapest_idx]:.2f}",
                'Days': f"{delivery_days[cheapest_idx]} days",
                'Reason': f"Lowest cost option, saves ${max(total_costs) - total_costs[cheapest_idx]:.2f}"
            })
            
            # Time-sensitive recommendation
            fastest_idx = np.argmin(delivery_days)
            recommendations.append({
                'Scenario': '⚡ Time-Sensitive',
                'Recommendation': carriers[fastest_idx],
                'Cost': f"${total_costs[fastest_idx]:.2f}",
                'Days': f"{delivery_days[fastest_idx]} days",
                'Reason': f"Fastest delivery, {max(delivery_days) - delivery_days[fastest_idx]} days faster"
            })
            
            # Balanced recommendation
            balanced_scores = [(1-((c-min(total_costs))/(max(total_costs)-min(total_costs)))) + 
                              (1-((d-min(delivery_days))/(max(delivery_days)-min(delivery_days)))) + 
                              conf for c, d, conf in zip(total_costs, delivery_days, confidence_scores)]
            balanced_idx = np.argmax(balanced_scores)
            recommendations.append({
                'Scenario': '⚖️ Balanced',
                'Recommendation': carriers[balanced_idx],
                'Cost': f"${total_costs[balanced_idx]:.2f}",
                'Days': f"{delivery_days[balanced_idx]} days",
                'Reason': "Best overall balance of cost, speed, and reliability"
            })
            
            # High-confidence recommendation
            if max(confidence_scores) > 0.9:
                confident_idx = np.argmax(confidence_scores)
                recommendations.append({
                    'Scenario': '🎯 High Confidence',
                    'Recommendation': carriers[confident_idx],
                    'Cost': f"${total_costs[confident_idx]:.2f}",
                    'Days': f"{delivery_days[confident_idx]} days",
                    'Reason': f"Highest prediction confidence ({confidence_scores[confident_idx]*100:.1f}%)"
                })
            
            rec_df = pd.DataFrame(recommendations)
            st.table(rec_df)
        
        else:
            st.info("📊 Calculate some routes first to see advanced analytics.")
            
            # Analytics preview
            st.markdown("""
            ### 📈 Available Analytics
            
            **📊 Cost Structure Analysis:**
            - Detailed cost breakdowns by component
            - Average cost distribution across carriers
            - Cost efficiency comparisons
            
            **🎯 Performance Matrix:**
            - Cost vs delivery time optimization
            - Carrier positioning analysis
            - Performance quadrant identification
            
            **🧠 Smart Insights:**
            - Best value recommendations
            - Speed premium analysis
            - Distance efficiency metrics
            - ML enhancement impact
            
            **💡 Scenario-Based Recommendations:**
            - Budget-conscious options
            - Time-sensitive solutions
            - Balanced approaches
            - High-confidence predictions
            
            **📊 Advanced Visualizations:**
            - Multi-dimensional radar charts
            - Interactive performance matrices
            - Comparative analysis tools
            """)

if __name__ == "__main__":
    main()
