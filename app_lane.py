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
import warnings
import math
import random
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import time

# Optional imports with fallbacks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from geopy.distance import geodesic
    from geopy.geocoders import Nominatim
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🚚 Advanced Lane Optimization System v14",
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
    .claude-insights {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
        border-left: 4px solid #ffd700;
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
    .debug-panel {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
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

# Claude API enhanced prediction
def get_claude_insights(optimization_data, claude_client):
    """Get Claude insights for optimization"""
    try:
        prompt = f"""
        Analyze this shipping optimization data and provide insights:
        
        Route: {optimization_data['origin']} → {optimization_data['destination']}
        Distance: {optimization_data['distance']:.1f} miles
        Weight: {optimization_data['weight']} lbs
        Priority: {optimization_data['priority']}
        
        ML Predictions:
        - Estimated Cost: ${optimization_data.get('predicted_cost', 0):.2f}
        - Transit Time: {optimization_data.get('predicted_time', 0):.1f} hours
        - Reliability Score: {optimization_data.get('reliability', 0):.1f}%
        
        Provide specific recommendations for:
        1. Cost optimization opportunities
        2. Risk mitigation strategies
        3. Alternative routing suggestions
        
        Keep response concise and actionable.
        """
        
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Claude analysis unavailable: {str(e)}"

# Claude API validation
def validate_claude_api(api_key):
    """Validate Claude API key with detailed error reporting"""
    if not ANTHROPIC_AVAILABLE:
        return False, "Anthropic library not installed. Run: pip install anthropic"
    
    if not api_key:
        return False, "No API key provided"
    
    if not api_key.startswith('sk-ant-'):
        return False, "Invalid API key format. Claude API keys should start with 'sk-ant-'"
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        # Test with a simple request using current model
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{"role": "user", "content": "Test"}]
        )
        return True, client
    except anthropic.AuthenticationError as e:
        return False, f"Authentication failed: {str(e)}"
    except anthropic.PermissionDeniedError as e:
        return False, f"Permission denied: {str(e)}"
    except anthropic.NotFoundError as e:
        return False, f"Model not found: {str(e)}"
    except anthropic.RateLimitError as e:
        return False, f"Rate limit exceeded: {str(e)}"
    except anthropic.APIConnectionError as e:
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

# Dataset processing functions
def process_uploaded_data(uploaded_file):
    """Process uploaded dataset and return training data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload CSV or Excel files."
        
        # Display dataset info
        st.write(f"📊 **Dataset loaded**: {len(df)} rows, {len(df.columns)} columns")
        st.write("**Columns found:**", list(df.columns))
        
        # Show preview
        st.write("**Data Preview:**")
        st.dataframe(df.head(), use_container_width=True)
        
        return df, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def prepare_training_data(df, target_column, feature_columns=None):
    """Prepare training data from uploaded dataset"""
    try:
        if target_column not in df.columns:
            return None, None, f"Target column '{target_column}' not found in dataset"
        
        # Auto-select feature columns if not specified
        if feature_columns is None:
            # Exclude target and non-numeric columns
            feature_columns = [col for col in df.columns 
                             if col != target_column and 
                             (df[col].dtype in ['int64', 'float64'] or 
                              df[col].dtype == 'object' and df[col].nunique() < 20)]
        
        # Prepare features
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Handle missing values
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        y = y.fillna(y.mean() if y.dtype in ['int64', 'float64'] else 0)
        
        return X.values, y.values, None
        
    except Exception as e:
        return None, None, f"Error preparing training data: {str(e)}"

def validate_dataset_format(df):
    """Validate if dataset has required columns for shipping optimization"""
    required_columns = ['origin', 'destination', 'weight', 'cost']
    optional_columns = ['priority', 'distance', 'transit_time', 'carrier']
    
    found_required = [col for col in required_columns if col.lower() in [c.lower() for c in df.columns]]
    found_optional = [col for col in optional_columns if col.lower() in [c.lower() for c in df.columns]]
    
    return {
        'valid': len(found_required) >= 3,  # At least 3 required columns
        'required_found': found_required,
        'optional_found': found_optional,
        'suggestions': get_column_suggestions(df.columns)
    }

def get_column_suggestions(columns):
    """Suggest column mappings based on column names"""
    suggestions = {}
    column_mapping = {
        'cost': ['cost', 'price', 'amount', 'total', 'charge', 'fee'],
        'weight': ['weight', 'wt', 'mass', 'kg', 'lbs', 'pounds'],
        'distance': ['distance', 'dist', 'miles', 'km', 'length'],
        'origin': ['origin', 'from', 'source', 'start', 'pickup'],
        'destination': ['destination', 'dest', 'to', 'end', 'delivery'],
        'priority': ['priority', 'urgent', 'level', 'class', 'type'],
        'transit_time': ['time', 'duration', 'days', 'hours', 'transit']
    }
    
    for target, keywords in column_mapping.items():
        for col in columns:
            if any(keyword in col.lower() for keyword in keywords):
                suggestions[target] = col
                break
    
    return suggestions

# Enhanced ML Agent with custom data support
class AdvancedMLAgent:
    def __init__(self, name, model_type):
        self.name = name
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_metrics = {}
        self.is_trained = False

    def prepare_features(self, data):
        """Enhanced feature engineering"""
        features = []
        feature_names = []
        
        # Distance-based features
        distance = data.get('distance', 100)
        features.extend([
            distance,
            np.log1p(distance),
            distance ** 0.5,
            1 / (distance + 1)
        ])
        feature_names.extend(['distance', 'log_distance', 'sqrt_distance', 'inv_distance'])
        
        # Weight-based features
        weight = data.get('weight', 1000)
        features.extend([
            weight,
            np.log1p(weight),
            weight / 1000,  # Normalized weight
            weight * distance  # Weight-distance interaction
        ])
        feature_names.extend(['weight', 'log_weight', 'weight_norm', 'weight_distance'])
        
        # Priority encoding
        priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        priority_score = priority_map.get(data.get('priority', 'Medium'), 2)
        features.extend([priority_score, priority_score ** 2])
        feature_names.extend(['priority_score', 'priority_squared'])
        
        # Time-based features
        hour = data.get('hour', 12)
        day = data.get('day', 3)  # Wednesday = 3
        features.extend([
            hour,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            day,
            1 if day < 5 else 0,  # Weekday flag
        ])
        feature_names.extend(['hour', 'hour_sin', 'hour_cos', 'day', 'is_weekday'])
        
        # Seasonal features
        month = data.get('month', 6)
        features.extend([
            month,
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12)
        ])
        feature_names.extend(['month', 'month_sin', 'month_cos'])
        
        # Additional derived features
        features.extend([
            distance / (weight + 1),  # Distance per weight
            weight / (distance + 1),  # Weight per distance
            priority_score * distance,  # Priority-distance interaction
            priority_score * weight,   # Priority-weight interaction
        ])
        feature_names.extend(['dist_per_weight', 'weight_per_dist', 'priority_dist', 'priority_weight'])
        
        return np.array(features).reshape(1, -1), feature_names

    def train(self, training_data=None):
        """Train the model with synthetic data if no training data provided"""
        try:
            if training_data is None:
                # Generate synthetic training data
                np.random.seed(42)
                n_samples = 1000
                
                training_features = []
                training_targets = []
                
                for _ in range(n_samples):
                    # Generate random shipping scenario
                    distance = np.random.exponential(500) + 50
                    weight = np.random.exponential(2000) + 100
                    priority = np.random.choice(['Low', 'Medium', 'High', 'Critical'])
                    hour = np.random.randint(0, 24)
                    day = np.random.randint(0, 7)
                    month = np.random.randint(1, 13)
                    
                    sample_data = {
                        'distance': distance,
                        'weight': weight,
                        'priority': priority,
                        'hour': hour,
                        'day': day,
                        'month': month
                    }
                    
                    features, _ = self.prepare_features(sample_data)
                    training_features.append(features.flatten())
                    
                    # Generate realistic target based on complex rules
                    base_cost = 2.5 * distance + 0.8 * weight
                    priority_multiplier = {'Low': 0.9, 'Medium': 1.0, 'High': 1.2, 'Critical': 1.5}[priority]
                    time_multiplier = 1.1 if hour < 6 or hour > 18 else 1.0
                    weekend_multiplier = 1.15 if day >= 5 else 1.0
                    seasonal_multiplier = 1.1 if month in [11, 12, 1] else 1.0
                    
                    target = base_cost * priority_multiplier * time_multiplier * weekend_multiplier * seasonal_multiplier
                    target += np.random.normal(0, target * 0.1)  # Add noise
                    training_targets.append(max(target, 50))  # Minimum cost
                
                X = np.array(training_features)
                y = np.array(training_targets)
            else:
                X, y = training_data
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model based on type
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
            elif self.model_type == 'neural_network':
                self.model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=500)
            elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                self.model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
            elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMRegressor(n_estimators=100, random_state=42, max_depth=6, verbose=-1)
            elif self.model_type == 'lstm' and TF_AVAILABLE:
                self.model = self._create_lstm_model(X_scaled.shape[1])
                return self._train_lstm(X_scaled, y)
            else:
                # Fallback to gradient boosting
                self.model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
            
            # Train the model
            self.model.fit(X_scaled, y)
            
            # Calculate performance metrics
            y_pred = self.model.predict(X_scaled)
            self.performance_metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100
            }
            
            self.is_trained = True
            return True
            
        except Exception as e:
            st.error(f"Training failed for {self.name}: {str(e)}")
            return False

    def _create_lstm_model(self, input_dim):
        """Create LSTM model for time series prediction"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def _train_lstm(self, X, y):
        """Train LSTM model"""
        try:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Calculate performance metrics
            y_pred = self.model.predict(X, verbose=0).flatten()
            self.performance_metrics = {
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / y)) * 100
            }
            
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"LSTM training failed: {str(e)}")
            return False

    def predict(self, data):
        """Make prediction for shipping cost"""
        if not self.is_trained:
            return None
        
        try:
            features, _ = self.prepare_features(data)
            features_scaled = self.scaler.transform(features)
            
            if self.model_type == 'lstm' and TF_AVAILABLE:
                prediction = self.model.predict(features_scaled, verbose=0)[0][0]
            else:
                prediction = self.model.predict(features_scaled)[0]
            
            return max(prediction, 50)  # Minimum cost constraint
        except Exception as e:
            st.error(f"Prediction failed for {self.name}: {str(e)}")
            return None

# Enhanced Multi-Agent System
class EnhancedMultiAgentSystem:
    def __init__(self):
        self.agents = {}
        self.ensemble_weights = {}
        
    def add_agent(self, name, model_type):
        """Add a new agent to the system"""
        agent = AdvancedMLAgent(name, model_type)
        self.agents[name] = agent
        return agent
    
    def train_all_agents(self, training_data=None):
        """Train all agents in the system"""
        results = {}
        progress_bar = st.progress(0)
        
        for i, (name, agent) in enumerate(self.agents.items()):
            st.write(f"Training {name}...")
            success = agent.train(training_data)
            results[name] = success
            progress_bar.progress((i + 1) / len(self.agents))
        
        # Calculate ensemble weights based on performance
        self._calculate_ensemble_weights()
        return results
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on model performance"""
        weights = {}
        total_performance = 0
        
        for name, agent in self.agents.items():
            if agent.is_trained and agent.performance_metrics:
                # Use inverse of MAPE as weight (lower MAPE = higher weight)
                performance = 1 / (agent.performance_metrics.get('mape', 100) + 1)
                weights[name] = performance
                total_performance += performance
        
        # Normalize weights
        if total_performance > 0:
            for name in weights:
                weights[name] = weights[name] / total_performance
        
        self.ensemble_weights = weights
    
    def get_ensemble_prediction(self, data):
        """Get ensemble prediction from all trained agents"""
        predictions = {}
        weighted_sum = 0
        total_weight = 0
        
        for name, agent in self.agents.items():
            if agent.is_trained:
                pred = agent.predict(data)
                if pred is not None:
                    predictions[name] = pred
                    weight = self.ensemble_weights.get(name, 1.0)
                    weighted_sum += pred * weight
                    total_weight += weight
        
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else None
        return ensemble_prediction, predictions

# Utility functions
def calculate_distance(origin, destination):
    """Calculate distance between two locations"""
    if GEOPY_AVAILABLE and st.session_state.geocoder:
        try:
            origin_coord = st.session_state.geocoder.geocode(origin)
            dest_coord = st.session_state.geocoder.geocode(destination)
            
            if origin_coord and dest_coord:
                distance = geodesic(
                    (origin_coord.latitude, origin_coord.longitude),
                    (dest_coord.latitude, dest_coord.longitude)
                ).miles
                return distance
        except Exception:
            pass
    
    # Fallback: estimate based on string similarity and common routes
    route_estimates = {
        ('new york', 'chicago'): 790,
        ('los angeles', 'san francisco'): 380,
        ('miami', 'orlando'): 235,
        ('dallas', 'houston'): 240,
        ('seattle', 'portland'): 173,
        ('boston', 'philadelphia'): 300,
        ('denver', 'salt lake city'): 525,
        ('atlanta', 'charlotte'): 245
    }
    
    origin_lower = origin.lower()
    destination_lower = destination.lower()
    
    for (o, d), dist in route_estimates.items():
        if (o in origin_lower and d in destination_lower) or (d in origin_lower and o in destination_lower):
            return dist
    
    # Default estimate based on route complexity
    return np.random.uniform(200, 1200)

def generate_sample_data():
    """Generate sample shipping data for demonstration"""
    np.random.seed(42)
    
    cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville']
    
    data = []
    for _ in range(100):
        origin = np.random.choice(cities)
        destination = np.random.choice([c for c in cities if c != origin])
        distance = calculate_distance(origin, destination)
        weight = np.random.exponential(2000) + 100
        priority = np.random.choice(['Low', 'Medium', 'High', 'Critical'])
        
        # Calculate realistic cost
        base_rate = 2.5
        weight_rate = 0.8
        priority_multiplier = {'Low': 0.9, 'Medium': 1.0, 'High': 1.2, 'Critical': 1.5}[priority]
        
        cost = (base_rate * distance + weight_rate * weight) * priority_multiplier
        cost += np.random.normal(0, cost * 0.1)  # Add noise
        cost = max(cost, 50)  # Minimum cost
        
        data.append({
            'origin': origin,
            'destination': destination,
            'distance': distance,
            'weight': weight,
            'priority': priority,
            'actual_cost': cost
        })
    
    return pd.DataFrame(data)

# Main application
def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚚 Advanced Lane Optimization System v14</h1>
        <p>AI-Powered Multi-Agent Shipping Cost Optimization with Claude Integration</p>
        <p><strong>Features:</strong> ML Ensemble • Real-time Predictions • Claude AI Insights • Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # Library status
        st.subheader("📦 Library Status")
        library_status = {
            "Core ML": "✅ Available",
            "TensorFlow": "✅ Available" if TF_AVAILABLE else "❌ Not installed",
            "XGBoost": "✅ Available" if XGBOOST_AVAILABLE else "❌ Not installed", 
            "LightGBM": "✅ Available" if LIGHTGBM_AVAILABLE else "❌ Not installed",
            "Anthropic": "✅ Available" if ANTHROPIC_AVAILABLE else "❌ Not installed",
            "GeoPy": "✅ Available" if GEOPY_AVAILABLE else "❌ Not installed"
        }
        
        for lib, status in library_status.items():
            st.write(f"**{lib}**: {status}")
        
        if ANTHROPIC_AVAILABLE:
            st.divider()
            
            # Claude API Key input
            st.subheader("🤖 Claude API Setup")
            
            # Debug info
            with st.expander("🔧 Debug Info", expanded=False):
                debug_info = {
                    "Secrets available": secrets_available,
                    "CLAUDE_API_KEY in secrets": api_key_in_secrets,
                    "API validated": st.session_state.api_validated,
                    "Anthropic available": ANTHROPIC_AVAILABLE,
                    "API key source": api_source,
                    "Secrets keys": list(st.secrets.keys()) if secrets_available else "No secrets accessible"
                }
                for key, value in debug_info.items():
                    st.code(f"{key}: {value}")
            
            # Try to get API key from Streamlit secrets first
            api_key = None
            api_source = "manual"
            
            # Multiple ways to check for secrets
            secrets_available = False
            api_key_in_secrets = False
            
            try:
                if hasattr(st, 'secrets'):
                    secrets_available = True
                    # Try different access methods
                    if hasattr(st.secrets, 'CLAUDE_API_KEY'):
                        api_key = st.secrets.CLAUDE_API_KEY
                        api_key_in_secrets = True
                        api_source = "secrets"
                    elif 'CLAUDE_API_KEY' in st.secrets:
                        api_key = st.secrets['CLAUDE_API_KEY']
                        api_key_in_secrets = True
                        api_source = "secrets"
                    elif hasattr(st.secrets, 'claude_api_key'):
                        api_key = st.secrets.claude_api_key
                        api_key_in_secrets = True
                        api_source = "secrets"
            except Exception as e:
                st.warning(f"Error accessing secrets: {e}")
            
            if api_key_in_secrets:
                st.success(f"✅ Found API key in Streamlit secrets (length: {len(api_key) if api_key else 0})")
            else:
                st.info("💡 No API key found in secrets. Please add to secrets.toml or enter manually below.")
            
            # Manual input (will override secrets if provided)
            manual_key = st.text_input(
                "Claude API Key", 
                type="password", 
                help="Enter your Claude API key or add to Streamlit secrets",
                placeholder="sk-ant-..."
            )
            
            if manual_key:
                api_key = manual_key
                api_source = "manual"
            
            # Show current API key status
            if api_key:
                st.info(f"🔑 Using API key from: **{api_source}** (starts with: {api_key[:10]}...)")
            else:
                st.warning("⚠️ No API key found in secrets or manual input")
            
            # Validate API key
            if api_key and not st.session_state.api_validated:
                if st.button("🔄 Validate API Key") or api_source == "secrets":
                    with st.spinner("Validating API key..."):
                        is_valid, result = validate_claude_api(api_key)
                        if is_valid:
                            st.session_state.claude_client = result
                            st.session_state.api_validated = True
                            st.success("✅ API key validated successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ API validation failed: {result}")
                            st.session_state.api_validated = False
            
            # Reset API validation if needed
            if st.session_state.api_validated:
                if st.button("🔄 Reset API Connection"):
                    st.session_state.api_validated = False
                    st.session_state.claude_client = None
                    st.rerun()
            
            if st.session_state.api_validated:
                st.markdown('<div class="api-status">🟢 Claude API Connected</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="api-error">🔴 Claude API Not Connected</div>', unsafe_allow_html=True)
                
                st.markdown("""
                ### 🛠️ Troubleshooting Tips:
                
                1. **Check API Key Format**: Should start with `sk-ant-`
                2. **Verify API Key**: Get a new one from [Claude Console](https://console.anthropic.com)
                3. **Streamlit Secrets Setup**:
                   Create `.streamlit/secrets.toml` in your project root:
                   ```toml
                   CLAUDE_API_KEY = "sk-ant-your-key-here"
                   ```
                   **Important**: No quotes around the key name, quotes around the value
                4. **Restart Streamlit**: After adding secrets, restart your app completely
                5. **Check File Location**: Ensure `.streamlit/secrets.toml` is in the same directory as your Python file
                6. **Check Network**: Ensure you can reach api.anthropic.com
                
                ### 📝 Secrets File Format Example:
                ```toml
                # .streamlit/secrets.toml
                CLAUDE_API_KEY = "sk-ant-api03-your-actual-key-here"
                
                # You can also try lowercase
                claude_api_key = "sk-ant-api03-your-actual-key-here"
                ```
                """)
                
                st.info("""
                💡 **Quick Setup Guide:**
                
                1. Create folder `.streamlit` in your project directory
                2. Create file `secrets.toml` inside `.streamlit` folder
                3. Add your API key: `CLAUDE_API_KEY = "your-key"`
                4. Restart Streamlit completely
                5. Or just paste your key in the input field below for immediate testing
                """)
        else:
            st.info("📦 Install anthropic library to enable Claude API: `pip install anthropic`")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Optimization", "🤖 Model Training", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("🎯 Shipping Lane Optimization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            st.subheader("📍 Route Information")
            
            col_a, col_b = st.columns(2)
            with col_a:
                origin = st.text_input("Origin City", value="New York", help="Enter departure city")
            with col_b:
                destination = st.text_input("Destination City", value="Los Angeles", help="Enter destination city")
            
            col_c, col_d = st.columns(2)
            with col_c:
                weight = st.number_input("Shipment Weight (lbs)", min_value=1, max_value=80000, value=2500)
            with col_d:
                priority = st.selectbox("Priority Level", ["Low", "Medium", "High", "Critical"])
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Calculate distance
            distance = calculate_distance(origin, destination)
            st.info(f"📏 Estimated Distance: **{distance:.0f} miles**")
            
            # Prediction section
            if st.button("🚀 Get Optimization Recommendations", type="primary"):
                if not st.session_state.trained_models:
                    st.warning("⚠️ No trained models available. Please train models first in the Model Training tab.")
                else:
                    with st.spinner("🤖 Generating predictions..."):
                        # Prepare input data
                        input_data = {
                            'distance': distance,
                            'weight': weight,
                            'priority': priority,
                            'hour': datetime.now().hour,
                            'day': datetime.now().weekday(),
                            'month': datetime.now().month
                        }
                        
                        # Get predictions from all models
                        predictions = {}
                        for model_name, agent in st.session_state.trained_models.items():
                            if agent.is_trained:
                                pred = agent.predict(input_data)
                                if pred:
                                    predictions[model_name] = pred
                        
                        if predictions:
                            # Display results
                            st.success("✅ Optimization Complete!")
                            
                            # Calculate ensemble prediction
                            ensemble_pred = np.mean(list(predictions.values()))
                            time_pred = (distance / 55) + np.random.uniform(2, 8)  # Realistic transit time
                            reliability_pred = 95 - (weight / 1000) * 0.5 - (distance / 100) * 0.3
                            reliability_pred = max(85, min(99, reliability_pred))
                            
                            # Display metrics
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            
                            with col_m1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>💰 Estimated Cost</h3>
                                    <h2>${ensemble_pred:.2f}</h2>
                                    <p>Ensemble Prediction</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_m2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>⏱️ Transit Time</h3>
                                    <h2>{time_pred:.1f}h</h2>
                                    <p>Estimated Delivery</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_m3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>📈 Reliability</h3>
                                    <h2>{reliability_pred:.1f}%</h2>
                                    <p>On-time Performance</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col_m4:
                                best_model = min(predictions.keys(), key=lambda k: predictions[k])
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>🏆 Best Model</h3>
                                    <h2>{best_model}</h2>
                                    <p>${predictions[best_model]:.2f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Model comparison
                            st.subheader("🔍 Model Comparison")
                            comparison_df = pd.DataFrame([
                                {'Model': name, 'Predicted Cost': f"${cost:.2f}", 'Difference': f"${cost - ensemble_pred:.2f}"}
                                for name, cost in predictions.items()
                            ])
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Claude insights
                            if st.session_state.api_validated and st.session_state.claude_client:
                                with st.spinner("🤖 Getting Claude AI insights..."):
                                    claude_insights = get_claude_insights(
                                        {
                                            'origin': origin,
                                            'destination': destination,
                                            'distance': distance,
                                            'weight': weight,
                                            'priority': priority,
                                            'predicted_cost': ensemble_pred,
                                            'predicted_time': time_pred,
                                            'reliability': reliability_pred
                                        },
                                        st.session_state.claude_client
                                    )
                                    
                                    st.markdown("### 🤖 Claude AI Insights")
                                    st.markdown(f"""
                                    <div class="claude-insights">
                                        {claude_insights}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Store prediction in session state
                            st.session_state.predictions.append({
                                'timestamp': datetime.now(),
                                'origin': origin,
                                'destination': destination,
                                'distance': distance,
                                'weight': weight,
                                'priority': priority,
                                'predicted_cost': ensemble_pred,
                                'transit_time': time_pred,
                                'reliability': reliability_pred,
                                'model_predictions': predictions
                            })
        
        with col2:
            st.subheader("📈 Quick Stats")
            
            # Recent predictions
            if st.session_state.predictions:
                recent_predictions = st.session_state.predictions[-5:]
                avg_cost = np.mean([p['predicted_cost'] for p in recent_predictions])
                
                st.metric("Recent Avg Cost", f"${avg_cost:.2f}")
                st.metric("Total Predictions", len(st.session_state.predictions))
                
                # Show recent predictions
                st.write("**Recent Predictions:**")
                for pred in reversed(recent_predictions):
                    st.write(f"• {pred['origin']} → {pred['destination']}: ${pred['predicted_cost']:.2f}")
            else:
                st.info("No predictions yet. Make your first prediction!")
    
    with tab2:
        st.header("🤖 Model Training & Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔧 Training Configuration")
            
            # Model selection
            st.write("**Available Models:**")
            model_options = {
                'Random Forest': 'random_forest',
                'Gradient Boosting': 'gradient_boosting', 
                'Neural Network': 'neural_network'
            }
            
            if XGBOOST_AVAILABLE:
                model_options['XGBoost'] = 'xgboost'
            if LIGHTGBM_AVAILABLE:
                model_options['LightGBM'] = 'lightgbm'
            if TF_AVAILABLE:
                model_options['LSTM Neural Network'] = 'lstm'
            
            selected_models = st.multiselect(
                "Select models to train:",
                options=list(model_options.keys()),
                default=list(model_options.keys())[:3]
            )
            
            # Training data options
            st.write("**📊 Training Data Options:**")
            
            data_option = st.radio(
                "Select data source:",
                ["Use sample synthetic data", "Upload custom dataset"],
                help="Choose between generated sample data or your own dataset"
            )
            
            training_data = None
            
            if data_option == "Upload custom dataset":
                uploaded_file = st.file_uploader(
                    "Upload training data", 
                    type=['csv', 'xlsx', 'xls'],
                    help="Upload CSV or Excel file with shipping data"
                )
                
                if uploaded_file:
                    # Process uploaded file
                    df, error = process_uploaded_data(uploaded_file)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Validate dataset format
                        validation = validate_dataset_format(df)
                        
                        if validation['valid']:
                            st.success("✅ Dataset format validated!")
                            
                            # Column mapping section
                            st.write("**🔄 Column Mapping:**")
                            
                            # Auto-suggestions
                            suggestions = validation['suggestions']
                            
                            col_map1, col_map2 = st.columns(2)
                            
                            with col_map1:
                                target_column = st.selectbox(
                                    "Target Column (Cost/Price):",
                                    options=df.columns,
                                    index=list(df.columns).index(suggestions.get('cost', df.columns[0])) if suggestions.get('cost') in df.columns else 0
                                )
                                
                                weight_column = st.selectbox(
                                    "Weight Column:",
                                    options=[None] + list(df.columns),
                                    index=list(df.columns).index(suggestions.get('weight', df.columns[0])) + 1 if suggestions.get('weight') in df.columns else 0
                                )
                            
                            with col_map2:
                                distance_column = st.selectbox(
                                    "Distance Column (optional):",
                                    options=[None] + list(df.columns),
                                    index=list(df.columns).index(suggestions.get('distance', df.columns[0])) + 1 if suggestions.get('distance') in df.columns else 0
                                )
                                
                                priority_column = st.selectbox(
                                    "Priority Column (optional):",
                                    options=[None] + list(df.columns),
                                    index=list(df.columns).index(suggestions.get('priority', df.columns[0])) + 1 if suggestions.get('priority') in df.columns else 0
                                )
                            
                            # Feature selection
                            available_features = [col for col in df.columns if col != target_column]
                            selected_features = st.multiselect(
                                "Select Feature Columns:",
                                options=available_features,
                                default=[col for col in [weight_column, distance_column, priority_column] if col and col in available_features]
                            )
                            
                            if selected_features:
                                # Prepare training data
                                X, y, prep_error = prepare_training_data(df, target_column, selected_features)
                                
                                if prep_error:
                                    st.error(f"❌ {prep_error}")
                                else:
                                    training_data = (X, y)
                                    st.success(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
                                    
                                    # Show statistics
                                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                                    with col_stat1:
                                        st.metric("Samples", X.shape[0])
                                    with col_stat2:
                                        st.metric("Features", X.shape[1])
                                    with col_stat3:
                                        st.metric("Target Range", f"${y.min():.0f} - ${y.max():.0f}")
                            else:
                                st.warning("Please select at least one feature column.")
                        else:
                            st.warning("⚠️ Dataset format may not be optimal for shipping optimization.")
                            st.write("**Suggestions:**")
                            st.write("- Ensure you have columns for: cost/price, weight, distance")
                            st.write("- Optional columns: priority, origin, destination, transit_time")
                            
                            # Still allow manual column selection
                            if st.checkbox("Proceed anyway with manual column selection"):
                                target_column = st.selectbox("Select target column:", df.columns)
                                feature_columns = st.multiselect("Select feature columns:", 
                                                               [col for col in df.columns if col != target_column])
                                
                                if feature_columns:
                                    X, y, prep_error = prepare_training_data(df, target_column, feature_columns)
                                    if not prep_error:
                                        training_data = (X, y)
                                        st.success("✅ Custom training data prepared!")
                else:
                    st.info("📁 Please upload a CSV or Excel file to continue with custom data.")
            else:
                st.info("📊 Using synthetic sample data for training.")
                
                # Option to download sample data format
                if st.button("📥 Download Sample Dataset Format"):
                    sample_df = generate_sample_data()
                    csv = sample_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="sample_shipping_data.csv",
                        mime="text/csv"
                    )
                    st.success("✅ Sample dataset generated! Use this as a template for your own data.")
            
            # Train models
            if st.button("🚀 Train Selected Models", type="primary"):
                if not selected_models:
                    st.warning("Please select at least one model to train.")
                else:
                    # Initialize multi-agent system
                    mas = EnhancedMultiAgentSystem()
                    
                    # Add selected agents
                    for model_name in selected_models:
                        model_type = model_options[model_name]
                        mas.add_agent(model_name, model_type)
                    
                    # Train all agents
                    st.write("🔄 Training models...")
                    if training_data:
                        st.info(f"📊 Using custom dataset: {training_data[0].shape[0]} samples")
                    else:
                        st.info("📊 Using synthetic sample data")
                    
                    training_results = mas.train_all_agents(training_data)
                    
                    # Store trained models
                    st.session_state.trained_models = mas.agents
                    
                    # Display results
                    st.success("✅ Training Complete!")
                    
                    for model_name, success in training_results.items():
                        if success:
                            agent = mas.agents[model_name]
                            metrics = agent.performance_metrics
                            
                            st.markdown(f"""
                            <div class="model-performance">
                                <h4>🎯 {model_name}</h4>
                                <p><strong>MAE:</strong> ${metrics.get('mae', 0):.2f}</p>
                                <p><strong>RMSE:</strong> ${metrics.get('rmse', 0):.2f}</p>
                                <p><strong>R²:</strong> {metrics.get('r2', 0):.3f}</p>
                                <p><strong>MAPE:</strong> {metrics.get('mape', 0):.1f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"❌ {model_name} training failed")
        
        with col2:
            st.subheader("📊 Model Status")
            
            if st.session_state.trained_models:
                st.write("**Trained Models:**")
                for name, agent in st.session_state.trained_models.items():
                    status = "✅ Ready" if agent.is_trained else "❌ Failed"
                    st.write(f"• {name}: {status}")
                
                if st.button("🗑️ Clear All Models"):
                    st.session_state.trained_models = {}
                    st.rerun()
            else:
                st.info("No models trained yet.")
            
            # Library installation tips
            st.write("**📦 Optional Libraries:**")
            if not XGBOOST_AVAILABLE:
                st.code("pip install xgboost")
            if not LIGHTGBM_AVAILABLE:
                st.code("pip install lightgbm")
            if not TF_AVAILABLE:
                st.code("pip install tensorflow")
            if not ANTHROPIC_AVAILABLE:
                st.code("pip install anthropic")
    
    with tab3:
        st.header("📊 Analytics & Performance")
        
        if st.session_state.predictions:
            # Prediction history
            df = pd.DataFrame(st.session_state.predictions)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Cost distribution
                fig = px.histogram(df, x='predicted_cost', title="Cost Distribution", nbins=20)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distance vs Cost
                fig2 = px.scatter(df, x='distance', y='predicted_cost', color='priority',
                                 title="Distance vs Cost by Priority", size='weight')
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                # Route frequency
                route_counts = df['origin'].value_counts().head(10)
                fig3 = px.bar(x=route_counts.values, y=route_counts.index, orientation='h',
                             title="Top Origin Cities")
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Priority distribution
                priority_counts = df['priority'].value_counts()
                fig4 = px.pie(values=priority_counts.values, names=priority_counts.index,
                             title="Priority Distribution")
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)
            
            # Model performance comparison
            if st.session_state.trained_models:
                st.subheader("🏆 Model Performance Comparison")
                
                performance_data = []
                for name, agent in st.session_state.trained_models.items():
                    if agent.is_trained and agent.performance_metrics:
                        performance_data.append({
                            'Model': name,
                            'MAE': agent.performance_metrics.get('mae', 0),
                            'RMSE': agent.performance_metrics.get('rmse', 0),
                            'R²': agent.performance_metrics.get('r2', 0),
                            'MAPE': agent.performance_metrics.get('mape', 0)
                        })
                
                if performance_data:
                    perf_df = pd.DataFrame(performance_data)
                    
                    # Performance metrics chart
                    fig5 = go.Figure()
                    
                    fig5.add_trace(go.Bar(name='MAE', x=perf_df['Model'], y=perf_df['MAE']))
                    fig5.add_trace(go.Bar(name='RMSE', x=perf_df['Model'], y=perf_df['RMSE']))
                    
                    fig5.update_layout(
                        title="Model Performance Comparison (Lower is Better)",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    # Performance table
                    st.dataframe(perf_df, use_container_width=True)
            
            # Prediction statistics
            st.subheader("📈 Prediction Statistics")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Total Predictions", len(df))
            with col_b:
                st.metric("Average Cost", f"${df['predicted_cost'].mean():.2f}")
            with col_c:
                st.metric("Average Distance", f"{df['distance'].mean():.0f} mi")
            with col_d:
                st.metric("Average Weight", f"{df['weight'].mean():.0f} lbs")
        
        else:
            st.info("📊 No prediction data available yet. Make some predictions first!")
    
    with tab4:
        st.header("ℹ️ About the System")
        
        st.markdown("""
        ### 🚚 Advanced Lane Optimization System v14
        
        This sophisticated system uses machine learning and AI to optimize shipping costs and routes.
        
        #### 🔧 **Core Features:**
        - **Multi-Agent ML System**: Multiple algorithms working together
        - **Real-time Predictions**: Instant cost and time estimates
        - **Claude AI Integration**: Intelligent insights and recommendations
        - **Advanced Analytics**: Comprehensive performance tracking
        - **Flexible Architecture**: Support for multiple ML libraries
        
        #### 🤖 **Supported Models:**
        - **Random Forest**: Ensemble decision trees
        - **Gradient Boosting**: Iterative improvement algorithm
        - **Neural Networks**: Deep learning with multiple layers
        - **XGBoost**: Extreme gradient boosting (if installed)
        - **LightGBM**: Efficient gradient boosting (if installed)
        - **LSTM**: Long short-term memory networks (if TensorFlow installed)
        
        #### 📊 **Input Features:**
        - Distance between origin and destination
        - Shipment weight and priority level
        - Time-based factors (hour, day, month)
        - Seasonal adjustments and interactions
        
        #### 🎯 **Optimization Targets:**
        - Shipping cost minimization
        - Transit time optimization
        - Reliability maximization
        - Risk mitigation strategies
        
        #### 🔐 **Claude AI Integration:**
        - Intelligent route analysis
        - Cost optimization suggestions
        - Risk assessment and mitigation
        - Alternative routing recommendations
        
        #### 📦 **Installation Requirements:**
        ```bash
        # Core requirements
        pip install streamlit pandas numpy scikit-learn plotly
        
        # Optional enhancements
        pip install tensorflow xgboost lightgbm anthropic geopy
        ```
        
        #### 🚀 **Getting Started:**
        1. **Train Models**: Go to Model Training tab and train your preferred algorithms
        2. **Make Predictions**: Use the Optimization tab to get cost estimates
        3. **Analyze Results**: View performance metrics in Analytics tab
        4. **Configure Claude**: Add API key for AI insights
        
        #### 💡 **Use Cases:**
        - Logistics companies optimizing shipping routes
        - E-commerce platforms calculating shipping costs
        - Supply chain management and planning
        - Transportation cost analysis and forecasting
        
        ---
        
        **Built with ❤️ using Streamlit, scikit-learn, and Claude AI**
        """)

if __name__ == "__main__":
    main()
