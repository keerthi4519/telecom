import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

# Time series (using simple implementations due to library constraints)
from scipy import stats
import math

# Page configuration
st.set_page_config(
    page_title="Smart Bandwidth Allocation",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-high {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-medium {
        background-color: #ffaa00;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-low {
        background-color: #00aa44;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .tower-status {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class BandwidthDatabase:
    """Simple database operations for tower data"""
    
    def __init__(self):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.create_tables()
        self.populate_sample_data()
    
    def create_tables(self):
        """Create necessary tables"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tower_data (
                id INTEGER PRIMARY KEY,
                tower_id TEXT,
                timestamp TEXT,
                bandwidth_used REAL,
                total_bandwidth REAL,
                active_users INTEGER,
                peak_hour INTEGER,
                day_of_week INTEGER,
                hour INTEGER
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tower_info (
                tower_id TEXT PRIMARY KEY,
                location TEXT,
                max_bandwidth REAL,
                installed_date TEXT
            )
        ''')
    
    def populate_sample_data(self):
        """Generate realistic sample data"""
        np.random.seed(42)
        tower_ids = ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T007', 'T008']
        locations = ['Downtown', 'Suburbs', 'Industrial', 'Residential', 'Commercial', 'Airport', 'University', 'Mall']
        
        # Tower info
        for i, (tower_id, location) in enumerate(zip(tower_ids, locations)):
            max_bw = np.random.uniform(800, 1500)
            self.conn.execute('''
                INSERT INTO tower_info VALUES (?, ?, ?, ?)
            ''', (tower_id, location, max_bw, '2020-01-01'))
        
        # Historical data (last 30 days)
        data_points = []
        start_date = datetime.now() - timedelta(days=30)
        
        for tower_id in tower_ids:
            # Get max bandwidth for this tower
            max_bw = self.conn.execute('SELECT max_bandwidth FROM tower_info WHERE tower_id = ?', (tower_id,)).fetchone()[0]
            
            for day in range(30):
                current_date = start_date + timedelta(days=day)
                day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
                
                for hour in range(24):
                    # Create realistic usage patterns
                    base_usage = 0.3  # 30% base usage
                    
                    # Peak hours (8-10 AM, 6-10 PM)
                    if hour in [8, 9, 18, 19, 20, 21]:
                        peak_multiplier = np.random.uniform(2.0, 3.5)
                        is_peak = 1
                    elif hour in [10, 11, 12, 13, 14, 15, 16, 17, 22]:
                        peak_multiplier = np.random.uniform(1.2, 2.0)
                        is_peak = 0
                    else:
                        peak_multiplier = np.random.uniform(0.5, 1.0)
                        is_peak = 0
                    
                    # Weekend patterns
                    if day_of_week in [5, 6]:  # Weekend
                        peak_multiplier *= np.random.uniform(0.7, 1.3)
                    
                    # Calculate usage
                    usage_ratio = min(base_usage * peak_multiplier + np.random.normal(0, 0.1), 0.95)
                    bandwidth_used = max_bw * max(0.1, usage_ratio)
                    active_users = int(bandwidth_used / np.random.uniform(0.8, 2.5))
                    
                    timestamp = current_date.replace(hour=hour).strftime('%Y-%m-%d %H:%M:%S')
                    
                    data_points.append((
                        tower_id, timestamp, bandwidth_used, max_bw, 
                        active_users, is_peak, day_of_week, hour
                    ))
        
        self.conn.executemany('''
            INSERT INTO tower_data (tower_id, timestamp, bandwidth_used, total_bandwidth, 
                                  active_users, peak_hour, day_of_week, hour)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_points)
        
        self.conn.commit()
    
    def get_tower_data(self, tower_id=None, days_back=7):
        """Get tower data"""
        query = '''
            SELECT * FROM tower_data 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days_back)
        
        if tower_id:
            query += f" AND tower_id = '{tower_id}'"
        
        query += " ORDER BY timestamp DESC"
        
        return pd.read_sql_query(query, self.conn)
    
    def get_tower_info(self):
        """Get tower information"""
        return pd.read_sql_query("SELECT * FROM tower_info", self.conn)

class SimpleForecastModel:
    """Simple forecasting model combining trend and seasonality"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_features(self, df):
        """Create features for forecasting"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        df = df.sort_values('timestamp')
        df['usage_lag_1'] = df['bandwidth_used'].shift(1)
        df['usage_lag_24'] = df['bandwidth_used'].shift(24)  # Same hour previous day
        df['usage_lag_168'] = df['bandwidth_used'].shift(168)  # Same hour previous week
        
        # Rolling statistics
        df['usage_mean_24h'] = df['bandwidth_used'].rolling(24).mean()
        df['usage_std_24h'] = df['bandwidth_used'].rolling(24).std()
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def fit(self, df):
        """Fit the forecasting model"""
        df_features = self.create_features(df)
        
        feature_cols = [
            'hour', 'day_of_week', 'day_of_month', 'is_weekend',
            'active_users', 'peak_hour', 'usage_lag_1', 'usage_lag_24', 'usage_lag_168',
            'usage_mean_24h', 'usage_std_24h', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
        ]
        
        # Remove rows with NaN values (due to lag features)
        df_clean = df_features.dropna(subset=feature_cols + ['bandwidth_used'])
        
        if len(df_clean) == 0:
            st.warning("Not enough data for training. Using simple average.")
            return
        
        X = df_clean[feature_cols]
        y = df_clean['bandwidth_used']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.feature_cols = feature_cols
        self.is_fitted = True
    
    def predict(self, df, periods=24):
        """Predict future bandwidth usage"""
        if not self.is_fitted:
            # Simple fallback: use recent average with hourly patterns
            recent_data = df.tail(168)  # Last week
            hourly_avg = recent_data.groupby('hour')['bandwidth_used'].mean()
            
            predictions = []
            last_time = pd.to_datetime(df['timestamp'].max())
            
            for i in range(periods):
                next_time = last_time + timedelta(hours=i+1)
                hour = next_time.hour
                pred = hourly_avg.get(hour, df['bandwidth_used'].mean())
                predictions.append(pred)
            
            return np.array(predictions)
        
        # Use trained model
        df_features = self.create_features(df)
        last_row = df_features.iloc[-1].copy()
        
        predictions = []
        
        for i in range(periods):
            # Update time features
            next_time = pd.to_datetime(last_row['timestamp']) + timedelta(hours=i+1)
            last_row['hour'] = next_time.hour
            last_row['day_of_week'] = next_time.dayofweek
            last_row['day_of_month'] = next_time.day
            last_row['is_weekend'] = 1 if next_time.dayofweek in [5, 6] else 0
            
            # Update cyclical features
            last_row['hour_sin'] = np.sin(2 * np.pi * last_row['hour'] / 24)
            last_row['hour_cos'] = np.cos(2 * np.pi * last_row['hour'] / 24)
            last_row['dow_sin'] = np.sin(2 * np.pi * last_row['day_of_week'] / 7)
            last_row['dow_cos'] = np.cos(2 * np.pi * last_row['day_of_week'] / 7)
            
            # Predict
            X_pred = last_row[self.feature_cols].values.reshape(1, -1)
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.model.predict(X_pred_scaled)[0]
            
            predictions.append(pred)
            
            # Update lag features for next prediction
            last_row['usage_lag_1'] = pred
        
        return np.array(predictions)

# Initialize database and model
@st.cache_resource
def init_app():
    db = BandwidthDatabase()
    model = SimpleForecastModel()
    return db, model

def calculate_alerts(df, tower_info):
    """Calculate alerts for towers nearing overload"""
    alerts = []
    
    for _, tower in tower_info.iterrows():
        tower_data = df[df['tower_id'] == tower['tower_id']]
        
        if len(tower_data) == 0:
            continue
            
        latest_usage = tower_data.iloc[0]['bandwidth_used']
        max_bandwidth = tower['max_bandwidth']
        utilization = latest_usage / max_bandwidth
        
        if utilization >= 0.9:
            alert_level = "HIGH"
            alert_class = "alert-high"
            message = f"🚨 Tower {tower['tower_id']} ({tower['location']}) at {utilization:.1%} capacity!"
        elif utilization >= 0.75:
            alert_level = "MEDIUM"
            alert_class = "alert-medium"
            message = f"⚠️ Tower {tower['tower_id']} ({tower['location']}) at {utilization:.1%} capacity"
        elif utilization >= 0.6:
            alert_level = "LOW"
            alert_class = "alert-low"
            message = f"ℹ️ Tower {tower['tower_id']} ({tower['location']}) at {utilization:.1%} capacity"
        else:
            continue
            
        alerts.append({
            'tower_id': tower['tower_id'],
            'location': tower['location'],
            'utilization': utilization,
            'alert_level': alert_level,
            'alert_class': alert_class,
            'message': message
        })
    
    return sorted(alerts, key=lambda x: x['utilization'], reverse=True)

def main():
    st.title("📡 Smart Bandwidth Allocation System")
    st.markdown("Optimize bandwidth distribution across telecom towers with AI-powered forecasting")
    
    # Initialize
    db, model = init_app()
    
    # Sidebar
    st.sidebar.header("🔧 Controls")
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_resource.clear()
        st.rerun()
    
    # Tower selection
    tower_info = db.get_tower_info()
    selected_towers = st.sidebar.multiselect(
        "Select Towers",
        options=tower_info['tower_id'].tolist(),
        default=tower_info['tower_id'].tolist()[:4]
    )
    
    # Time range
    days_back = st.sidebar.slider("Days of Historical Data", 1, 30, 7)
    forecast_hours = st.sidebar.slider("Forecast Hours", 6, 72, 24)
    
    # Get data
    df = db.get_tower_data(days_back=days_back)
    
    if df.empty:
        st.error("No data available")
        return
    
    # Filter by selected towers
    if selected_towers:
        df = df[df['tower_id'].isin(selected_towers)]
        tower_info = tower_info[tower_info['tower_id'].isin(selected_towers)]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Towers</h3>
            <h2>{}</h2>
        </div>
        """.format(len(selected_towers)), unsafe_allow_html=True)
    
    with col2:
        avg_utilization = (df.groupby('tower_id')['bandwidth_used'].last() / 
                          df.groupby('tower_id')['total_bandwidth'].last()).mean()
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Utilization</h3>
            <h2>{:.1%}</h2>
        </div>
        """.format(avg_utilization), unsafe_allow_html=True)
    
    with col3:
        total_users = df.groupby('tower_id')['active_users'].last().sum()
        st.markdown("""
        <div class="metric-card">
            <h3>Active Users</h3>
            <h2>{:,}</h2>
        </div>
        """.format(total_users), unsafe_allow_html=True)
    
    with col4:
        peak_towers = (df.groupby('tower_id')['peak_hour'].last() == 1).sum()
        st.markdown("""
        <div class="metric-card">
            <h3>Peak Hour Towers</h3>
            <h2>{}</h2>
        </div>
        """.format(peak_towers), unsafe_allow_html=True)
    
    # Alerts Section
    st.header("🚨 Tower Alerts")
    alerts = calculate_alerts(df, tower_info)
    
    if alerts:
        alert_col1, alert_col2 = st.columns(2)
        
        with alert_col1:
            st.subheader("Critical Alerts")
            high_alerts = [a for a in alerts if a['alert_level'] == 'HIGH']
            if high_alerts:
                for alert in high_alerts:
                    st.markdown(f'<div class="{alert["alert_class"]}">{alert["message"]}</div>', 
                               unsafe_allow_html=True)
            else:
                st.success("No critical alerts")
        
        with alert_col2:
            st.subheader("Warning Alerts")
            medium_alerts = [a for a in alerts if a['alert_level'] == 'MEDIUM']
            if medium_alerts:
                for alert in medium_alerts:
                    st.markdown(f'<div class="{alert["alert_class"]}">{alert["message"]}</div>', 
                               unsafe_allow_html=True)
            else:
                st.info("No warning alerts")
    else:
        st.success("All towers operating within normal parameters")
    
    # Forecasting Section
    st.header("🔮 Bandwidth Forecasting")
    
    # Train models and generate forecasts
    forecasts = {}
    
    for tower_id in selected_towers:
        tower_data = df[df['tower_id'] == tower_id].sort_values('timestamp')
        
        if len(tower_data) >= 24:  # Need at least 24 hours of data
            # Train model for this tower
            tower_model = SimpleForecastModel()
            tower_model.fit(tower_data)
            
            # Generate forecast
            forecast = tower_model.predict(tower_data, forecast_hours)
            
            # Create forecast timestamps
            last_time = pd.to_datetime(tower_data['timestamp'].iloc[-1])
            forecast_times = [last_time + timedelta(hours=i+1) for i in range(forecast_hours)]
            
            forecasts[tower_id] = {
                'times': forecast_times,
                'values': forecast,
                'max_bandwidth': tower_data['total_bandwidth'].iloc[-1]
            }
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📊 Real-time Dashboard", "📈 Forecasting", "📋 Tower Status"])
    
    with tab1:
        st.subheader("Current Bandwidth Usage by Tower")
        
        # Real-time usage chart
        latest_data = df.groupby(['tower_id', 'timestamp']).first().reset_index()
        latest_data = latest_data.loc[latest_data.groupby('tower_id')['timestamp'].idxmax()]
        
        fig = px.bar(
            latest_data, 
            x='tower_id', 
            y='bandwidth_used',
            color='bandwidth_used',
            color_continuous_scale='Viridis',
            title="Current Bandwidth Usage by Tower",
            labels={'bandwidth_used': 'Bandwidth Used (Mbps)', 'tower_id': 'Tower ID'}
        )
        
        # Add capacity line
        for _, tower in tower_info.iterrows():
            fig.add_hline(
                y=tower['max_bandwidth'], 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Max: {tower['max_bandwidth']:.0f} Mbps"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Usage patterns over time
        st.subheader("24-Hour Usage Patterns")
        
        fig2 = go.Figure()
        
        for tower_id in selected_towers:
            tower_data = df[df['tower_id'] == tower_id].tail(24)
            if not tower_data.empty:
                fig2.add_trace(go.Scatter(
                    x=pd.to_datetime(tower_data['timestamp']),
                    y=tower_data['bandwidth_used'],
                    mode='lines+markers',
                    name=f'Tower {tower_id}',
                    line=dict(width=2)
                ))
        
        fig2.update_layout(
            title="Bandwidth Usage - Last 24 Hours",
            xaxis_title="Time",
            yaxis_title="Bandwidth Used (Mbps)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Bandwidth Demand Forecasting")
        
        if forecasts:
            # Forecast visualization
            fig3 = make_subplots(
                rows=len(forecasts), 
                cols=1,
                subplot_titles=[f"Tower {tid}" for tid in forecasts.keys()],
                vertical_spacing=0.1
            )
            
            for i, (tower_id, forecast_data) in enumerate(forecasts.items(), 1):
                # Historical data
                tower_data = df[df['tower_id'] == tower_id].tail(48)
                
                fig3.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(tower_data['timestamp']),
                        y=tower_data['bandwidth_used'],
                        mode='lines',
                        name=f'{tower_id} - Actual',
                        line=dict(color='blue', width=2)
                    ),
                    row=i, col=1
                )
                
                # Forecast
                fig3.add_trace(
                    go.Scatter(
                        x=forecast_data['times'],
                        y=forecast_data['values'],
                        mode='lines',
                        name=f'{tower_id} - Forecast',
                        line=dict(color='red', dash='dash', width=2)
                    ),
                    row=i, col=1
                )
                
                # Capacity line
                fig3.add_hline(
                    y=forecast_data['max_bandwidth'],
                    line_dash="dot",
                    line_color="orange",
                    row=i, col=1
                )
            
            fig3.update_layout(
                height=300 * len(forecasts),
                title_text="Bandwidth Forecast vs Historical Data",
                showlegend=True
            )
            
            st.plotly_chart(fig3, use_container_width=True)
            
            # Forecast summary
            st.subheader("Forecast Summary")
            
            summary_data = []
            for tower_id, forecast_data in forecasts.items():
                max_forecast = np.max(forecast_data['values'])
                avg_forecast = np.mean(forecast_data['values'])
                utilization = max_forecast / forecast_data['max_bandwidth']
                
                summary_data.append({
                    'Tower ID': tower_id,
                    'Max Predicted Usage': f"{max_forecast:.1f} Mbps",
                    'Avg Predicted Usage': f"{avg_forecast:.1f} Mbps",
                    'Peak Utilization': f"{utilization:.1%}",
                    'Status': '🔴 Overload Risk' if utilization > 0.9 else '🟡 Monitor' if utilization > 0.7 else '🟢 Normal'
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        else:
            st.warning("Insufficient data for forecasting. Need at least 24 hours of historical data.")
    
    with tab3:
        st.subheader("Detailed Tower Status")
        
        # Create detailed tower status
        status_data = []
        
        for _, tower in tower_info.iterrows():
            tower_data = df[df['tower_id'] == tower['tower_id']]
            
            if not tower_data.empty:
                latest = tower_data.iloc[0]
                utilization = latest['bandwidth_used'] / latest['total_bandwidth']
                
                # Calculate trends
                if len(tower_data) >= 24:
                    recent_24h = tower_data.head(24)['bandwidth_used'].mean()
                    previous_24h = tower_data.iloc[24:48]['bandwidth_used'].mean() if len(tower_data) >= 48 else recent_24h
                    trend = ((recent_24h - previous_24h) / previous_24h * 100) if previous_24h > 0 else 0
                else:
                    trend = 0
                
                status_data.append({
                    'Tower ID': tower['tower_id'],
                    'Location': tower['location'],
                    'Current Usage': f"{latest['bandwidth_used']:.1f} Mbps",
                    'Max Capacity': f"{latest['total_bandwidth']:.1f} Mbps",
                    'Utilization': f"{utilization:.1%}",
                    'Active Users': latest['active_users'],
                    '24h Trend': f"{trend:+.1f}%",
                    'Status': '🔴 Critical' if utilization >= 0.9 else '🟡 Warning' if utilization >= 0.7 else '🟢 Good'
                })
        
        if status_data:
            st.dataframe(pd.DataFrame(status_data), use_container_width=True)
            
            # Download data option
            csv = pd.DataFrame(status_data).to_csv(index=False)
            st.download_button(
                label="📥 Download Tower Status Report",
                data=csv,
                file_name=f"tower_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Smart Bandwidth Allocation System | Real-time monitoring and AI-powered forecasting</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
