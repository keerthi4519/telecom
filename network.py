import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Telecom Billing Analyzer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .normal-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

class TelecomBillingAnalyzer:
    def __init__(self):
        self.data = None
        self.anomalies = None
        self.models = {}
        
    def generate_sample_data(self, n_records: int = 1000) -> pd.DataFrame:
        """Generate realistic telecom billing data"""
        np.random.seed(42)
        
        # Generate base data
        user_ids = [f"USER_{i:06d}" for i in range(1, n_records + 1)]
        
        # Plan types with different characteristics
        plan_types = ['Basic', 'Standard', 'Premium', 'Enterprise']
        plan_weights = [0.4, 0.3, 0.2, 0.1]
        plans = np.random.choice(plan_types, n_records, p=plan_weights)
        
        # Base charges by plan
        base_charges = {
            'Basic': (25, 45),
            'Standard': (45, 75),
            'Premium': (75, 120),
            'Enterprise': (120, 200)
        }
        
        # Generate charges based on plan
        charges = []
        data_usage_gb = []
        voice_minutes = []
        sms_count = []
        
        for plan in plans:
            min_charge, max_charge = base_charges[plan]
            
            # Normal charges with some variation
            if np.random.random() > 0.95:  # 5% anomalies
                # Create anomalies - unusually high charges
                charge = np.random.uniform(max_charge * 2, max_charge * 5)
                data_gb = np.random.uniform(50, 200)  # High data usage
                voice_min = np.random.uniform(2000, 5000)  # High voice usage
            elif np.random.random() > 0.98:  # Very rare - billing errors
                charge = np.random.uniform(1, 10)  # Unusually low charges
                data_gb = np.random.uniform(0.1, 2)
                voice_min = np.random.uniform(10, 100)
            else:
                # Normal usage patterns
                charge = np.random.uniform(min_charge, max_charge)
                if plan == 'Basic':
                    data_gb = np.random.exponential(2) + 0.5
                    voice_min = np.random.exponential(300) + 50
                elif plan == 'Standard':
                    data_gb = np.random.exponential(5) + 2
                    voice_min = np.random.exponential(500) + 100
                elif plan == 'Premium':
                    data_gb = np.random.exponential(10) + 5
                    voice_min = np.random.exponential(800) + 200
                else:  # Enterprise
                    data_gb = np.random.exponential(20) + 10
                    voice_min = np.random.exponential(1200) + 300
            
            charges.append(charge)
            data_usage_gb.append(min(data_gb, 500))  # Cap at 500GB
            voice_minutes.append(min(voice_min, 10000))  # Cap at 10000 minutes
            sms_count.append(np.random.poisson(50))
        
        # Generate dates (last 3 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        dates = [start_date + timedelta(days=np.random.randint(0, 90)) for _ in range(n_records)]
        
        # Create regions
        regions = ['North', 'South', 'East', 'West', 'Central']
        region_list = np.random.choice(regions, n_records)
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': user_ids,
            'billing_date': dates,
            'plan_type': plans,
            'region': region_list,
            'total_charges': np.round(charges, 2),
            'data_usage_gb': np.round(data_usage_gb, 2),
            'voice_minutes': np.round(voice_minutes, 0),
            'sms_count': sms_count,
        })
        
        # Add derived features
        df['charge_per_gb'] = df['total_charges'] / (df['data_usage_gb'] + 0.1)
        df['charge_per_minute'] = df['total_charges'] / (df['voice_minutes'] + 1)
        df['usage_ratio'] = df['data_usage_gb'] / (df['voice_minutes'] / 100 + 1)
        
        return df.sort_values('billing_date').reset_index(drop=True)
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies using multiple ML algorithms"""
        
        # Prepare features for ML
        feature_cols = ['total_charges', 'data_usage_gb', 'voice_minutes', 
                       'sms_count', 'charge_per_gb', 'charge_per_minute', 'usage_ratio']
        
        # Create plan dummy variables
        plan_dummies = pd.get_dummies(df['plan_type'], prefix='plan')
        features = pd.concat([df[feature_cols], plan_dummies], axis=1)
        
        # Handle any infinite or NaN values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        results = {}
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        iso_anomalies = iso_forest.fit_predict(features_scaled)
        iso_scores = iso_forest.score_samples(features_scaled)
        results['isolation_forest'] = {
            'predictions': iso_anomalies,
            'scores': iso_scores,
            'anomaly_count': np.sum(iso_anomalies == -1)
        }
        
        # 2. DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features_scaled)
        dbscan_anomalies = np.where(dbscan_labels == -1, -1, 1)
        results['dbscan'] = {
            'predictions': dbscan_anomalies,
            'labels': dbscan_labels,
            'anomaly_count': np.sum(dbscan_labels == -1)
        }
        
        # 3. Statistical Method (Z-score based)
        z_scores = np.abs(features_scaled).max(axis=1)
        z_threshold = 3.0
        z_anomalies = np.where(z_scores > z_threshold, -1, 1)
        results['statistical'] = {
            'predictions': z_anomalies,
            'scores': z_scores,
            'anomaly_count': np.sum(z_anomalies == -1)
        }
        
        return results, features, scaler
    
    def create_anomaly_dashboard(self, df: pd.DataFrame, anomaly_results: Dict):
        """Create interactive dashboard"""
        
        # Sidebar for method selection
        st.sidebar.subheader("🔍 Detection Method")
        method = st.sidebar.selectbox(
            "Select Anomaly Detection Method:",
            ['isolation_forest', 'dbscan', 'statistical'],
            format_func=lambda x: {
                'isolation_forest': 'Isolation Forest',
                'dbscan': 'DBSCAN Clustering',
                'statistical': 'Statistical (Z-score)'
            }[x]
        )
        
        # Get predictions for selected method
        predictions = anomaly_results[method]['predictions']
        anomaly_mask = predictions == -1
        
        # Add anomaly column to dataframe
        df_display = df.copy()
        df_display['is_anomaly'] = anomaly_mask
        df_display['anomaly_score'] = anomaly_results[method].get('scores', np.zeros(len(df)))
        
        # Main dashboard
        st.markdown('<div class="main-header">📞 Telecom Billing Analyzer Dashboard</div>', 
                   unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>📊 Total Records</h3>
                <h2>{len(df):,}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            anomaly_count = anomaly_results[method]['anomaly_count']
            st.markdown(f'''
            <div class="anomaly-card">
                <h3>🚨 Anomalies Found</h3>
                <h2>{anomaly_count}</h2>
                <p>({anomaly_count/len(df)*100:.1f}%)</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            normal_count = len(df) - anomaly_count
            st.markdown(f'''
            <div class="normal-card">
                <h3>✅ Normal Bills</h3>
                <h2>{normal_count:,}</h2>
                <p>({normal_count/len(df)*100:.1f}%)</p>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            avg_anomaly_charge = df_display[df_display['is_anomaly']]['total_charges'].mean()
            if not np.isnan(avg_anomaly_charge):
                st.markdown(f'''
                <div class="metric-card">
                    <h3>💰 Avg Anomaly Charge</h3>
                    <h2>${avg_anomaly_charge:.2f}</h2>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="metric-card">
                    <h3>💰 Avg Anomaly Charge</h3>
                    <h2>N/A</h2>
                </div>
                ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview Plots", "🔍 Anomaly Analysis", 
                                          "📋 Data Table", "📊 Statistical Summary"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot: Charges vs Data Usage
                fig1 = px.scatter(
                    df_display, 
                    x='data_usage_gb', 
                    y='total_charges',
                    color='is_anomaly',
                    color_discrete_map={True: 'red', False: 'blue'},
                    title='Total Charges vs Data Usage',
                    labels={'is_anomaly': 'Anomaly'}
                )
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Box plot by plan type
                fig2 = px.box(
                    df_display,
                    x='plan_type',
                    y='total_charges',
                    color='is_anomaly',
                    title='Charges Distribution by Plan Type'
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Time series plot
            df_time = df_display.groupby(['billing_date', 'is_anomaly']).size().reset_index(name='count')
            fig3 = px.line(
                df_time,
                x='billing_date',
                y='count',
                color='is_anomaly',
                title='Anomalies Over Time',
                color_discrete_map={True: 'red', False: 'blue'}
            )
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Anomaly score distribution
                if 'scores' in anomaly_results[method]:
                    fig4 = px.histogram(
                        x=anomaly_results[method]['scores'],
                        title=f'Anomaly Score Distribution ({method})',
                        nbins=50
                    )
                    fig4.update_layout(height=400)
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Anomaly scores not available for this method")
            
            with col2:
                # Anomalies by region
                region_anom = df_display.groupby(['region', 'is_anomaly']).size().reset_index(name='count')
                fig5 = px.bar(
                    region_anom,
                    x='region',
                    y='count',
                    color='is_anomaly',
                    title='Anomalies by Region',
                    color_discrete_map={True: 'red', False: 'blue'}
                )
                fig5.update_layout(height=400)
                st.plotly_chart(fig5, use_container_width=True)
            
            # Feature importance for anomalies (simplified)
            st.subheader("🎯 Anomaly Characteristics")
            if anomaly_count > 0:
                anomaly_data = df_display[df_display['is_anomaly']]
                normal_data = df_display[~df_display['is_anomaly']]
                
                metrics = ['total_charges', 'data_usage_gb', 'voice_minutes', 'charge_per_gb']
                comparison_data = []
                
                for metric in metrics:
                    comparison_data.append({
                        'Metric': metric,
                        'Anomaly Avg': anomaly_data[metric].mean(),
                        'Normal Avg': normal_data[metric].mean(),
                        'Difference': anomaly_data[metric].mean() - normal_data[metric].mean()
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.round(2), use_container_width=True)
        
        with tab3:
            st.subheader("🗂️ Detailed Data View")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                show_anomalies = st.checkbox("Show Anomalies Only", False)
            with col2:
                plan_filter = st.multiselect("Filter by Plan Type", 
                                           df['plan_type'].unique(),
                                           default=df['plan_type'].unique())
            
            # Apply filters
            filtered_df = df_display[df_display['plan_type'].isin(plan_filter)]
            if show_anomalies:
                filtered_df = filtered_df[filtered_df['is_anomaly']]
            
            # Display table
            display_cols = ['user_id', 'billing_date', 'plan_type', 'region',
                          'total_charges', 'data_usage_gb', 'voice_minutes', 
                          'is_anomaly', 'anomaly_score']
            
            st.dataframe(
                filtered_df[display_cols].round(2),
                use_container_width=True,
                height=400
            )
        
        with tab4:
            st.subheader("📊 Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Normal Bills Statistics:**")
                normal_stats = df_display[~df_display['is_anomaly']].describe()
                st.dataframe(normal_stats.round(2))
            
            with col2:
                if anomaly_count > 0:
                    st.write("**Anomalous Bills Statistics:**")
                    anomaly_stats = df_display[df_display['is_anomaly']].describe()
                    st.dataframe(anomaly_stats.round(2))
                else:
                    st.info("No anomalies found for statistical analysis")
        
        return df_display
    
    def export_report(self, df_display: pd.DataFrame, method: str):
        """Export anomaly report"""
        st.sidebar.subheader("📥 Export Report")
        
        if st.sidebar.button("Generate Anomaly Report"):
            anomalies_df = df_display[df_display['is_anomaly']].copy()
            
            # Create report content
            report_content = f"""
TELECOM BILLING ANOMALY REPORT
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Detection Method: {method.replace('_', ' ').title()}

SUMMARY:
- Total Records Analyzed: {len(df_display):,}
- Anomalies Detected: {len(anomalies_df):,} ({len(anomalies_df)/len(df_display)*100:.2f}%)
- Total Anomaly Amount: ${anomalies_df['total_charges'].sum():.2f}
- Average Anomaly Amount: ${anomalies_df['total_charges'].mean():.2f}

TOP ANOMALIES BY CHARGE AMOUNT:
{anomalies_df.nlargest(10, 'total_charges')[['user_id', 'total_charges', 'data_usage_gb', 'plan_type']].to_string(index=False)}

ANOMALIES BY PLAN TYPE:
{anomalies_df.groupby('plan_type').agg({'user_id': 'count', 'total_charges': ['mean', 'sum']}).round(2).to_string()}

ANOMALIES BY REGION:
{anomalies_df.groupby('region').agg({'user_id': 'count', 'total_charges': ['mean', 'sum']}).round(2).to_string()}
            """
            
            # Create download button
            st.sidebar.download_button(
                label="📄 Download Text Report",
                data=report_content,
                file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # CSV export
            csv_buffer = io.StringIO()
            anomalies_df.to_csv(csv_buffer, index=False)
            st.sidebar.download_button(
                label="📊 Download CSV Data",
                data=csv_buffer.getvalue(),
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.sidebar.success("Reports ready for download!")

def main():
    analyzer = TelecomBillingAnalyzer()
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Generate Sample Data", "Upload CSV File"]
    )
    
    if data_source == "Generate Sample Data":
        n_records = st.sidebar.slider("Number of Records", 100, 5000, 1000, 100)
        if st.sidebar.button("Generate Data"):
            with st.spinner("Generating sample data..."):
                analyzer.data = analyzer.generate_sample_data(n_records)
            st.sidebar.success(f"Generated {n_records} records!")
    
    else:  # Upload CSV
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                analyzer.data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Loaded {len(analyzer.data)} records!")
            except Exception as e:
                st.sidebar.error(f"Error loading file: {e}")
    
    # Main application
    if analyzer.data is not None:
        # Detect anomalies
        with st.spinner("Detecting anomalies..."):
            anomaly_results, features, scaler = analyzer.detect_anomalies(analyzer.data)
        
        # Create dashboard
        df_display = analyzer.create_anomaly_dashboard(analyzer.data, anomaly_results)
        
        # Export functionality
        method = st.session_state.get('method', 'isolation_forest')
        analyzer.export_report(df_display, method)
        
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>📞 Welcome to Telecom Billing Analyzer</h1>
            <p style="font-size: 1.2rem; color: #666;">
                Detect billing anomalies and prevent fraud in telecom invoices
            </p>
            <p>👈 Use the sidebar to get started by generating sample data or uploading your CSV file</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🤖 ML-Powered Detection
            - Isolation Forest
            - DBSCAN Clustering  
            - Statistical Analysis
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Interactive Dashboard
            - Real-time visualization
            - Multiple chart types
            - Filtering & analysis
            """)
        
        with col3:
            st.markdown("""
            ### 📥 Export Capabilities
            - Detailed text reports
            - CSV data export
            - Audit-ready format
            """)

if __name__ == "__main__":
    main()