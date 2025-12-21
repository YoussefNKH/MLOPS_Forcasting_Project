import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    h2 {
        color: #764ba2;
        font-weight: 600;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://backend-api:8000/api"

# Helper Functions
def check_api_health():
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_model_info():
    """Get current model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def make_prediction(data):
    """Make a single prediction"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def make_batch_prediction(data_list):
    """Make batch predictions"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict-batch",
            json={"data": data_list},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Batch prediction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error making batch prediction: {str(e)}")
        return None

# Main App
def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>📊 Sales Forecasting Dashboard</h1>", unsafe_allow_html=True)
    
    # Check API Status
    api_healthy = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Cannot connect to the API. Please ensure the FastAPI server is running at http://localhost:8000")
        st.info("Run the API with: `python -m app.main`")
        st.stop()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.markdown("### 🤖 Model Information")
        model_info = get_model_info()
        
        if model_info:
            info_data = model_info.get('model_info', {})
            st.success("✅ Model Loaded")
            st.markdown(f"""
            <div class='info-box'>
                <strong>Model Name:</strong> {info_data.get('name', 'N/A')}<br>
                <strong>Version:</strong> {info_data.get('version', 'N/A')}<br>
                <strong>Stage:</strong> {info_data.get('stage', 'N/A')}<br>
                <strong>Run ID:</strong> {info_data.get('run_id', 'N/A')[:8]}...
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model info not available")
        
        st.markdown("---")
        st.markdown("### 📋 Quick Guide")
        st.markdown("""
        1. **Single Prediction**: Fill in the form for one prediction
        2. **Batch Prediction**: Upload a CSV file for multiple predictions
        3. View results and download predictions
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ API Settings")
        st.text_input("API URL", value=API_BASE_URL, disabled=True)
    
    # Main Content - Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📦 Batch Prediction", "📈 Analytics"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.markdown("## Make a Single Prediction")
        st.markdown("Fill in the form below to get a sales forecast for a specific item.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🏷️ Product Identifiers")
            id_val = st.number_input("ID", value=1, step=1)
            item_id = st.number_input("Item ID", value=1001, step=1)
            dept_id = st.number_input("Department ID", value=1, step=1)
            cat_id = st.number_input("Category ID", value=1, step=1)
            store_id = st.number_input("Store ID", value=1, step=1)
            state_id = st.number_input("State ID", value=1, step=1)
            
        with col2:
            st.markdown("#### 📅 Time Features")
            d = st.number_input("Day (d)", value=1000, step=1)
            wm_yr_wk = st.number_input("Week (wm_yr_wk)", value=11500, step=1)
            weekday = st.number_input("Weekday", value=1, min_value=0, max_value=6, step=1)
            wday = st.number_input("Day of Week", value=2, min_value=1, max_value=7, step=1)
            month = st.number_input("Month", value=6, min_value=1, max_value=12, step=1)
            year = st.number_input("Year", value=2016, step=1)
            
            st.markdown("#### 🎉 Event Features")
            event_name_1 = st.number_input("Event Name 1", value=0, step=1)
            event_type_1 = st.number_input("Event Type 1", value=0, step=1)
            event_name_2 = st.number_input("Event Name 2", value=0, step=1)
            event_type_2 = st.number_input("Event Type 2", value=0, step=1)
            
        with col3:
            st.markdown("#### 💰 Sales Features")
            sell_price = st.number_input("Sell Price", value=3.97, step=0.01, format="%.2f")
            revenue = st.number_input("Revenue", value=11.91, step=0.01, format="%.2f")
            
            st.markdown("#### 📊 SNAP Program")
            snap_CA = st.number_input("SNAP CA", value=0, min_value=0, max_value=1, step=1)
            snap_TX = st.number_input("SNAP TX", value=0, min_value=0, max_value=1, step=1)
            snap_WI = st.number_input("SNAP WI", value=0, min_value=0, max_value=1, step=1)
        
        # Expandable section for lag features
        with st.expander("📉 Lag Features (Click to expand)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                sold_lag_1 = st.number_input("Sold Lag 1", value=3.0, step=0.1)
                sold_lag_2 = st.number_input("Sold Lag 2", value=2.0, step=0.1)
                sold_lag_3 = st.number_input("Sold Lag 3", value=1.0, step=0.1)
            with col2:
                sold_lag_6 = st.number_input("Sold Lag 6", value=4.0, step=0.1)
                sold_lag_12 = st.number_input("Sold Lag 12", value=2.5, step=0.1)
                sold_lag_24 = st.number_input("Sold Lag 24", value=3.0, step=0.1)
            with col3:
                sold_lag_36 = st.number_input("Sold Lag 36", value=2.8, step=0.1)
        
        # Expandable section for average features
        with st.expander("📊 Average Features (Click to expand)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                iteam_sold_avg = st.number_input("Item Sold Avg", value=2.5, step=0.1)
                state_sold_avg = st.number_input("State Sold Avg", value=150.0, step=1.0)
                store_sold_avg = st.number_input("Store Sold Avg", value=50.0, step=1.0)
                cat_sold_avg = st.number_input("Category Sold Avg", value=75.0, step=1.0)
            with col2:
                dept_sold_avg = st.number_input("Department Sold Avg", value=30.0, step=1.0)
                cat_dept_sold_avg = st.number_input("Cat-Dept Sold Avg", value=25.0, step=1.0)
                store_item_sold_avg = st.number_input("Store-Item Sold Avg", value=2.3, step=0.1)
                cat_item_sold_avg = st.number_input("Cat-Item Sold Avg", value=2.4, step=0.1)
            with col3:
                dept_item_sold_avg = st.number_input("Dept-Item Sold Avg", value=2.6, step=0.1)
                state_store_sold_avg = st.number_input("State-Store Sold Avg", value=45.0, step=1.0)
                state_store_cat_sold_avg = st.number_input("State-Store-Cat Sold Avg", value=22.0, step=1.0)
                store_cat_dept_sold_avg = st.number_input("Store-Cat-Dept Sold Avg", value=18.0, step=1.0)
        
        # Expandable section for rolling features
        with st.expander("📈 Rolling & Trend Features (Click to expand)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                rolling_sold_mean = st.number_input("Rolling Sold Mean", value=2.7, step=0.1)
            with col2:
                expanding_sold_mean = st.number_input("Expanding Sold Mean", value=2.5, step=0.1)
            with col3:
                selling_trend = st.number_input("Selling Trend", value=0.05, step=0.01, format="%.2f")
        
        st.markdown("---")
        
        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔮 Make Prediction", use_container_width=True)
        
        if predict_button:
            # Prepare data
            input_data = {
                "id": id_val,
                "item_id": item_id,
                "dept_id": dept_id,
                "cat_id": cat_id,
                "store_id": store_id,
                "state_id": state_id,
                "d": d,
                "wm_yr_wk": wm_yr_wk,
                "weekday": weekday,
                "wday": wday,
                "month": month,
                "year": year,
                "event_name_1": event_name_1,
                "event_type_1": event_type_1,
                "event_name_2": event_name_2,
                "event_type_2": event_type_2,
                "snap_CA": snap_CA,
                "snap_TX": snap_TX,
                "snap_WI": snap_WI,
                "sell_price": sell_price,
                "revenue": revenue,
                "sold_lag_1": sold_lag_1,
                "sold_lag_2": sold_lag_2,
                "sold_lag_3": sold_lag_3,
                "sold_lag_6": sold_lag_6,
                "sold_lag_12": sold_lag_12,
                "sold_lag_24": sold_lag_24,
                "sold_lag_36": sold_lag_36,
                "iteam_sold_avg": iteam_sold_avg,
                "state_sold_avg": state_sold_avg,
                "store_sold_avg": store_sold_avg,
                "cat_sold_avg": cat_sold_avg,
                "dept_sold_avg": dept_sold_avg,
                "cat_dept_sold_avg": cat_dept_sold_avg,
                "store_item_sold_avg": store_item_sold_avg,
                "cat_item_sold_avg": cat_item_sold_avg,
                "dept_item_sold_avg": dept_item_sold_avg,
                "state_store_sold_avg": state_store_sold_avg,
                "state_store_cat_sold_avg": state_store_cat_sold_avg,
                "store_cat_dept_sold_avg": store_cat_dept_sold_avg,
                "rolling_sold_mean": rolling_sold_mean,
                "expanding_sold_mean": expanding_sold_mean,
                "selling_trend": selling_trend
            }
            
            with st.spinner("Making prediction..."):
                result = make_prediction(input_data)
            
            if result:
                st.success("✅ Prediction Complete!")
                
                # Display prediction
                st.markdown(f"""
                <div class='prediction-box'>
                    Predicted Sales: {result['prediction']:.2f} units
                </div>
                """, unsafe_allow_html=True)
                
                # Display model info
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model Used", result['model_name'])
                with col2:
                    st.metric("Model Version", result['model_version'])
                
                # Show input summary
                with st.expander("📋 View Input Summary"):
                    st.json(input_data)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.markdown("## Batch Prediction from CSV")
        st.markdown("Upload a CSV file containing multiple records for batch predictions.")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! {len(df)} records found.")
                
                # Show preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Predict button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    batch_predict_button = st.button("🔮 Make Batch Predictions", use_container_width=True)
                
                if batch_predict_button:
                    with st.spinner(f"Making predictions for {len(df)} records..."):
                        # Convert dataframe to list of dicts
                        data_list = df.to_dict('records')
                        result = make_batch_prediction(data_list)
                    
                    if result:
                        st.success("✅ Batch Prediction Complete!")
                        
                        # Add predictions to dataframe
                        df['predicted_sales'] = result['predictions']
                        
                        # Display results
                        st.markdown("### 📊 Prediction Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Records", len(df))
                        with col2:
                            st.metric("Avg Prediction", f"{df['predicted_sales'].mean():.2f}")
                        with col3:
                            st.metric("Min Prediction", f"{df['predicted_sales'].min():.2f}")
                        with col4:
                            st.metric("Max Prediction", f"{df['predicted_sales'].max():.2f}")
                        
                        # Visualization
                        st.markdown("### 📈 Prediction Distribution")
                        fig = px.histogram(df, x='predicted_sales', nbins=30,
                                         title='Distribution of Predicted Sales',
                                         labels={'predicted_sales': 'Predicted Sales'})
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Model info
                        st.info(f"Model: {result['model_name']} (v{result['model_version']})")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        else:
            st.info("👆 Upload a CSV file to get started")
            
            # Show expected format
            with st.expander("📄 Expected CSV Format"):
                st.markdown("""
                Your CSV should contain the following columns:
                - id, item_id, dept_id, cat_id, store_id, state_id
                - d, wm_yr_wk, weekday, wday, month, year
                - event_name_1, event_type_1, event_name_2, event_type_2
                - snap_CA, snap_TX, snap_WI
                - sell_price, revenue
                - sold_lag_1, sold_lag_2, sold_lag_3, sold_lag_6, sold_lag_12, sold_lag_24, sold_lag_36
                - iteam_sold_avg, state_sold_avg, store_sold_avg, cat_sold_avg, dept_sold_avg
                - cat_dept_sold_avg, store_item_sold_avg, cat_item_sold_avg, dept_item_sold_avg
                - state_store_sold_avg, state_store_cat_sold_avg, store_cat_dept_sold_avg
                - rolling_sold_mean, expanding_sold_mean, selling_trend
                """)
    
    # Tab 3: Analytics
    with tab3:
        st.markdown("## 📈 Analytics Dashboard")
        st.info("This section will show historical predictions and trends. Upload predictions or make some predictions first!")
        
        # Placeholder for analytics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Quick Stats")
            st.markdown("""
            <div class='metric-card'>
                <h2>Coming Soon</h2>
                <p>Historical prediction analytics and trends will be displayed here</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Performance Metrics")
            st.markdown("""
            <div class='metric-card'>
                <h2>Coming Soon</h2>
                <p>Model performance metrics and comparison charts</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()