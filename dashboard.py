import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Smart Home Energy Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #00D9FF;
        --secondary-color: #7C3AED;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --dark-bg: #0F172A;
        --card-bg: #1E293B;
    }
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(124, 58, 237, 0.2);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(124, 58, 237, 0.4);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #00D9FF 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #E2E8F0;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #CBD5E1;
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #00D9FF 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.6);
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput {
        border-radius: 8px;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 16px;
        border: 1px solid rgba(124, 58, 237, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #7C3AED;
    }
    </style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('dataset.csv')
    
    # Convert date & time
    df['Date'] = pd.to_datetime(df['Date'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
    
    # Extract useful features
    df['hour'] = df['Time'].dt.hour
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df['day_of_month'] = df['Date'].dt.day
    
    # Add time period
    df['time_period'] = pd.cut(df['hour'], 
                               bins=[0, 6, 12, 18, 24], 
                               labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                               include_lowest=True)
    
    return df

# Cache model training
@st.cache_resource
def train_model(df):
    """Train the energy consumption prediction model"""
    y = df['Energy Consumption (kWh)']
    X = df.drop(['Energy Consumption (kWh)', 'Date', 'Time', 'time_period', 'day_of_month'], axis=1)
    
    categorical_features = ['Appliance Type', 'Season']
    numerical_features = ['Outdoor Temperature (°C)', 'Household Size', 'hour', 'month', 'weekday']
    
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])
    
    model = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.05,
        max_iter=300,
        random_state=42
    )
    
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # Calculate metrics
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    return pipeline, {'MAE': mae, 'RMSE': rmse, 'R2': r2}, X_test, y_test, predictions

# Load data and train model
df = load_data()
pipeline, metrics, X_test, y_test, predictions = train_model(df)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/energy.png", width=80)
    st.title("⚡ Navigation")
    
    page = st.radio(
        "Select Page",
        ["🏠 Overview", "📊 Analytics", "🔮 Predictions", "📈 Model Performance", "🔍 Data Explorer"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Quick Stats")
    st.metric("Total Records", f"{len(df):,}")
    st.metric("Unique Homes", df['Home ID'].nunique())
    st.metric("Appliance Types", df['Appliance Type'].nunique())
    
    st.markdown("---")
    st.markdown("### 🎯 Filters")
    
    # Date range filter
    date_range = st.date_input(
        "Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )
    
    # Season filter
    seasons = st.multiselect(
        "Season",
        options=df['Season'].unique(),
        default=df['Season'].unique()
    )
    
    # Apply filters
    if len(date_range) == 2:
        mask = (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))
        df_filtered = df[mask]
    else:
        df_filtered = df.copy()
    
    if seasons:
        df_filtered = df_filtered[df_filtered['Season'].isin(seasons)]

# Main content
if page == "🏠 Overview":
    st.title("⚡ Smart Home Energy Consumption Dashboard")
    st.markdown("### Real-time insights into household energy usage patterns")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consumption = df_filtered['Energy Consumption (kWh)'].sum()
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{total_consumption:,.0f}</p>
                <p class="metric-label">Total Energy (kWh)</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_consumption = df_filtered['Energy Consumption (kWh)'].mean()
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{avg_consumption:.2f}</p>
                <p class="metric-label">Avg Consumption</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        peak_consumption = df_filtered['Energy Consumption (kWh)'].max()
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{peak_consumption:.2f}</p>
                <p class="metric-label">Peak Usage</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        unique_homes = df_filtered['Home ID'].nunique()
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{unique_homes}</p>
                <p class="metric-label">Active Homes</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔌 Energy by Appliance")
        appliance_data = df_filtered.groupby('Appliance Type')['Energy Consumption (kWh)'].sum().sort_values(ascending=True)
        fig = px.bar(
            x=appliance_data.values,
            y=appliance_data.index,
            orientation='h',
            color=appliance_data.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Total Energy (kWh)', 'y': 'Appliance Type'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🌡️ Energy by Season")
        season_data = df_filtered.groupby('Season')['Energy Consumption (kWh)'].sum()
        fig = px.pie(
            values=season_data.values,
            names=season_data.index,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Time series
    st.markdown("### 📅 Daily Energy Consumption Trend")
    daily_data = df_filtered.groupby(df_filtered['Date'].dt.date)['Energy Consumption (kWh)'].sum().reset_index()
    daily_data.columns = ['Date', 'Energy Consumption (kWh)']
    
    fig = px.area(
        daily_data,
        x='Date',
        y='Energy Consumption (kWh)',
        color_discrete_sequence=['#7C3AED']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E2E8F0'),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Analytics":
    st.title("📊 Advanced Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Time Analysis", "🏠 Household Analysis", "🌡️ Temperature Impact", "📊 Comparative Analysis"])
    
    with tab1:
        st.markdown("### Energy Consumption by Time of Day")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly consumption
            hourly_data = df_filtered.groupby('hour')['Energy Consumption (kWh)'].mean().reset_index()
            fig = px.line(
                hourly_data,
                x='hour',
                y='Energy Consumption (kWh)',
                markers=True,
                color_discrete_sequence=['#00D9FF']
            )
            fig.update_layout(
                title="Average Consumption by Hour",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Time period distribution
            period_data = df_filtered.groupby('time_period')['Energy Consumption (kWh)'].sum()
            fig = px.bar(
                x=period_data.index,
                y=period_data.values,
                color=period_data.values,
                color_continuous_scale='Turbo',
                labels={'x': 'Time Period', 'y': 'Total Energy (kWh)'}
            )
            fig.update_layout(
                title="Energy by Time Period",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap
        st.markdown("### 🔥 Hourly Consumption Heatmap by Appliance")
        heatmap_data = df_filtered.pivot_table(
            values='Energy Consumption (kWh)',
            index='Appliance Type',
            columns='hour',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            heatmap_data,
            color_continuous_scale='Viridis',
            aspect='auto',
            labels=dict(x="Hour of Day", y="Appliance Type", color="Avg kWh")
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Household Size Impact")
        
        col1, col2 = st.columns(2)
        
        with col1:
            household_data = df_filtered.groupby('Household Size')['Energy Consumption (kWh)'].agg(['mean', 'sum']).reset_index()
            fig = px.bar(
                household_data,
                x='Household Size',
                y='sum',
                color='mean',
                color_continuous_scale='Plasma',
                labels={'sum': 'Total Energy (kWh)', 'mean': 'Avg Energy'}
            )
            fig.update_layout(
                title="Total Energy by Household Size",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(
                df_filtered,
                x='Household Size',
                y='Energy Consumption (kWh)',
                color='Household Size',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(
                title="Energy Distribution by Household Size",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Temperature Impact on Energy Consumption")
        
        # Scatter plot
        fig = px.scatter(
            df_filtered,
            x='Outdoor Temperature (°C)',
            y='Energy Consumption (kWh)',
            color='Season',
            size='Household Size',
            hover_data=['Appliance Type'],
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature bins
        col1, col2 = st.columns(2)
        
        with col1:
            df_filtered['temp_range'] = pd.cut(
                df_filtered['Outdoor Temperature (°C)'],
                bins=[-10, 0, 10, 20, 30, 40],
                labels=['Very Cold', 'Cold', 'Moderate', 'Warm', 'Hot']
            )
            temp_data = df_filtered.groupby('temp_range')['Energy Consumption (kWh)'].mean().reset_index()
            
            fig = px.bar(
                temp_data,
                x='temp_range',
                y='Energy Consumption (kWh)',
                color='Energy Consumption (kWh)',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(
                title="Average Energy by Temperature Range",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Correlation by appliance
            appliances = df_filtered['Appliance Type'].unique()[:5]
            correlations = []
            for appliance in appliances:
                app_data = df_filtered[df_filtered['Appliance Type'] == appliance]
                corr = app_data['Outdoor Temperature (°C)'].corr(app_data['Energy Consumption (kWh)'])
                correlations.append({'Appliance': appliance, 'Correlation': corr})
            
            corr_df = pd.DataFrame(correlations)
            fig = px.bar(
                corr_df,
                x='Appliance',
                y='Correlation',
                color='Correlation',
                color_continuous_scale='RdBu',
                range_color=[-1, 1]
            )
            fig.update_layout(
                title="Temperature-Energy Correlation by Appliance",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### Comparative Analysis")
        
        # Multi-select for comparison
        col1, col2 = st.columns(2)
        
        with col1:
            selected_appliances = st.multiselect(
                "Select Appliances to Compare",
                options=df_filtered['Appliance Type'].unique(),
                default=df_filtered['Appliance Type'].unique()[:3]
            )
        
        with col2:
            selected_seasons = st.multiselect(
                "Select Seasons to Compare",
                options=df_filtered['Season'].unique(),
                default=df_filtered['Season'].unique()
            )
        
        if selected_appliances and selected_seasons:
            comparison_data = df_filtered[
                (df_filtered['Appliance Type'].isin(selected_appliances)) &
                (df_filtered['Season'].isin(selected_seasons))
            ]
            
            # Grouped bar chart
            grouped_data = comparison_data.groupby(['Appliance Type', 'Season'])['Energy Consumption (kWh)'].mean().reset_index()
            
            fig = px.bar(
                grouped_data,
                x='Appliance Type',
                y='Energy Consumption (kWh)',
                color='Season',
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig.update_layout(
                title="Average Energy Consumption: Appliance vs Season",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔮 Predictions":
    st.title("🔮 Energy Consumption Predictor")
    st.markdown("### Predict energy consumption based on various parameters")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📝 Input Parameters")
        
        input_col1, input_col2 = st.columns(2)
        
        with input_col1:
            appliance = st.selectbox(
                "Appliance Type",
                options=sorted(df['Appliance Type'].unique())
            )
            
            season = st.selectbox(
                "Season",
                options=['Spring', 'Summer', 'Fall', 'Winter']
            )
            
            temperature = st.slider(
                "Outdoor Temperature (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=20.0,
                step=0.5
            )
            
            household_size = st.slider(
                "Household Size",
                min_value=1,
                max_value=5,
                value=3
            )
        
        with input_col2:
            hour = st.slider(
                "Hour of Day",
                min_value=0,
                max_value=23,
                value=12
            )
            
            month = st.slider(
                "Month",
                min_value=1,
                max_value=12,
                value=6
            )
            
            # Weekday selection
            weekday_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            weekday = st.selectbox(
                "Day of Week",
                options=list(weekday_map.keys())
            )
            weekday_num = weekday_map[weekday]
        
        if st.button("🔮 Predict Energy Consumption", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame([{
                'Home ID': 1,
                'Appliance Type': appliance,
                'Season': season,
                'Outdoor Temperature (°C)': temperature,
                'Household Size': household_size,
                'hour': hour,
                'month': month,
                'weekday': weekday_num
            }])
            
            # Make prediction
            prediction = pipeline.predict(input_data)[0]
            
            # Display result
            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{prediction:.2f}</p>
                        <p class="metric-label">Predicted kWh</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                # Get average for this appliance
                avg_appliance = df[df['Appliance Type'] == appliance]['Energy Consumption (kWh)'].mean()
                diff_pct = ((prediction - avg_appliance) / avg_appliance) * 100
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">{diff_pct:+.1f}%</p>
                        <p class="metric-label">vs Avg {appliance}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                # Estimated cost (assuming $0.12 per kWh)
                cost = prediction * 0.12
                st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-value">${cost:.2f}</p>
                        <p class="metric-label">Estimated Cost</p>
                    </div>
                """, unsafe_allow_html=True)
            
            # Show insights
            st.markdown("### 💡 Insights")
            
            if prediction > avg_appliance * 1.2:
                st.warning(f"⚠️ This prediction is **{diff_pct:.1f}% higher** than average for {appliance}. Consider energy-saving measures.")
            elif prediction < avg_appliance * 0.8:
                st.success(f"✅ This prediction is **{abs(diff_pct):.1f}% lower** than average for {appliance}. Great energy efficiency!")
            else:
                st.info(f"ℹ️ This prediction is within normal range for {appliance}.")
    
    with col2:
        st.markdown("#### 📊 Quick Stats")
        
        # Show appliance stats
        appliance_stats = df[df['Appliance Type'] == appliance]['Energy Consumption (kWh)']
        
        st.metric("Average", f"{appliance_stats.mean():.2f} kWh")
        st.metric("Min", f"{appliance_stats.min():.2f} kWh")
        st.metric("Max", f"{appliance_stats.max():.2f} kWh")
        st.metric("Std Dev", f"{appliance_stats.std():.2f} kWh")
        
        st.markdown("---")
        
        # Distribution plot
        fig = px.histogram(
            df[df['Appliance Type'] == appliance],
            x='Energy Consumption (kWh)',
            nbins=30,
            color_discrete_sequence=['#7C3AED']
        )
        fig.update_layout(
            title=f"{appliance} Distribution",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0', size=10),
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Model Performance":
    st.title("📈 Model Performance Metrics")
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{metrics['MAE']:.4f}</p>
                <p class="metric-label">Mean Absolute Error</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{metrics['RMSE']:.4f}</p>
                <p class="metric-label">Root Mean Squared Error</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{metrics['R2']:.4f}</p>
                <p class="metric-label">R² Score</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction vs Actual
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Predictions vs Actual Values")
        
        comparison_df = pd.DataFrame({
            'Actual': y_test[:1000],
            'Predicted': predictions[:1000]
        })
        
        fig = px.scatter(
            comparison_df,
            x='Actual',
            y='Predicted',
            color_discrete_sequence=['#00D9FF'],
            opacity=0.6
        )
        
        # Add perfect prediction line
        max_val = max(comparison_df['Actual'].max(), comparison_df['Predicted'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='#EF4444', dash='dash')
            )
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Residual Distribution")
        
        residuals = y_test - predictions
        
        fig = px.histogram(
            x=residuals,
            nbins=50,
            color_discrete_sequence=['#7C3AED']
        )
        fig.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residual (Actual - Predicted)",
            yaxis_title="Frequency",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E2E8F0'),
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (approximate)
    st.markdown("### 🔍 Model Insights")
    
    st.info("""
    **Model Details:**
    - **Algorithm:** Histogram-based Gradient Boosting Regressor
    - **Max Depth:** 8
    - **Learning Rate:** 0.05
    - **Iterations:** 300
    - **Features:** Appliance Type, Season, Temperature, Household Size, Hour, Month, Weekday
    
    The model achieves strong performance with an R² score of {:.2f}, indicating it explains {:.1f}% of the variance in energy consumption.
    """.format(metrics['R2'], metrics['R2'] * 100))

else:  # Data Explorer
    st.title("🔍 Data Explorer")
    
    # Search and filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_appliance = st.multiselect(
            "Filter by Appliance",
            options=df['Appliance Type'].unique(),
            default=None
        )
    
    with col2:
        search_season = st.multiselect(
            "Filter by Season",
            options=df['Season'].unique(),
            default=None
        )
    
    with col3:
        search_household = st.multiselect(
            "Filter by Household Size",
            options=sorted(df['Household Size'].unique()),
            default=None
        )
    
    # Apply filters
    filtered_df = df_filtered.copy()
    
    if search_appliance:
        filtered_df = filtered_df[filtered_df['Appliance Type'].isin(search_appliance)]
    if search_season:
        filtered_df = filtered_df[filtered_df['Season'].isin(search_season)]
    if search_household:
        filtered_df = filtered_df[filtered_df['Household Size'].isin(search_household)]
    
    # Display stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Energy", f"{filtered_df['Energy Consumption (kWh)'].mean():.2f} kWh")
    with col3:
        st.metric("Total Energy", f"{filtered_df['Energy Consumption (kWh)'].sum():,.0f} kWh")
    with col4:
        st.metric("Unique Homes", filtered_df['Home ID'].nunique())
    
    st.markdown("---")
    
    # Display data
    st.markdown("### 📋 Data Table")
    
    # Format the dataframe
    display_df = filtered_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
    display_df['Time'] = display_df['Time'].dt.strftime('%H:%M')
    
    st.dataframe(
        display_df[['Home ID', 'Appliance Type', 'Energy Consumption (kWh)', 
                   'Time', 'Date', 'Outdoor Temperature (°C)', 'Season', 'Household Size']],
        use_container_width=True,
        height=400
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="energy_consumption_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    
    summary_stats = filtered_df[['Energy Consumption (kWh)', 'Outdoor Temperature (°C)', 
                                 'Household Size', 'hour']].describe()
    st.dataframe(summary_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #94A3B8; padding: 20px;'>
        <p>⚡ Smart Home Energy Dashboard | Built with Streamlit & Plotly</p>
        <p style='font-size: 0.8rem;'>Powered by Machine Learning for Sustainable Energy Management</p>
    </div>
""", unsafe_allow_html=True)
