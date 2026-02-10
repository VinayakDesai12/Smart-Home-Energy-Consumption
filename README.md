# ⚡ Smart Home Energy Consumption Dashboard

A comprehensive, interactive dashboard for analyzing and predicting smart home energy consumption patterns using machine learning.

## 🌟 Features

### 📊 **5 Interactive Pages**

1. **🏠 Overview**
   - Real-time key metrics (Total Energy, Average Consumption, Peak Usage, Active Homes)
   - Energy consumption by appliance and season
   - Daily consumption trends with interactive charts

2. **📊 Analytics**
   - **Time Analysis**: Hourly patterns, time period distribution, heatmaps
   - **Household Analysis**: Energy usage by household size with box plots
   - **Temperature Impact**: Correlation analysis and scatter plots
   - **Comparative Analysis**: Multi-dimensional comparisons

3. **🔮 Predictions**
   - ML-powered energy consumption predictor
   - Input parameters: Appliance type, season, temperature, household size, time
   - Real-time predictions with cost estimates
   - Comparison with historical averages

4. **📈 Model Performance**
   - Performance metrics (MAE, RMSE, R² Score)
   - Prediction vs Actual scatter plots
   - Residual distribution analysis
   - Model insights and details

5. **🔍 Data Explorer**
   - Advanced filtering capabilities
   - Interactive data table
   - Summary statistics
   - CSV export functionality

## 🎨 Design Features

- **Modern Dark Theme** with gradient styling
- **Glassmorphism Effects** for cards and containers
- **Interactive Plotly Charts** with smooth animations
- **Responsive Layout** that adapts to different screen sizes
- **Premium Color Palette** with purple and cyan gradients

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   streamlit run dashboard.py
   ```

3. **Access the Dashboard**
   - The dashboard will automatically open in your default browser
   - Default URL: `http://localhost:8501`

## 📦 Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning model

## 📊 Dataset

The dashboard uses the `dataset.csv` file containing:
- **Home ID**: Unique identifier for each home
- **Appliance Type**: Type of appliance (AC, Heater, Fridge, etc.)
- **Energy Consumption (kWh)**: Energy usage in kilowatt-hours
- **Time**: Time of day
- **Date**: Date of measurement
- **Outdoor Temperature (°C)**: Temperature in Celsius
- **Season**: Season of the year
- **Household Size**: Number of people in the household

## 🤖 Machine Learning Model

- **Algorithm**: Histogram-based Gradient Boosting Regressor
- **Features**: Appliance Type, Season, Temperature, Household Size, Hour, Month, Weekday
- **Performance**: 
  - R² Score: ~0.76 (explains 76% of variance)
  - MAE: ~0.48 kWh
  - RMSE: ~0.76 kWh

## 🎯 Use Cases

1. **Energy Monitoring**: Track and analyze household energy consumption patterns
2. **Cost Optimization**: Identify high-consumption periods and appliances
3. **Predictive Planning**: Forecast future energy needs
4. **Sustainability**: Make data-driven decisions for energy efficiency
5. **Research**: Analyze correlations between various factors and energy usage

## 📱 Navigation

Use the sidebar to:
- Switch between different pages
- View quick statistics
- Apply date range filters
- Filter by season

## 💡 Tips for Best Experience

1. **Explore Filters**: Use sidebar filters to focus on specific time periods or seasons
2. **Interactive Charts**: Hover over charts for detailed information
3. **Try Predictions**: Experiment with different parameters in the Predictions page
4. **Export Data**: Download filtered data from the Data Explorer for further analysis
5. **Compare Patterns**: Use the Analytics page to identify consumption patterns

## 🔧 Customization

You can customize the dashboard by:
- Modifying color schemes in the CSS section
- Adding new visualizations
- Adjusting model parameters
- Creating additional analysis pages

## 📈 Future Enhancements

- Real-time data integration
- Advanced anomaly detection
- Energy-saving recommendations
- Multi-home comparison
- Mobile app version
- API integration for smart home devices

## 🤝 Contributing

Feel free to fork this project and submit pull requests for any improvements!

## 📄 License

This project is open source and available for educational and research purposes.

## 🙏 Acknowledgments

Built with modern web technologies and machine learning for sustainable energy management.

---

**⚡ Powered by Streamlit, Plotly, and Machine Learning**
