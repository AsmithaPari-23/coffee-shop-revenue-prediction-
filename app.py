import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# Set page configuration for centered layout and title
st.set_page_config(page_title="Enhanced Sales Prediction Dashboard", layout="centered")

# App title and description
st.title("☕ Coffee Shop Sales Prediction Dashboard")
st.write("Upload your CSV file with sales data, and forecast future sales with a variety of visualizations and insights.")

# Sidebar for forecast options and additional settings
st.sidebar.header("🔧 Customization Settings")
forecast_period = st.sidebar.slider("Select Forecast Period (days)", min_value=7, max_value=90, value=30, step=1)
show_confidence_intervals = st.sidebar.checkbox("Show Confidence Intervals", value=True)
plot_color = st.sidebar.color_picker("Pick a Plot Color", value="#1f77b4")
metric_color = st.sidebar.color_picker("Pick a Metric Color", value="#FF5733")
smooth_forecast = st.sidebar.checkbox("Smooth Forecast Line", value=False)

# File uploader for the dataset
uploaded_file = st.file_uploader("Upload your CSV file with Date and Total Sales columns", type="csv")

# Cache the model training function
@st.cache(allow_output_mutation=True)
def train_and_forecast(data, days=90):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast, model

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Load and prepare the dataset
        df_sales = pd.read_csv(uploaded_file)
        df_sales['Date'] = pd.to_datetime(df_sales['Date'])
        df_sales = df_sales.rename(columns={'Date': 'ds', 'Total_Sales': 'y'})
        
        # Show a preview of the uploaded data
        st.write("### 📄 Data Preview")
        st.write(df_sales.head())
        
        # Summary statistics
        st.write("### 📈 Sales Summary")
        total_sales = df_sales['y'].sum()
        avg_sales = df_sales['y'].mean()
        min_sales = df_sales['y'].min()
        max_sales = df_sales['y'].max()
        
        # Display colored metrics
        st.markdown(f'<p style="color:{metric_color}; font-size:20px;">Total Sales: ₹{total_sales:,.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{metric_color}; font-size:20px;">Average Daily Sales: ₹{avg_sales:,.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{metric_color}; font-size:20px;">Min Sales: ₹{min_sales:,.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{metric_color}; font-size:20px;">Max Sales: ₹{max_sales:,.2f}</p>', unsafe_allow_html=True)
        
        # Train model and forecast for 90 days
        with st.spinner("Training model..."):
            forecast, model = train_and_forecast(df_sales, days=90)
        
        # Display forecast table based on selected forecast period
        forecast_table = forecast[['ds', 'yhat']].tail(forecast_period)
        forecast_table = forecast_table.rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales'})
        forecast_table['Predicted Sales'] = forecast_table['Predicted Sales'].apply(lambda x: f"₹{x:,.2f}")
        
        st.write(f"### 🔮 Forecasted Sales for the Next {forecast_period} Days")
        st.write(forecast_table)
        
        # Allow users to download forecasted data as CSV
        csv = forecast_table.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="forecasted_sales.csv">📥 Download Forecasted Data as CSV</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Visualization 1: Forecast Plot with Confidence Intervals
        plt.style.use('default')
        st.write("### 📊 Forecast Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if show_confidence_intervals:
            ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color=plot_color, alpha=0.2, label="Confidence Interval")
        
        ax.plot(forecast['ds'], forecast['yhat'], color=plot_color, label="Predicted Sales")
        
        if smooth_forecast:
            forecast['smoothed_yhat'] = forecast['yhat'].rolling(window=7).mean()  # 7-day moving average
            ax.plot(forecast['ds'], forecast['smoothed_yhat'], color="orange", linestyle="--", label="Smoothed Sales")
        
        ax.set_xlim([forecast['ds'].iloc[-forecast_period], forecast['ds'].iloc[-1]])
        ax.set_title(f"Forecast for the Next {forecast_period} Days", fontsize=16)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (₹)")
        ax.legend()
        st.pyplot(fig)
        
        # Visualization 2: Histogram of Historical Sales Data
        st.write("### 🏷️ Distribution of Historical Sales")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df_sales['y'], kde=True, color=plot_color, ax=ax)
        ax.set_title("Distribution of Daily Sales")
        ax.set_xlabel("Sales (₹)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
        
        # Actionable Insights Section
        st.write("### 📈 Sales Improvement Insights")
        
        with st.expander("**1. Sales Trend Analysis by Day of the Week**"):
            df_sales['Day of Week'] = df_sales['ds'].dt.day_name()
            average_sales_by_day = df_sales.groupby('Day of Week')['y'].mean().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            )
            st.bar_chart(average_sales_by_day)
            high_sales_day = average_sales_by_day.idxmax()
            low_sales_day = average_sales_by_day.idxmin()
            st.write(f"🔹 **Peak Sales Day**: {high_sales_day} - Plan promotions on this day.")
            st.write(f"🔹 **Slow Sales Day**: {low_sales_day} - Consider discounts or special offers.")

        with st.expander("**2. Top Products Recommendation**"):
            if 'Product' in df_sales.columns:
                product_sales = df_sales.groupby('Product')['y'].sum().sort_values(ascending=False).head(5)
                st.write("**Top 5 Best-Selling Products**")
                st.bar_chart(product_sales)
                st.write("Prioritize stocking these items to meet demand and prevent stockouts.")
            else:
                st.write("Upload data with a 'Product' column to see recommendations.")
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

# App Instructions
st.write("#### ℹ️ Instructions")
st.write("""
1. Upload a CSV file with at least two columns: **Date** (for dates) and **Total_Sales** (for sales data).
2. Adjust the forecast period, color preferences, and display options from the sidebar.
3. The dashboard includes multiple visualizations to help you understand the sales forecast and historical trends.
""")

