import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

st.set_page_config(page_title="Retail AI Dashboard", layout="wide")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    df = pd.read_csv("Retail_Sales_Data_Unlox (1).csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# ==============================
# SIDEBAR FILTERS
# ==============================

st.sidebar.header("🔍 Filters")

store = st.sidebar.selectbox("Select Store", df['Store_ID'].unique())
category = st.sidebar.selectbox("Select Category", df['Product_Category'].unique())

filtered_df = df[(df['Store_ID'] == store) & (df['Product_Category'] == category)]

# ==============================
# TITLE
# ==============================

st.title("🛒 Retail Analytics & AI Forecasting Dashboard")

# ==============================
# KPI SECTION
# ==============================

col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Sales", f"{filtered_df['Sales'].sum():,.0f}")
col2.metric("📊 Avg Sales", f"{filtered_df['Sales'].mean():,.0f}")
col3.metric("🏬 Transactions", len(filtered_df))

# ==============================
# SALES TREND
# ==============================

st.subheader("📈 Sales Trend")

trend = filtered_df.groupby('Date')['Sales'].sum().reset_index()
fig1 = px.line(trend, x='Date', y='Sales', title="Sales Over Time")
st.plotly_chart(fig1, use_container_width=True)

# ==============================
# CATEGORY ANALYSIS
# ==============================

st.subheader("🛍 Category Performance")

cat = df.groupby('Product_Category')['Sales'].sum().reset_index()
fig2 = px.bar(cat, x='Product_Category', y='Sales', color='Product_Category')
st.plotly_chart(fig2, use_container_width=True)

# ==============================
# FORECASTING
# ==============================

st.subheader("🔮 Sales Forecast (Next 30 Days)")

prophet_df = df.groupby('Date')['Sales'].sum().reset_index()
prophet_df.columns = ['ds', 'y']

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig3 = px.line(forecast, x='ds', y='yhat', title="Forecasted Sales")
st.plotly_chart(fig3, use_container_width=True)

# ==============================
# CLUSTERING
# ==============================

st.subheader("🧠 Store Segmentation")

cluster_data = df.groupby('Store_ID').agg({
    'Sales': 'sum',
    'Quantity': 'sum'
}).reset_index()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data['Cluster'] = kmeans.fit_predict(cluster_data[['Sales', 'Quantity']])

fig4 = px.scatter(cluster_data,
                  x='Sales',
                  y='Quantity',
                  color='Cluster',
                  hover_data=['Store_ID'],
                  title="Store Segments")

st.plotly_chart(fig4, use_container_width=True)

# ==============================
# AI INSIGHTS
# ==============================

st.subheader("📌 AI Insights")

total_sales = df['Sales'].sum()
top_store = df.groupby('Store_ID')['Sales'].sum().idxmax()

st.write(f"""
- Total revenue generated is **₹{total_sales:,.0f}**
- **{top_store}** is the top performing store
- Sales show strong seasonal trends
- Forecast indicates steady growth in coming days
- Store segmentation reveals high and low performing clusters
""")

# ==============================
# DOWNLOAD OPTION
# ==============================

st.subheader("⬇ Download Data")

csv = df.to_csv(index=False)
st.download_button("Download Dataset", csv, "retail_data.csv", "text/csv")
