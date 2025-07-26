import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from datetime import datetime

st.title("🏠 London Housing Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/london_housing.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
boroughs = st.sidebar.multiselect("Select Borough(s)", df["Borough"].unique(), df["Borough"].unique())
date_range = st.sidebar.date_input("Date Range", [df["Date"].min().date(), df["Date"].max().date()])
model_choice = st.sidebar.selectbox("Model", ["RandomForest", "KNN", "LGBM"])

filtered = df[
    (df["Borough"].isin(boroughs)) &
    (df["Date"].dt.date.between(date_range[0], date_range[1]))
]

# KPIs
st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Average Price £", f"{filtered['Average_price'].mean():,.0f}")
col2.metric("Total Sales", f"{filtered['No_of_Sales'].sum():,}")

# Trends
st.subheader("Price Over Time")
fig1 = px.line(filtered, x="Date", y="Average_price", color="Borough")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Sales Over Time")
fig2 = px.line(filtered, x="Date", y="No_of_Sales", color="Borough")
st.plotly_chart(fig2, use_container_width=True)

# Borough Comparison at latest date
latest = filtered[filtered["Date"] == filtered["Date"].max()]
st.subheader(f"Average Price as of {filtered['Date'].max().date()}")
fig3 = px.bar(latest, x="Borough", y="Average_price", color="Average_price",
               color_continuous_scale="viridis")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Prediction section (if model and user inputs exist)
if "Square_meters" in df.columns:
    st.subheader("Predict Housing Price")
    model = load_model()

    sqm = st.number_input("Size (sqm)", value=50)
    rooms = st.number_input("Number of rooms", min_value=0, step=1)
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    # add other features as available...

    if st.button("Predict Price (£)"):
        X = pd.DataFrame([[sqm, rooms, bedrooms]], columns=["sqm","rooms","bedrooms"])
        y_pred = model.predict(X)[0]
        st.success(f"Estimated Price: £{y_pred:,.0f}")

# Download
st.markdown("### Download Filtered Data")
csv = filtered.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "filtered_london_housing.csv", "text/csv")
