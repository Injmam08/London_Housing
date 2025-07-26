import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.title("🏠 London Housing Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data/london_housing.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/model.pkl")

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
boroughs = st.sidebar.multiselect("Select Area(s)", df["area"].unique(), df["area"].unique())
date_range = st.sidebar.date_input("Date Range", [df["date"].min().date(), df["date"].max().date()])
model_choice = st.sidebar.selectbox("Model", ["RandomForest", "KNN", "LGBM"])

filtered = df[
    (df["area"].isin(boroughs)) &
    (df["date"].dt.date.between(date_range[0], date_range[1]))
]

# KPIs
st.subheader("Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Average Price £", f"{filtered['average_price'].mean():,.0f}")
col2.metric("Total Sales", f"{filtered['houses_sold'].sum():,}")

# Trends
st.subheader("Price Over Time")
fig1 = px.line(filtered, x="date", y="average_price", color="area")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("Sales Over Time")
fig2 = px.line(filtered, x="date", y="houses_sold", color="area")
st.plotly_chart(fig2, use_container_width=True)

# Borough Comparison at latest date
latest = filtered[filtered["date"] == filtered["date"].max()]
st.subheader(f"Average Price as of {filtered['date'].max().date()}")
fig3 = px.bar(latest, x="area", y="average_price", color="average_price",
               color_continuous_scale="viridis")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")

# Prediction section (optional; only works if you have model & input features)
if "Square_meters" in df.columns:
    st.subheader("Predict Housing Price")
    model = load_model()

    sqm = st.number_input("Size (sqm)", value=50)
    rooms = st.number_input("Number of rooms", min_value=0, step=1)
    bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
    
    if st.button("Predict Price (£)"):
        X = pd.DataFrame([[sqm, rooms, bedrooms]], columns=["sqm", "rooms", "bedrooms"])
        y_pred = model.predict(X)[0]
        st.success(f"Estimated Price: £{y_pred:,.0f}")

# Download
st.markdown("### Download Filtered Data")
csv = filtered.to_csv(index=False).encode()
st.download_button("Download CSV", csv, "filtered_london_housing.csv", "text/csv")
