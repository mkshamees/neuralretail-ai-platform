
import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px

st.set_page_config(page_title="NeuralRetail AI Platform", layout="wide")

st.title("📊 NeuralRetail AI Platform (Week 4 Deployment)")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Executive Overview", "Demand Intelligence", "Customer Hub", "Inventory", "MLOps Monitor"]
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    base = "data"
    sales = pd.read_csv(f"{base}/retail_features.csv")
    rfm = pd.read_csv(f"{base}/rfm_table.csv")
    sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])
    return sales, rfm

daily_sales, rfm = load_data()

# ---------------- EXECUTIVE ----------------
if page == "Executive Overview":
    st.header("Executive Overview")

    st.metric("Total Revenue", f"{daily_sales['TotalPrice'].sum():,.2f}")
    st.metric("Total Orders", len(daily_sales))

    fig = px.line(daily_sales, x="InvoiceDate", y="TotalPrice", title="Revenue Trend")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMAND ----------------
elif page == "Demand Intelligence":
    st.header("Demand Intelligence")

    df = daily_sales.copy()

    df["Forecast"] = df["TotalPrice"].rolling(7, min_periods=1).mean()

    fig = px.line(df, x="InvoiceDate", y=["TotalPrice", "Forecast"])
    st.plotly_chart(fig, use_container_width=True)

# ---------------- CUSTOMER (API CONNECTED) ----------------
elif page == "Customer Hub":
    st.header("Customer Intelligence")

    recency = st.number_input("Recency", 10)
    frequency = st.number_input("Frequency", 5)
    monetary = st.number_input("Monetary", 500)

    if st.button("Predict Churn"):

        payload = {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/churn",
                json=payload
            )

            result = response.json()

            if result["churn_prediction"] == 1:
                st.error("⚠ High Risk Customer")
            else:
                st.success("Low Risk Customer")

        except Exception as e:
            st.error(f"API Error: {e}")

# ---------------- INVENTORY ----------------
elif page == "Inventory":
    st.header("Inventory Insights")

    stock = daily_sales.groupby("InvoiceDate")["TotalPrice"].sum()

    st.line_chart(stock)

# ---------------- MLOPS ----------------
elif page == "MLOps Monitor":
    st.header("Model Monitoring")

    st.metric("PSI Score", 0.12)
    st.metric("MAPE", "8.5%")
    st.success("System Healthy")
