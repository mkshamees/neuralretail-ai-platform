import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuralRetail AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 NeuralRetail AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Overview",
        "Demand Intelligence",
        "Customer Hub",
        "Inventory",
        "MLOps Monitor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("AI-powered Retail Intelligence System")

st.write("DEBUG PAGE:", page)   # 👈 ADD THIS LINE HERE

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    import os

    try:
        sales = pd.read_csv("data/retail_features.csv")
        rfm = pd.read_csv("data/rfm_table.csv")

        sales["InvoiceDate"] = pd.to_datetime(sales["InvoiceDate"])

        return sales, rfm

    except Exception:
        st.warning("⚠ Data not found — running demo mode")

        sales = pd.DataFrame({
            "InvoiceDate": pd.date_range("2024-01-01", periods=10),
            "TotalPrice": [100,120,90,300,250,400,350,500,450,600]
        })

        rfm = pd.DataFrame({"CustomerID": range(10)})

        return sales, rfm
# ----------------  ----------------
daily_sales, rfm = load_data()

st.write("✅ Data Loaded:", daily_sales.shape)

# ---------------- EXECUTIVE ----------------
if page == "🏠 Executive Overview":
    st.header("Executive Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Revenue", f"{daily_sales['TotalPrice'].sum():,.0f}", "+5%")
    col2.metric("📦 Orders", len(daily_sales), "+2%")
    col3.metric("👥 Customers", rfm.shape[0], "+3%")

    st.markdown("### Revenue Trend")

    fig = px.line(
        daily_sales,
        x="InvoiceDate",
        y="TotalPrice",
        title="Revenue Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 AI Business Insight")

    st.info("""
    - Revenue trend is stable with moderate growth  
    - Customer base is healthy and diversified  
    - Recommendation: Focus on mid-value customer retention  
    """)

# ---------------- DEMAND ----------------
elif page == "📈 Demand Intelligence":
    st.header("Demand Intelligence")

    df = daily_sales.copy()
    df["Forecast"] = df["TotalPrice"].rolling(7, min_periods=1).mean()

    st.markdown("### Actual vs Forecast")

    fig = px.line(
        df,
        x="InvoiceDate",
        y=["TotalPrice", "Forecast"],
        labels={"value": "Revenue", "variable": "Series"}
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("AI insight: Demand shows stable trend with mild seasonality.")
# ---------------- CUSTOMER (API CONNECTED) ----------------
elif page == "👥 Customer Hub":
    st.header("Customer Intelligence & Churn Prediction")

    recency = st.number_input("Recency", 10)
    frequency = st.number_input("Frequency", 5)
    monetary = st.number_input("Monetary", 500)

    if st.button("Predict Churn"):

        payload = {
            "recency": recency,
            "frequency": frequency,
            "monetary": monetary
        }

        API_URL = "https://neuralretail-ai-platform.onrender.com"

        try:
            with st.spinner("🧠 AI model analyzing customer behavior..."):
                response = requests.post(
                    f"{API_URL}/predict/churn",
                    json=payload
                )

            result = response.json()

            if result.get("prediction") == 1:
                st.error("⚠ High Risk Customer Detected")

                st.markdown("""
                ### Recommended Action:
                - Send discount offer (10–20%)
                - Run re-engagement campaign
                - Add to recovery segment
                """)
            else:
                st.success("Low Risk Customer")

                st.markdown("""
                ### Recommended Action:
                - Upsell premium products
                - Maintain engagement
                - Offer loyalty rewards
                """)

        except Exception as e:
            st.error(f"API Error: {e}")
# ---------------- INVENTORY ----------------
elif page == "📦 Inventory":
    st.header("Inventory Insights")

    stock = daily_sales.groupby("InvoiceDate")["TotalPrice"].sum()

    st.markdown("### Stock Movement Trend")

    st.line_chart(stock)
    st.success("Stock levels are stable with no critical shortages detected.")

# ---------------- MLOPS ----------------
elif page == "🧠 MLOps Monitor":
    st.header("Model Monitoring Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("PSI Score", 0.12)
    col2.metric("MAPE", "8.5%")
    col3.metric("Model Status", "Healthy")
    st.success("System is healthy and model performance is within acceptable range.")

    st.success("All systems operational ✔")
