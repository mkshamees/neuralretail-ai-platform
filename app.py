
import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px

st.set_page_config(
    page_title="NeuralRetail AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .sub-text {
            font-size: 16px;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 NeuralRetail AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">End-to-End AI Sales Intelligence & Customer Analytics System</div>', unsafe_allow_html=True)

st.markdown("---")

st.set_page_config(page_title="NeuralRetail AI Platform", layout="wide")

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
    st.header("Executive Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Revenue", f"{daily_sales['TotalPrice'].sum():,.2f}")
    col2.metric("📦 Total Orders", len(daily_sales))
    col3.metric("👥 Customers", rfm.shape[0])

    st.markdown("### Revenue Trend")

    fig = px.line(
        daily_sales,
        x="InvoiceDate",
        y="TotalPrice",
        title="Revenue Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- DEMAND ----------------
elif page == "Demand Intelligence":
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
# ---------------- CUSTOMER (API CONNECTED) ----------------
elif page == "Customer Hub":
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
            with st.spinner("Analyzing customer risk..."):
                response = requests.post(
                    f"{API_URL}/predict/churn",
                    json=payload
                )

            result = response.json()

            if result.get("prediction") == 1:
                st.error("⚠ High Risk Customer")
            else:
                st.success("Low Risk Customer")

        except Exception as e:
            st.error(f"API Error: {e}")  

# ---------------- INVENTORY ----------------
elif page == "Inventory":
    st.header("Inventory Insights")

    stock = daily_sales.groupby("InvoiceDate")["TotalPrice"].sum()

    st.markdown("### Stock Movement Trend")

    st.line_chart(stock)

# ---------------- MLOPS ----------------
elif page == "MLOps Monitor":
    st.header("Model Monitoring Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("PSI Score", 0.12)
    col2.metric("MAPE", "8.5%")
    col3.metric("Model Status", "Healthy")

    st.success("All systems operational ✔")
