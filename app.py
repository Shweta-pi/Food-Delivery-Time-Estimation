import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Delivery Time Predictor",
    page_icon="🚀",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("delivery_model.pkl")

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🚀 Smart Food Delivery Time Predictor")
st.markdown("### AI-powered ETA Estimation & Risk Analysis System")

st.markdown("---")

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Delivery Partner Details")
    age = st.slider("Delivery Partner Age", 18, 60, 25)
    rating = st.slider("Delivery Partner Rating", 1.0, 5.0, 4.0)
    distance = st.slider("Distance (km)", 0.5, 25.0, 5.0)

with col2:
    st.subheader("🚴 Order Details")
    order_type = st.selectbox("Order Type",
                              ["Drinks", "Snack", "Meal", "Buffet"])

    vehicle_type = st.selectbox("Vehicle Type",
                                ["Bike", "Scooter", "Cycle"])

# ---------------- ENCODING ----------------
order_map = {"Drinks": 0, "Snack": 1, "Meal": 2, "Buffet": 3}
vehicle_map = {"Bike": 0, "Scooter": 1, "Cycle": 2}

order_encoded = order_map[order_type]
vehicle_encoded = vehicle_map[vehicle_type]

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Delivery Time"):

    input_data = np.array([[age, rating, distance,
                            order_encoded, vehicle_encoded]])

    prediction = model.predict(input_data)[0]

    # Safety Buffer Logic
    min_time = round(prediction)
    max_time = round(prediction + (prediction * 0.10) + 3)

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    # KPI CARDS
    k1, k2, k3 = st.columns(3)

    k1.metric("Predicted Time (mins)", f"{round(prediction,2)}")
    k2.metric("Minimum ETA", f"{min_time} mins")
    k3.metric("Maximum ETA", f"{max_time} mins")

    st.markdown("---")

    # ---------------- RISK ANALYSIS ----------------
    if max_time > 45:
        st.error("🚦 High Delay Risk Detected")
        risk_score = 85
    elif max_time > 30:
        st.warning("⚠ Moderate Delay Risk")
        risk_score = 60
    else:
        st.success("✅ Low Delay Risk")
        risk_score = 30

    # ---------------- GAUGE CHART ----------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Delivery Time (Minutes)"},
        gauge={
            'axis': {'range': [0, 60]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 40], 'color': "yellow"},
                {'range': [40, 60], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- RISK BAR ----------------
    risk_fig = px.bar(
        x=["Risk Score"],
        y=[risk_score],
        text=[risk_score],
        range_y=[0,100],
        title="Delivery Delay Risk Score (%)"
    )

    st.plotly_chart(risk_fig, use_container_width=True)

    # ---------------- FEATURE IMPORTANCE (IF AVAILABLE) ----------------
    if hasattr(model, "feature_importances_"):
        st.markdown("---")
        st.subheader("🔍 Feature Importance Analysis")

        features = ["Age", "Rating", "Distance",
                    "Order Type", "Vehicle Type"]

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        imp_fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation='h',
            title="Model Feature Impact"
        )

        st.plotly_chart(imp_fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning & Streamlit")
