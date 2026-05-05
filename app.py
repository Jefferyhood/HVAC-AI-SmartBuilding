import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# --------------------------
# PAGE SETUP
# --------------------------
st.set_page_config(
    page_title="Smart Building AI Dashboard",
    page_icon="🏢",
    layout="wide"
)

# --------------------------
# LOAD DATA
# --------------------------
df = pd.read_parquet("hvac_data_cleaned.parquet")

features = [
    "ambient_temp",
    "ambient_humidity",
    "active_power",
    "co2_1",
    "on_off",
    "damper",
    "inlet_temp",
    "outlet_temp",
    "outside_temp",
    "high_pressure_1",
    "low_pressure_1"
]

df_clean = df[features].dropna()

# --------------------------
# TRAIN MODELS
# --------------------------
anomaly_model = IsolationForest(contamination=0.05, random_state=42)
anomaly_model.fit(df_clean)

df_clean["anomaly"] = anomaly_model.predict(df_clean)
df_clean["anomaly"] = df_clean["anomaly"].map({1: 0, -1: 1})

forecast_features = [
    "ambient_humidity",
    "active_power",
    "co2_1",
    "on_off",
    "damper",
    "inlet_temp",
    "outlet_temp",
    "outside_temp",
    "high_pressure_1",
    "low_pressure_1"
]

X = df_clean[forecast_features]
y = df_clean["ambient_temp"]

forecast_model = LinearRegression()
forecast_model.fit(X, y)

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("🏢 Smart Building AI")
st.sidebar.write("Use this app to monitor HVAC behavior and simulate building conditions.")

st.sidebar.markdown("---")
st.sidebar.subheader("Project Features")
st.sidebar.write("✅ Anomaly Detection")
st.sidebar.write("✅ Temperature Prediction")
st.sidebar.write("✅ Scenario Simulation")
st.sidebar.write("✅ HVAC Dashboard")

# --------------------------
# HEADER
# --------------------------
st.title("🏢 Smart Building AI Dashboard")
st.markdown(
    """
    This web application uses machine learning to analyze HVAC sensor behavior, 
    detect abnormal patterns, predict ambient temperature, and simulate smart building scenarios.
    """
)

st.markdown("---")

# --------------------------
# TOP METRICS
# --------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Records", f"{len(df_clean):,}")
col2.metric("Detected Anomalies", int(df_clean["anomaly"].sum()))
col3.metric("Avg Ambient Temp", f"{df_clean['ambient_temp'].mean():.2f} °C")
col4.metric("Avg Power", f"{df_clean['active_power'].mean():.2f}")

# --------------------------
# TABS
# --------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard",
    "🧠 Scenario Simulator",
    "⚙️ HVAC Signals",
    "📁 Dataset"
])

# --------------------------
# TAB 1: DASHBOARD
# --------------------------
with tab1:
    st.header("AI Monitoring Overview")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Ambient Temperature with AI-Detected Anomalies")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_clean["ambient_temp"].values[:500], label="Ambient Temperature")

        plot_data = df_clean.head(500).copy()
        detected = plot_data[plot_data["anomaly"] == 1]

        ax.scatter(
            detected.index,
            detected["ambient_temp"],
            label="Detected Anomaly"
        )

        ax.set_xlabel("Record")
        ax.set_ylabel("Temperature °C")
        ax.set_title("Temperature Trend and Anomaly Detection")
        ax.legend()

        st.pyplot(fig)

    with right:
        st.subheader("System Summary")

        anomaly_rate = (df_clean["anomaly"].sum() / len(df_clean)) * 100

        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        st.metric("Average CO2", f"{df_clean['co2_1'].mean():.2f}")
        st.metric("Average Humidity", f"{df_clean['ambient_humidity'].mean():.2f}%")

        st.info(
            "The AI model flags unusual HVAC patterns using temperature, humidity, power, CO2, damper, and pressure signals."
        )

# --------------------------
# TAB 2: SCENARIO SIMULATOR
# --------------------------
with tab2:
    st.header("Interactive HVAC Scenario Simulator")

    st.write(
        "Adjust the building conditions below. The AI will predict ambient temperature, check if the scenario looks abnormal, and provide an operational insight."
    )

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        humidity = st.slider("Ambient Humidity (%)", 0.0, 100.0, 40.0)
        power = st.slider("Active Power", 0.0, 2.0, 0.5)
        co2 = st.slider("CO2 Level", 300, 2000, 800)

    with input_col2:
        on_off = st.selectbox("System On/Off", [0, 1], index=1)
        damper = st.slider("Damper Position (%)", 0.0, 100.0, 50.0)
        outside_temp = st.slider("Outside Temperature (°C)", 0.0, 45.0, 24.0)

    with input_col3:
        inlet_temp = st.slider("Inlet Temperature (°C)", 0.0, 40.0, 22.0)
        outlet_temp = st.slider("Outlet Temperature (°C)", 0.0, 40.0, 24.0)
        high_pressure = st.slider("High Pressure", 0.0, 30.0, 15.0)
        low_pressure = st.slider("Low Pressure", 0.0, 30.0, 15.0)

    st.markdown("---")

    if st.button("Run AI Scenario", use_container_width=True):

        input_df = pd.DataFrame([{
            "ambient_humidity": humidity,
            "active_power": power,
            "co2_1": co2,
            "on_off": on_off,
            "damper": damper,
            "inlet_temp": inlet_temp,
            "outlet_temp": outlet_temp,
            "outside_temp": outside_temp,
            "high_pressure_1": high_pressure,
            "low_pressure_1": low_pressure
        }])

        predicted_temp = forecast_model.predict(input_df)[0]

        anomaly_input = pd.DataFrame([{
            "ambient_temp": predicted_temp,
            "ambient_humidity": humidity,
            "active_power": power,
            "co2_1": co2,
            "on_off": on_off,
            "damper": damper,
            "inlet_temp": inlet_temp,
            "outlet_temp": outlet_temp,
            "outside_temp": outside_temp,
            "high_pressure_1": high_pressure,
            "low_pressure_1": low_pressure
        }])

        anomaly_result = anomaly_model.predict(anomaly_input)
        status = "Anomaly" if anomaly_result[0] == -1 else "Normal"

        result1, result2, result3 = st.columns(3)

        result1.metric("Predicted Ambient Temp", f"{predicted_temp:.2f} °C")
        result2.metric("AI System Status", status)
        result3.metric("Scenario Power", f"{power:.2f}")

        if status == "Anomaly":
            st.error("AI Status: This scenario appears abnormal compared to learned HVAC behavior.")
        else:
            st.success("AI Status: This scenario appears normal based on learned HVAC behavior.")

        st.subheader("AI Insight")

        if on_off == 0 and power > 0.5:
            st.warning("System is marked OFF but power usage is still present. This may indicate abnormal equipment behavior.")
        elif power > 1.0 and humidity > 50:
            st.warning("Possible HVAC inefficiency or high-load condition. The system may be using more energy while humidity remains elevated.")
        elif co2 > 1000:
            st.warning("Poor ventilation may be present due to elevated CO2 levels.")
        elif damper < 20 and predicted_temp > 25:
            st.warning("Damper position is low while predicted temperature is high. This may suggest airflow restriction.")
        elif high_pressure > 22 or low_pressure < 5:
            st.warning("Pressure readings may indicate possible HVAC system stress.")
        elif status == "Anomaly":
            st.error("The AI detected an unusual HVAC operating pattern.")
        else:
            st.success("System appears to be operating normally.")

# --------------------------
# TAB 3: HVAC SIGNALS
# --------------------------
with tab3:
    st.header("HVAC System Signals")

    st.subheader("Equipment Status")
    status_col1, status_col2 = st.columns(2)

    status_col1.metric("Average On/Off Status", f"{df_clean['on_off'].mean():.2f}")
    status_col2.metric("Average Damper Position", f"{df_clean['damper'].mean():.2f}")

    st.line_chart(df_clean[["on_off", "damper"]].head(500))

    st.subheader("Temperature Signals")
    st.line_chart(
        df_clean[
            ["ambient_temp", "inlet_temp", "outlet_temp", "outside_temp"]
        ].head(500)
    )

    st.subheader("Pressure Signals")
    st.line_chart(
        df_clean[
            ["high_pressure_1", "low_pressure_1"]
        ].head(500)
    )

    st.subheader("Power and Air Quality")
    st.line_chart(
        df_clean[
            ["active_power", "co2_1", "ambient_humidity"]
        ].head(500)
    )

# --------------------------
# TAB 4: DATASET
# --------------------------
with tab4:
    st.header("Dataset Preview")

    st.write("This is the cleaned HVAC dataset used to train the AI models.")

    st.dataframe(df_clean.head(100), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(df_clean.describe(), use_container_width=True)

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.markdown("Built by Jeffrey Hood | Smart Building AI Final Project")