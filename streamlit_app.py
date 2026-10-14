import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="AI Energy Dashboard", layout="wide")

st.title("⚡ AI-Powered Energy Forecasting System")

# Load model
model_path = os.path.join("models", "energy_model.pkl")
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.error(f"Model not found at {model_path}. Run main.py or train the model first.")
    st.stop()

# -------------------------------
# FILE UPLOAD FEATURE 🔥
# -------------------------------
st.sidebar.header("📂 Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

def process_uploaded_file(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    # detect datetime column
    datetime_col = None
    for col in df.columns:
        if "datetime" in col.lower() or "date" in col.lower() or "time" in col.lower():
            datetime_col = col
            break

    if datetime_col is None:
        st.error("Uploaded CSV must contain a datetime-like column (name containing 'datetime', 'date' or 'time').")
        return pd.DataFrame()

    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
    except Exception as e:
        st.error(f"Failed to parse datetime column '{datetime_col}': {e}")
        return pd.DataFrame()

    # rename first non-index column to Energy if possible
    non_index_cols = list(df.columns)
    if len(non_index_cols) == 0:
        st.error("Uploaded CSV contains no data columns besides the datetime index.")
        return pd.DataFrame()

    df.rename(columns={non_index_cols[0]: "Energy"}, inplace=True)

    return df

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_default_data():
    all_data = []

    data_dir = "data"
    if not os.path.exists(data_dir):
        st.warning(f"Data directory not found: {data_dir}")
        return pd.DataFrame()

    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(data_dir, file))
            except Exception as e:
                st.warning(f"Skipping {file}: {e}")
                continue

            df.columns = df.columns.str.strip()

            datetime_col = None
            for col in df.columns:
                if "datetime" in col.lower() or "date" in col.lower() or "time" in col.lower():
                    datetime_col = col
                    break

            if datetime_col:
                try:
                    df[datetime_col] = pd.to_datetime(df[datetime_col])
                    df.set_index(datetime_col, inplace=True)
                except Exception as e:
                    st.warning(f"Failed parsing datetime in {file}: {e}")
                    continue
            else:
                st.warning(f"No datetime column detected in {file}; skipping.")
                continue

            non_index_cols = list(df.columns)
            if len(non_index_cols) == 0:
                st.warning(f"No data columns in {file}; skipping.")
                continue

            df.rename(columns={non_index_cols[0]: "Energy"}, inplace=True)
            df["region"] = file.replace(".csv", "")

            all_data.append(df)

    if not all_data:
        st.warning("No valid CSV files found in data/ folder.")
        return pd.DataFrame()

    return pd.concat(all_data)

# Choose data source
if uploaded_file is not None:
    data = process_uploaded_file(uploaded_file)
    st.success("✅ Using uploaded dataset")
else:
    data = load_default_data()
    st.info("Using default dataset")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("🔮 Prediction Inputs")

hour = st.sidebar.slider("Hour", 0, 23, 12)
day = st.sidebar.slider("Day (0=Mon)", 0, 6, 3)
month = st.sidebar.slider("Month", 1, 12, 6)

lag_1 = st.sidebar.number_input("Previous Hour Energy", value=10000)
lag_24 = st.sidebar.number_input("Previous Day Same Hour", value=12000)
rolling = st.sidebar.number_input("24hr Avg Energy", value=11000)

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("Predict"):
    features = [[hour, day, month, lag_1, lag_24, rolling]]
    try:
        if hasattr(model, 'predict'):
            pred = model.predict(features)
            st.success(f"⚡ Predicted Energy: {pred[0]:.2f} MW")
        else:
            st.error("Loaded model does not support .predict().")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# VISUALIZATION
# -------------------------------
st.subheader("📊 Energy Trend")
if not data.empty and 'Energy' in data.columns:
    st.line_chart(data['Energy'].head(1000))
else:
    st.info("No energy data available to display.")

# -------------------------------
# ANOMALY DETECTION 🔥
# -------------------------------
st.subheader("🚨 Anomaly Detection (Energy Spikes)")

# anomaly detection may rely on external module; guard import
try:
    from src.anomaly import detect_anomalies  # keep local import near usage
except Exception as e:
    st.warning(f"Anomaly detection module unavailable: {e}. Skipping anomaly detection.")

    def detect_anomalies(df):
        df['anomaly'] = 0
        return df

# only run anomaly detection if data not empty
if data.empty:
    st.info("No data available to run anomaly detection.")
else:
    try:
        anomaly_data = detect_anomalies(data.copy())
        anomalies = anomaly_data[anomaly_data.get('anomaly', 0) != 0]

        st.write(f"⚠️ Total anomalies detected: {len(anomalies)}")
        if 'Energy' in anomaly_data.columns:
            st.line_chart(anomaly_data['Energy'].head(500))

        if not anomalies.empty:
            st.warning("⚡ Unusual energy spikes detected!")
        else:
            st.success("✅ Energy usage is stable")
    except Exception as e:
        st.error(f"Error running anomaly detection: {e}")

# -------------------------------
# SMART INSIGHTS 🔥
# -------------------------------
st.subheader("🧠 Smart Insights")

if data.empty or 'Energy' not in data.columns or not isinstance(data.index, pd.DatetimeIndex):
    st.info("Insufficient time-series data to compute insights. Ensure data has a datetime index and 'Energy' column.")
else:
    try:
        peak_hour = int(data.groupby(data.index.hour)['Energy'].mean().idxmax())
        avg_usage = data['Energy'].mean()
        max_usage = data['Energy'].max()

        st.markdown(f"""
### 📊 Insights:

- 🔥 Peak energy usage occurs at **{peak_hour}:00**
- ⚡ Average consumption is **{avg_usage:.2f} MW**
- 🚀 Maximum recorded usage is **{max_usage:.2f} MW**
""")

        # Trend detection
        trend = data['Energy'].tail(50)
        if len(trend) >= 2:
            if trend.iloc[-1] > trend.iloc[0]:
                st.info("📈 Energy usage trend is increasing")
            else:
                st.info("📉 Energy usage trend is decreasing")
        else:
            st.info("Not enough points to determine trend.")
    except Exception as e:
        st.error(f"Failed to compute insights: {e}")

# -------------------------------
# PEAK ANALYSIS
# -------------------------------
st.subheader("🔥 Peak Hour Analysis")
if data.empty or 'Energy' not in data.columns or not isinstance(data.index, pd.DatetimeIndex):
    st.info("Insufficient data for peak hour analysis.")
else:
    try:
        peak_hour = int(data.groupby(data.index.hour)['Energy'].mean().idxmax())
        st.info(f"⚡ Peak usage hour: {peak_hour}:00")
    except Exception as e:
        st.error(f"Peak hour analysis failed: {e}")

# -------------------------------
# REGION FILTER (ONLY DEFAULT DATA)
# -------------------------------
if 'region' in data.columns:
    st.subheader("🌍 Region-wise Analysis")

    regions = data['region'].unique()
    selected_region = st.selectbox("Select Region", regions)

    filtered = data[data['region'] == selected_region]
    st.line_chart(filtered['Energy'].head(500))

# -------------------------------
# DISTRIBUTION
# -------------------------------
st.subheader("📈 Load Distribution")
if not data.empty and 'Energy' in data.columns:
    st.bar_chart(data['Energy'].head(100))
else:
    st.info("No energy data available for distribution chart.")

st.markdown("### 🚀 Built for Smart Cities & Energy Optimization")