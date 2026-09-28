
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Demand Prediction with ARIMA",
    page_icon="📈",
    layout="centered"
)

st.title("Demand Prediction with ARIMA Model")

# -----------------------------
# Load data and model
# -----------------------------
DATA_FILE = Path("demand_data - demand_data.csv")
MODEL_FILE = Path("arima_model.joblib")

try:
    model = joblib.load(MODEL_FILE)
    st.success("Model 'arima_model.joblib' loaded successfully.")
except Exception as e:
    st.error(f"Could not load the ARIMA model: {e}")
    st.stop()

try:
    df = pd.read_csv(DATA_FILE)
    df["Month"] = pd.to_datetime(df["Month"], format="%b-%Y")
    df = df.sort_values("Month").reset_index(drop=True)
except Exception as e:
    st.error(f"Could not load demand_data.csv: {e}")
    st.stop()

# -----------------------------
# Prediction section
# -----------------------------
st.header("Make a Prediction")

periods = st.slider(
    "Select number of periods to forecast (months):",
    min_value=1,
    max_value=12,
    value=8,
    step=1
)

if st.button("Generate Forecast"):
    try:
        # The model in the assignment was trained on the first 80% of observations.
        # Therefore, its forecast starts immediately after the training period.
        train_size = int(len(df) * 0.8)
        last_training_date = df.loc[train_size - 1, "Month"]

        predictions = model.predict(n_periods=periods)

        forecast_dates = pd.date_range(
            start=last_training_date + pd.offsets.MonthBegin(1),
            periods=periods,
            freq="MS"
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_dates.strftime("%Y-%m-%d"),
            "Predicted Demand (000L)": predictions.round(0).astype(int)
        })

        st.subheader(f"Forecast for the next {periods} months:")
        st.dataframe(
            forecast_df,
            use_container_width=True,
            hide_index=False
        )

        # Optional visualisation
        st.line_chart(
            forecast_df.set_index("Date")["Predicted Demand (000L)"]
        )

    except Exception as e:
        st.error(f"Forecast could not be generated: {e}")
