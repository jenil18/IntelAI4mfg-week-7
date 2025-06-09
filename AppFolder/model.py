import os
import pandas as pd
import streamlit as st
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# —————————————————————————————— 
# 1. Load and cache the data + model training 
# —————————————————————————————— 
@st.cache_data
def load_and_train(path):
    # Ensure the file is not None and is of valid type
    ext = os.path.splitext(path.name)[1].lower()  # Use path.name to get the file name
    if ext not in (".xls", ".xlsx", ".csv"):
        st.error("Unsupported file type. Please upload a .csv or .xls/.xlsx file.")
        st.stop()

    try:
        if ext == ".xlsx":
            # Attempt to read Excel file with openpyxl
            df = pd.read_excel(path, engine='openpyxl')
        elif ext == ".xls":
            # Attempt to read older Excel file with xlrd
            df = pd.read_excel(path, engine='xlrd')
        elif ext == ".csv":
            # Attempt to read CSV file
            df = pd.read_csv(path)
        else:
            st.error("Unsupported file type! Only CSV and Excel files are allowed.")
            st.stop()

    except ValueError as e:
        st.error(f"Error: {e}. Ensure the file is an Excel file (.xls or .xlsx) or CSV.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.stop()

    X = df.drop(columns=["wait_time"])
    y = df["wait_time"]

    categorical_cols = ["geolocation_state_customer", "geolocation_state_seller"]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_test)
    return model, numerical_cols, categorical_cols, mean_absolute_error(y_test, val_pred), r2_score(y_test, val_pred)

# —————————————————————————————— 
# 2. Streamlit UI Setup 
# —————————————————————————————— 
st.title("🚛 Delivery Time Predictor")

data_file = st.sidebar.file_uploader(
    "Upload your data file", 
    type=["csv", "xls", "xlsx"]
)

if data_file is not None:
    try:
        model, numerical_cols, categorical_cols, mae, r2 = load_and_train(data_file)

        st.sidebar.markdown(f"**Validation MAE:** {mae:.2f}")
        st.sidebar.markdown(f"**Validation R²:**  {r2:.2f}")

        # —————————————————————————————— 
        # 3. User Input Section 
        # —————————————————————————————— 
        st.header("Enter New Order Details")
        input_data = {}

        for col in numerical_cols:
            input_data[col] = st.number_input(
                label=col.replace("_", " ").title(),
                value=0.0
            )

        for col in categorical_cols:
            input_data[col] = st.text_input(
                label=col.replace("_", " ").title(),
                value=""
            )

        # —————————————————————————————— 
        # 4. Prediction Button 
        # —————————————————————————————— 
        if st.button("Predict Delivery Time"):
            df_new = pd.DataFrame({k: [v] for k, v in input_data.items()})
            pred = model.predict(df_new)[0]
            st.success(f"🚚 Predicted order‑to‑delivery time: **{pred:.2f}**")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload a dataset to continue.")
