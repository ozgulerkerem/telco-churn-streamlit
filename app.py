
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Churn Risk + Retention Simulator", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.joblib")


def clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # blanks / whitespace-only -> NA
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # TotalCharges fix (common ' ' values)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # drop columns not needed
    for col in ["customerID", "Churn"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


model = load_model()

st.title("Churn Risk Scoring + Retention Simulator")
tab1, tab2, tab3 = st.tabs(["Single Customer", "Batch Scoring", "Retention Simulator"])


with tab1:
    st.subheader("Single Customer (one-row CSV)")
    file_one = st.file_uploader(
        "Upload one-row CSV (same columns as dataset, excluding Churn)",
        type=["csv"],
        key="one",
    )

    if file_one:
        x_one = pd.read_csv(file_one)
        x_one = clean_input(x_one)

        if len(x_one) == 0:
            st.error("Uploaded CSV has no rows.")
        else:
            x_one = x_one.iloc[[0]]
            proba = model.predict_proba(x_one)[:, 1][0]
            st.metric("Churn probability", f"{proba:.3f}")


with tab2:
    st.subheader("Batch Scoring (CSV Upload)")
    file_batch = st.file_uploader(
        "Upload CSV (same columns as dataset, excluding Churn)",
        type=["csv"],
        key="batch",
    )

    if file_batch:
        df_in = pd.read_csv(file_batch)
        df_in = clean_input(df_in)

        if len(df_in) == 0:
            st.error("Uploaded CSV has no rows.")
        else:
            probs = model.predict_proba(df_in)[:, 1]
            out = df_in.copy()
            out["churn_proba"] = probs
            out = out.sort_values("churn_proba", ascending=False)

            st.dataframe(out.head(20), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scored CSV",
                data=csv_bytes,
                file_name="scored_customers.csv",
                mime="text/csv",
            )


with tab3:
    st.subheader("Retention ROI Simulator")

    success_rate = st.slider("Success rate", 0.0, 1.0, 0.25, 0.05)
    expected_months = st.slider("Expected months remaining", 1, 24, 6, 1)
    contact_cost = st.number_input("Contact cost (€)", min_value=0.0, value=2.0, step=1.0)
    discount_cost = st.number_input("Discount cost (€)", min_value=0.0, value=10.0, step=1.0)

    strategy = st.radio("Targeting strategy", ["Top-k", "Threshold"], horizontal=True)

    file_roi = st.file_uploader("Upload CSV for ROI simulation", type=["csv"], key="roi")

    if file_roi:
        df_sim = pd.read_csv(file_roi)
        df_sim = clean_input(df_sim)

        if len(df_sim) == 0:
            st.error("Uploaded CSV has no rows.")
        else:
            probs = model.predict_proba(df_sim)[:, 1]
            df_sim["churn_proba"] = probs

            if strategy == "Top-k":
                top_k = st.slider("Top-k %", 0.01, 0.50, 0.10, 0.01)
                k = int(np.ceil(len(df_sim) * top_k))
                df_sim = df_sim.sort_values("churn_proba", ascending=False)
                df_sim["targeted"] = 0
                df_sim.iloc[:k, df_sim.columns.get_loc("targeted")] = 1
            else:
                threshold = st.slider("Threshold", 0.05, 0.95, 0.35, 0.05)
                df_sim["targeted"] = (df_sim["churn_proba"] >= threshold).astype(int)

            if "MonthlyCharges" not in df_sim.columns:
                st.error("MonthlyCharges column is missing in the uploaded CSV.")
            else:
                df_sim["expected_value_at_risk"] = (
                    df_sim["MonthlyCharges"] * expected_months * df_sim["churn_proba"]
                )
                total_cost = (df_sim["targeted"] * (contact_cost + discount_cost)).sum()
                expected_saved = (df_sim["targeted"] * success_rate * df_sim["expected_value_at_risk"]).sum()
                net_profit = expected_saved - total_cost

                st.metric("Targeted customers", int(df_sim["targeted"].sum()))
                st.metric("Expected saved value (€)", f"{expected_saved:,.2f}")
                st.metric("Total cost (€)", f"{total_cost:,.2f}")
                st.metric("Net profit (€)", f"{net_profit:,.2f}")

                st.dataframe(
                    df_sim.sort_values("churn_proba", ascending=False).head(20),
                    use_container_width=True,
                )
