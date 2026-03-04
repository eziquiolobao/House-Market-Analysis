"""
Worcester County House Price Predictor — Streamlit App
"""

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="Worcester County House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---- Custom CSS ----
st.markdown("""
<style>
.price-card {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: white;
    padding: 2rem;
    border-radius: 1rem;
    text-align: center;
    margin: 1rem 0;
}
.price-card h1 {
    font-size: 3rem;
    margin: 0;
}
.price-card p {
    font-size: 1.1rem;
    opacity: 0.9;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")


@st.cache_resource
def load_model():
    """Load the trained model artifact."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def build_feature_vector(sqft, beds, baths, lot_size, year_built, sale_month):
    """Build feature array matching the notebook's FEATURES list."""
    house_age = 2025 - year_built
    sale_month_sin = np.sin(2 * np.pi * sale_month / 12)
    sale_month_cos = np.cos(2 * np.pi * sale_month / 12)

    return pd.DataFrame([{
        "square_feet": sqft,
        "baths": baths,
        "beds": beds,
        "lot_size": lot_size,
        "house_age": house_age,
        "sale_month_sin": sale_month_sin,
        "sale_month_cos": sale_month_cos,
    }])


def main():
    artifact = load_model()

    if artifact is None:
        st.error(
            "Model not found at `model/best_model.pkl`. "
            "Run the analysis notebook first to train and export the model."
        )
        return

    pipeline = artifact["pipeline"]
    features = artifact["features"]
    model_name = artifact["model_name"]
    train_mae = artifact["train_mae"]
    price_stats = artifact["price_stats"]
    n_samples = artifact["n_samples"]

    # ---- Sidebar ----
    st.sidebar.title("Property Details")

    sqft = st.sidebar.slider("Square Feet", 1000, 6500, 2800, step=50)
    beds = st.sidebar.selectbox("Bedrooms", [2, 3, 4, 5], index=1)
    baths = st.sidebar.selectbox(
        "Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], index=3
    )
    lot_size = st.sidebar.number_input(
        "Lot Size (sq ft)", min_value=10_000, max_value=1_500_000,
        value=87_000, step=5_000
    )
    year_built = st.sidebar.slider("Year Built", 1700, 2025, 1976)
    city = st.sidebar.selectbox("City", ["Harvard", "Devens", "Ayer"])

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    current_month_idx = datetime.datetime.now().month - 1
    sale_month_name = st.sidebar.selectbox(
        "Sale Month", months, index=current_month_idx
    )
    sale_month = months.index(sale_month_name) + 1

    st.sidebar.markdown("---")

    # Refresh data button
    if st.sidebar.button("Refresh Data & Retrain"):
        with st.spinner("Fetching fresh data from Redfin..."):
            try:
                import data_pipeline
                data_pipeline.load_data(use_cache=False)
                st.sidebar.success("Data refreshed. Re-run the notebook to retrain the model.")
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    # ---- Main area ----
    st.title("Worcester County House Price Predictor")
    st.caption(f"Model: {model_name} | Trained on {n_samples} sales | Worcester County, MA")

    # Prediction
    X_input = build_feature_vector(sqft, beds, baths, lot_size, year_built, sale_month)
    # Reorder columns to match training features
    X_input = X_input[features]
    predicted_price = pipeline.predict(X_input)[0]

    # Price card
    st.markdown(f"""
    <div class="price-card">
        <p>Predicted Sale Price</p>
        <h1>${predicted_price:,.0f}</h1>
        <p>&plusmn; ${train_mae:,.0f} based on model accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Comparison & details ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("How Your Home Compares")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3))
        prices = np.linspace(price_stats["min"], price_stats["max"], 50)
        # Simple KDE-like visualization using the stats we have
        ax.axvline(predicted_price, color="#2563eb", linewidth=2.5, label="Your prediction")
        ax.axvline(price_stats["median"], color="#f59e0b", linewidth=2, linestyle="--",
                   label=f"Median ({price_stats['median']:,.0f})")
        ax.axvspan(price_stats["min"], price_stats["max"], alpha=0.1, color="#10b981")
        ax.set_xlim(price_stats["min"] * 0.9, price_stats["max"] * 1.1)
        ax.set_xlabel("Price ($)")
        ax.set_title("Price Range in Dataset")
        ax.legend(fontsize=8)
        ax.set_yticks([])
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Your Inputs")
        input_summary = pd.DataFrame({
            "Feature": ["Square Feet", "Bedrooms", "Bathrooms", "Lot Size",
                        "Year Built", "City", "Sale Month"],
            "Value": [f"{sqft:,}", beds, baths, f"{lot_size:,}",
                      year_built, city, sale_month_name],
        })
        st.dataframe(input_summary, hide_index=True, use_container_width=True)

    # ---- Disclaimer ----
    st.markdown("---")
    st.caption(
        f"**Disclaimer:** This model was trained on only {n_samples} property sales "
        "in Worcester County, MA (primarily Harvard). Predictions are approximate and "
        "should not be used as a substitute for professional appraisals. "
        "The model does not account for interior condition, renovations, or "
        "neighborhood-specific factors."
    )


if __name__ == "__main__":
    main()
