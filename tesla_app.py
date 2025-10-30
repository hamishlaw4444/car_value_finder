# requirements.txt (for Cursor)
# streamlit
# pandas
# numpy
# scikit-learn
# plotly
# statsmodels
# plotly.graph_objects

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration must be first
st.set_page_config(page_title="Tesla Model 3 Value Finder", layout="wide", initial_sidebar_state="expanded")

# Initialize session state for user preferences
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'miles_per_year': 5000,
        'hold_years': 2,
        'ranking_method': 'Lowest Total Monthly Cost (Recommended)',
        'price_filter_enabled': False,
        'min_budget': 15000,
        'max_budget': 30000,
        'filter_high_mileage': True
    }

# CSV Configuration (standalone mode)
CSV_PATH = "tesla_long_range_listings.csv"

# ----------------------------
# Utilities
# ----------------------------


def load_data_from_csv(path: str = CSV_PATH) -> pd.DataFrame:
    """Load data from a local CSV file for standalone, explorable deployment."""
    try:
        st.info("Loading data from CSV...")
        df = pd.read_csv(path)
        st.success(f"✅ Loaded {len(df)} listings from CSV")
        return df
    except FileNotFoundError:
        st.error(f"CSV not found at {path}. Please ensure the file exists in the repository.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def _extract_year(x: str) -> float:
    if pd.isna(x):
        return np.nan
    m = re.search(r"(20\d{2}|19\d{2})", str(x))
    return float(m.group(1)) if m else np.nan


def calculate_age_since_new(registration_year: float) -> float:
    """Calculate age in years since new based on current year"""
    current_year = pd.Timestamp.now().year
    if pd.isna(registration_year):
        return np.nan
    age = current_year - registration_year
    return max(0, age)


def _extract_advert_id_from_url(url: str) -> Optional[str]:
    """Extract the numeric advert/listing id from an AutoTrader car-details URL."""
    if pd.isna(url):
        return None
    m = re.search(r"/car-details/(\d+)", str(url))
    return m.group(1) if m else None


def clean_autotrader_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # Ensure expected columns exist, create if missing
    for col in [
        "title", "subtitle", "price", "mileage", "registered_year", "dealer_rating",
        "location", "url", "advert_id", "main_image", "price_clean", "mileage_clean",
        "year_clean", "age_clean"
    ]:
        if col not in df.columns:
            df[col] = np.nan

    # 1) Filter to Tesla Model 3 only (keep if title contains)
    mask_tm3 = df["title"].astype(str).str.contains("Tesla Model 3", case=False, na=False)
    df = df.loc[mask_tm3].copy()

    # 2) price_clean: prefer existing numeric; otherwise derive from raw price string
    if df["price_clean"].notna().any():
        df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")
    else:
        df["price_clean"] = (
            df["price"].astype(str)
            .str.replace("£", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        df["price_clean"] = pd.to_numeric(df["price_clean"], errors="coerce")

    # 3) mileage_clean: prefer existing numeric; otherwise derive from raw mileage string
    if df["mileage_clean"].notna().any():
        df["mileage_clean"] = pd.to_numeric(df["mileage_clean"], errors="coerce")
    else:
        df["mileage_clean"] = (
            df["mileage"].astype(str)
            .str.replace("miles", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        df["mileage_clean"] = pd.to_numeric(df["mileage_clean"], errors="coerce")

    # 4) year_clean: prefer existing; otherwise derive from registered_year
    if df["year_clean"].notna().any():
        df["year_clean"] = pd.to_numeric(df["year_clean"], errors="coerce")
    else:
        df["year_clean"] = df["registered_year"].apply(_extract_year)
        df["year_clean"] = pd.to_numeric(df["year_clean"], errors="coerce")

    # 5) age_clean: prefer existing; otherwise compute from year
    if df["age_clean"].notna().any():
        df["age_clean"] = pd.to_numeric(df["age_clean"], errors="coerce")
    else:
        df["age_clean"] = df["year_clean"].apply(calculate_age_since_new)

    # 6) Populate advert_id from URL when missing
    if "advert_id" not in df.columns or not df["advert_id"].notna().any():
        df["advert_id"] = df["url"].apply(_extract_advert_id_from_url)
    else:
        missing_mask = df["advert_id"].isna()
        df.loc[missing_mask, "advert_id"] = df.loc[missing_mask, "url"].apply(_extract_advert_id_from_url)

    # Final cleaning
    df = df.dropna(subset=["price_clean", "mileage_clean", "year_clean", "age_clean"]).reset_index(drop=True)
    df = df[df["mileage_clean"] >= 5000].reset_index(drop=True)

    if df["advert_id"].notna().any():
        df = df.drop_duplicates(subset=["advert_id"], keep="first")

    return df

# ----------------------------
# Modeling (NumPy-based)
# ----------------------------

class SimpleLinearModel:
    def __init__(self, intercept: float, coef: np.ndarray):
        self.intercept_ = float(intercept)
        self.coef_ = np.asarray(coef, dtype=float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return self.intercept_ + X_arr @ self.coef_


def _train_test_split_idx(n: int, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(n * (1 - test_size))
    return idx[:split], idx[split:]


def _fit_linear(X: np.ndarray, y: np.ndarray) -> SimpleLinearModel:
    # Add bias term for intercept
    X_design = np.column_stack([np.ones(len(X)), X])
    # Solve least squares
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    intercept = beta[0]
    coef = beta[1:]
    return SimpleLinearModel(intercept, coef)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _kfold_r2(model_features: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 42) -> np.ndarray:
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    scores = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        X_tr, y_tr = model_features[train_idx], y[train_idx]
        X_te, y_te = model_features[test_idx], y[test_idx]
        model = _fit_linear(X_tr, y_tr)
        y_pred = model.predict(X_te)
        scores.append(_r2_score(y_te, y_pred))
    return np.asarray(scores, dtype=float)


def fit_linear_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit linear regression model with simple validation and k-fold CV (NumPy)."""
    if len(df) < 10:
        st.warning("⚠️ Not enough data for reliable model training (need at least 10 listings)")
        return {}

    X = df[["mileage_clean", "age_clean"]].astype(float).values
    y = df["price_clean"].astype(float).values

    tr_idx, te_idx = _train_test_split_idx(len(df), test_size=0.2, seed=42)
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    model = _fit_linear(X_tr, y_tr)
    y_pred = model.predict(X_te)

    r2 = _r2_score(y_te, y_pred)
    mae = _mae(y_te, y_pred)
    rmse = _rmse(y_te, y_pred)
    cv_scores = _kfold_r2(X, y, k=5, seed=42)

    coefs = {
        "intercept": float(model.intercept_),
        "mileage_coef": float(model.coef_[0]),
        "age_coef": float(model.coef_[1])
    }

    models = {
        'linear': {
            'model': model,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_scores': cv_scores,
            'coefs': coefs
        },
        'best_model': 'linear',
        'best_model_obj': model
    }
    return models




# Removed fit_grouped_models and predict_with_grouped_models functions as they're no longer needed





def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def compute_deal_metrics(df: pd.DataFrame, depreciation_model, miles_per_year: int, hold_years: int) -> pd.DataFrame:
    """Compute deal metrics including depreciation and opportunity costs"""
    df = df.copy()
    
    # Calculate depreciation estimates
    depr_df = estimate_depreciation(df, depreciation_model, miles_per_year, hold_years)
    
    # Add depreciation data to the dataframe
    df["annual_depr_est"] = depr_df["annual_depr_est"]
    df["depreciation_est"] = depr_df["depreciation_est"]
    df["pred_future_model"] = depr_df["pred_future_model"]
    
    # Calculate monthly cost of ownership (depreciation only)
    df["monthly_cost_ownership"] = df["depreciation_est"] / (hold_years * 12)
    
    # Calculate monthly cost with opportunity cost (5% annual return on capital tied up)
    # Opportunity cost = what you could earn if the purchase price was invested at 5% annually
    annual_opportunity_cost = df["price_clean"] * 0.05
    monthly_opportunity_cost = annual_opportunity_cost / 12
    df["monthly_cost_with_opportunity"] = df["monthly_cost_ownership"] + monthly_opportunity_cost
    
    return df

def compute_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy value score function for backward compatibility"""
    df = df.copy()
    if "dealer_rating" in df.columns:
        med = df["dealer_rating"].median(skipna=True)
        df["dealer_rating_imputed"] = df["dealer_rating"].fillna(med)
    else:
        df["dealer_rating_imputed"] = np.nan

    df["_n_value_pct"] = minmax(df["value_pct"].fillna(0))
    df["_n_age"] = minmax(df["age_clean"])  # younger better
    df["_n_mileage"] = minmax(df["mileage_clean"])  # lower better (we'll subtract)
    df["_n_dealer"] = minmax(df["dealer_rating_imputed"]) if df["dealer_rating_imputed"].notna().any() else 0

    df["value_score"] = (
        0.60 * df["_n_value_pct"] +
        0.10 * (1 - df["_n_age"]) -
        0.10 * df["_n_mileage"] +
        0.05 * df["_n_dealer"]
    )
    return df


def percentile_rank(col: pd.Series, x: float) -> float:
    if len(col) == 0:
        return np.nan
    return 100.0 * (col < x).mean()


def build_reasoning(row: pd.Series, ref_df: pd.DataFrame) -> str:
    gap = row.get("value_gap", np.nan)
    pred = row.get("predicted_price", np.nan)
    actual = row.get("price_clean", np.nan)
    age = int(row.get("age_clean", np.nan)) if not pd.isna(row.get("age_clean", np.nan)) else None
    miles = row.get("mileage_clean", np.nan)
    rating = row.get("dealer_rating", np.nan)

    mile_pct = percentile_rank(ref_df["mileage_clean"], miles) if not pd.isna(miles) else np.nan

    parts = []
    if not pd.isna(gap) and not pd.isna(pred) and not pd.isna(actual):
        parts.append(f"Priced £{gap:,.0f} below model (£{pred:,.0f} vs £{actual:,.0f}).")
    if age is not None:
        parts.append(f"{age} years old")
    if not pd.isna(miles):
        parts.append(f"with {miles:,.0f} miles (~{mile_pct:.0f}th pct).")
    if not pd.isna(rating):
        parts.append(f"Dealer {rating:.1f}/5.")
    parts.append("Good value given year/mileage relative to peers.")
    return " ".join(parts)


def estimate_depreciation(df: pd.DataFrame, model: SimpleLinearModel, miles_per_year: int, hold_years: int) -> pd.DataFrame:
    df = df.copy()
    delta_miles = miles_per_year * hold_years
    X_now = df[["mileage_clean", "age_clean"]].astype(float).values
    X_future = np.column_stack([
        df["mileage_clean"].astype(float).values + float(delta_miles),
        df["age_clean"].astype(float).values + float(hold_years),  # Age increases over time
    ])

    pred_now = model.predict(X_now)
    pred_future = model.predict(X_future)

    df["pred_now_model"] = pred_now
    df["pred_future_model"] = pred_future

    df["depreciation_est"] = df["price_clean"] - df["pred_future_model"]
    df["annual_depr_est"] = df["depreciation_est"] / max(hold_years, 1)
    return df



def create_depreciation_visualization(df: pd.DataFrame, miles_per_year: int, hold_years: int) -> go.Figure:
    """Create depreciation curve visualization"""
    # Sample a few cars for visualization
    sample_df = df.sample(min(10, len(df))).copy()
    
    # Create future years for projection
    years_range = np.arange(0, hold_years + 1)
    
    fig = go.Figure()
    
    for idx, row in sample_df.iterrows():
        current_price = row['price_clean']
        current_miles = row['mileage_clean']
        current_age = row['age_clean']
        
        # Calculate depreciation curve
        future_prices = []
        for year in years_range:
            future_miles = current_miles + (miles_per_year * year)
            future_age = current_age + year
            
            # Simple linear depreciation model for visualization
            depreciation_rate = 0.15  # 15% per year
            future_price = current_price * (1 - depreciation_rate * year)
            future_prices.append(max(future_price, 0))  # Price can't go negative
        
        fig.add_trace(go.Scatter(
            x=years_range,
            y=future_prices,
            mode='lines+markers',
            name=f'£{current_price:,.0f} ({current_age} years old)',
            hovertemplate=f'Age {current_age}<br>Current: £{current_price:,.0f}<br>Future: £%{{y:,.0f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Depreciation Projection ({hold_years} years, {miles_per_year:,} miles/year)",
        xaxis_title="Years from Now",
        yaxis_title="Estimated Value (£)",
        height=500,
        showlegend=True
    )
    
    return fig

# ----------------------------
# UI
# ----------------------------

# Custom styling

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        background: #e8f4fd;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚗 Tesla Model 3 — Best Value Finder & Depreciation Estimator</h1>
    <p>Advanced analytics to find the best value Tesla Model 3 listings and estimate ownership costs</p>
</div>
""", unsafe_allow_html=True)

# Show data freshness indicator (removed)

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Always use CSV as data source (standalone)
    st.markdown("**📊 Data Source**")
    st.info("📄 Using local CSV dataset for this explorable app")
    
    # Listing filter FIRST
    st.markdown("**🎯 Listing Filter**")
    filter_high_mileage = st.checkbox(
        "🚗 Filter: Mileage > 25k only", 
        value=st.session_state.user_preferences.get('filter_high_mileage', True),
        help="Only show listings with more than 25,000 miles. This affects both model training and all subsequent analysis."
    )
    # Update session state when checkbox changes
    if filter_high_mileage != st.session_state.user_preferences.get('filter_high_mileage', True):
        st.session_state.user_preferences['filter_high_mileage'] = filter_high_mileage

    st.markdown("---")
    
    # Ownership settings AFTER filter
    st.markdown("**🔧 Ownership Settings**")
    
    miles_per_year = st.number_input(
        "🚗 Miles per year", 
        value=5000, 
        min_value=0, 
        step=500,
        help="Your expected annual mileage for depreciation calculations"
    )
    
    hold_years = st.number_input(
        "⏰ Hold period (years)", 
        value=2, 
        min_value=1, 
        step=1,
        help="How long you plan to keep the car"
    )
    
    # Removed exclude_refresh and use_drive_models options for simplicity
    
    # Removed legacy sweet spot preset for simplicity
    
    st.markdown("**🏆 Ranking Options**")
    ranking_method = st.radio(
        "How should we rank the best deals?",
        ["Lowest Total Monthly Cost (Recommended)", "Best Initial Value"],
        help="Lowest Total Monthly Cost: Sort by monthly cost including opportunity cost - finds cars that are most economical to own\nBest Initial Value: Sort by biggest discount vs predicted price - finds the best bargains at purchase"
    )

# Always load data from CSV (standalone)
base_df = load_data_from_csv(CSV_PATH)

# Fallback: allow user upload if CSV not found
if base_df.empty:
    st.warning("CSV not found or empty. You can upload a CSV manually below.")
    uploaded_csv = st.file_uploader("Upload tesla_long_range_listings.csv", type=["csv"], accept_multiple_files=False)
    if uploaded_csv is not None:
        try:
            base_df = pd.read_csv(uploaded_csv)
            st.success(f"✅ Loaded {len(base_df)} listings from uploaded CSV")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")

# Check if we have data
if base_df.empty:
    st.error("No data loaded. Please ensure the CSV exists in the repo or upload one.")
    st.stop()

clean_df = clean_autotrader_df(base_df)
# Removed exclude_refresh filter - now includes all years including 2024-2025

# Apply mileage filter if checkbox is selected
if filter_high_mileage:
    original_count = len(clean_df)
    clean_df = clean_df[clean_df["mileage_clean"] > 25000].reset_index(drop=True)
    filtered_count = len(clean_df)
    if filtered_count > 0:
        st.success(f"✅ Applied mileage filter: {filtered_count:,} listings with >25k miles (from {original_count:,} total)")
    else:
        st.warning("⚠️ No listings found with mileage >25k miles")
        st.stop()

if clean_df.empty:
    st.warning("No Tesla Model 3 rows after filtering/cleaning.")
    st.stop()

# ============================================================================
# SECTION 1: 📊 DATASET SUMMARY
# ============================================================================

st.markdown("""
<div class="section-header">
    <h2>📊 Section 1: Dataset Summary</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>This section provides an overview of the Tesla Model 3 listings data we're working with.</p>
</div>
""", unsafe_allow_html=True)

# Show filter status if active
if filter_high_mileage:
    st.info("🔍 **Active Filter:** Only showing listings with mileage > 25,000 miles")

# Summary cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📋 Total Listings</h4>
        <h2>{len(clean_df):,}</h2>
        <p>Tesla Model 3 cars</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>💰 Average Price</h4>
        <h2>£{clean_df['price_clean'].mean():,.0f}</h2>
        <p>Mean listing price</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>🚗 Average Mileage</h4>
        <h2>{clean_df['mileage_clean'].mean():,.0f}</h2>
        <p>Mean miles per car</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>📅 Median Age</h4>
        <h2>{int(np.nanmedian(clean_df['age_clean']))} years</h2>
        <p>Typical car age</p>
    </div>
    """, unsafe_allow_html=True)

# Show raw data being used for analysis
st.markdown("---")
with st.expander("📋 View Raw Listings Data", expanded=False):
    st.markdown("""
    <div class="info-box">
        <h4>🔍 Data Preview</h4>
        <p>This is the cleaned dataset. All listings shown here meet the following criteria:</p>
        <ul>
            <li>✅ Tesla Model 3 only</li>
            <li>✅ Valid price, mileage, and year data</li>
            <li>✅ Mileage ≥ 5,000 miles (filters out unrealistic listings)</li>
            <li>✅ Deduplicated by advert_id</li>
    """, unsafe_allow_html=True)
    
    if filter_high_mileage:
        st.markdown("""
            <li>✅ <strong>Mileage > 25,000 miles</strong> (active filter)</li>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </ul>
        <p><em>💡 Total of {count} listings in the dataset.</em></p>
    </div>
    """.format(count=len(clean_df)), unsafe_allow_html=True)
    
    # Show key columns
    display_cols = [
        "main_image", "price_clean", "year_clean", "age_clean", "mileage_clean", 
        "dealer_rating", "location", "title", "subtitle", "url"
    ]
    
    # Filter to only columns that exist
    display_cols_present = [col for col in display_cols if col in clean_df.columns]
    
    # Configure column display
    col_config = {
        "price_clean": st.column_config.NumberColumn("Price (£)", format="£%d"),
        "year_clean": st.column_config.NumberColumn("Year", format="%d"),
        "age_clean": st.column_config.NumberColumn("Age (Years)", format="%d"),
        "mileage_clean": st.column_config.NumberColumn("Mileage", format="%d"),
        "dealer_rating": st.column_config.NumberColumn("Dealer Rating", format="%.1f"),
    }
    
    if "main_image" in display_cols_present:
        col_config["main_image"] = st.column_config.ImageColumn("Photo", width="small")
    
    # Display the dataframe
    st.dataframe(
        clean_df[display_cols_present], 
        use_container_width=True,
        column_config=col_config,
        height=400
    )
    
    # Summary statistics
    st.markdown("### 📊 Data Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Price Range", f"£{clean_df['price_clean'].min():,.0f} - £{clean_df['price_clean'].max():,.0f}")
        st.metric("Mean Price", f"£{clean_df['price_clean'].mean():,.0f}")
        st.metric("Median Price", f"£{clean_df['price_clean'].median():,.0f}")
    
    with col2:
        st.metric("Age Range", f"{clean_df['age_clean'].min():.0f} - {clean_df['age_clean'].max():.0f} years")
        st.metric("Mean Age", f"{clean_df['age_clean'].mean():.1f} years")
        st.metric("Median Age", f"{clean_df['age_clean'].median():.0f} years")
    
    with col3:
        st.metric("Mileage Range", f"{clean_df['mileage_clean'].min():,.0f} - {clean_df['mileage_clean'].max():,.0f}")
        st.metric("Mean Mileage", f"{clean_df['mileage_clean'].mean():,.0f}")
        st.metric("Median Mileage", f"{clean_df['mileage_clean'].median():,.0f}")
    
    # Download option
    csv_data = clean_df[display_cols_present].to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Dataset (CSV)",
        data=csv_data,
        file_name="tesla_model3_dataset.csv",
        mime="text/csv",
        help="Download the complete dataset"
    )

# Price range filter UI (but don't apply yet - will apply after model training)
with st.sidebar:
    st.markdown("**💰 Price Range Filter**")
    st.info("🎯 Price range filter will be applied to results after model training")
    
    # Calculate reasonable default range (25th to 75th percentile)
    if not clean_df.empty:
        price_25th = int(clean_df["price_clean"].quantile(0.25))
        price_75th = int(clean_df["price_clean"].quantile(0.75))
        price_min = int(clean_df["price_clean"].min())
        price_max = int(clean_df["price_clean"].max())
    else:
        price_25th, price_75th, price_min, price_max = 15000, 25000, 10000, 35000
    
    col1, col2 = st.columns(2)
    with col1:
        min_budget = st.number_input(
            "Min budget (£)", 
            value=price_25th,
            min_value=price_min,
            max_value=price_max,
            step=1000,
            help="Minimum price you're willing to pay"
        )
    with col2:
        max_budget = st.number_input(
            "Max budget (£)", 
            value=price_75th,
            min_value=price_min,
            max_value=price_max,
            step=1000,
            help="Maximum price you're willing to pay"
        )

# ============================================================================
# SECTION 2: 🤖 DEPRECIATION MODEL
# ============================================================================

st.markdown("""
<div class="section-header">
    <h2>🤖 Section 2: Depreciation Model</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>This section explains the linear regression model used to predict Tesla Model 3 prices based on age and mileage. 
    The model helps identify good deals and estimate future depreciation.</p>
</div>
""", unsafe_allow_html=True)

# Show filter impact on model training
if filter_high_mileage:
    st.info("🎯 **Model Training Note:** The linear regression model is being trained on only high-mileage cars (>25k miles), which may affect its accuracy for predicting prices of lower-mileage vehicles.")

# Linear regression model training with progress indicator
with st.spinner("🤖 Training linear regression model..."):
    models = fit_linear_regression(clean_df)

# Model Explanation Tabs
st.markdown("### 🎯 Understanding the Model")

overview_tab1, overview_tab2, overview_tab3 = st.tabs(["📚 How It Works", "📈 Visualizations & Calculator", "💰 Depreciation Patterns"])

with overview_tab1:
    st.markdown("""
    <div class="info-box">
        <h3>🎯 How Our Linear Regression Model Works</h3>
        <p>We use <strong>multiple linear regression</strong> to model the relationship between Tesla Model 3 prices and key factors:</p>
        <ul>
            <li><strong>Price = Intercept + (Age Coefficient × Age) + (Mileage Coefficient × Mileage)</strong></li>
            <li>This creates a <strong>linear plane</strong> in 3D space that best fits the data</li>
            <li>The model captures how cars depreciate with both age AND mileage simultaneously</li>
            <li>Each coefficient tells us the marginal effect of that variable holding others constant</li>
        </ul>
        <p><em>💡 In the next tab, you'll see interactive visualizations showing the regression lines, surface, and heatmaps!</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Model Components Explained</h4>
            <ul>
                <li><strong>Intercept (Base Price):</strong> Theoretical price of a brand new car with 0 miles</li>
                <li><strong>Age Coefficient:</strong> How much value is lost per year (holding mileage constant)</li>
                <li><strong>Mileage Coefficient:</strong> How much value is lost per mile (holding age constant)</li>
                <li><strong>R² Score:</strong> How well the model explains price variation (0-1 scale)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 What Makes a Good Deal?</h4>
            <p><strong>Two key metrics to consider:</strong></p>
            <ul>
                <li><strong>Value Gap:</strong> How much below predicted price (immediate savings at purchase)</li>
                <li><strong>Monthly Cost + Opp.:</strong> True economic cost including depreciation and foregone investment returns</li>
            </ul>
            <p><em>The best deals combine good initial value with low total ownership cost!</em></p>
        </div>
        """, unsafe_allow_html=True)

with overview_tab2:
    # Show the actual price relationships from the data
    st.markdown("### 📈 Real Price Relationships in Your Data")
    
    if models and 'linear' in models:
        viz_model = models['linear']['model']
        viz_coefs = models['linear']['coefs']
        
        # Create 2D scatter plots with regression lines
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📉 Price vs Age (Linear Regression)**")
            
            # Sample data for performance
            sample_df = clean_df.sample(min(500, len(clean_df))).copy()
            
            # Create scatter plot
            fig_age = px.scatter(
                sample_df,
                x="age_clean",
                y="price_clean",
                opacity=0.5,
                labels={"age_clean": "Age (Years)", "price_clean": "Price (£)"},
                title=f"Price drops £{abs(viz_coefs['age_coef']):,.0f} per year"
            )
            
            # Add regression line (holding mileage at median)
            median_mileage = clean_df["mileage_clean"].median()
            age_range = np.linspace(clean_df["age_clean"].min(), clean_df["age_clean"].max(), 100)
            pred_prices = viz_coefs['intercept'] + viz_coefs['age_coef'] * age_range + viz_coefs['mileage_coef'] * median_mileage
            
            fig_age.add_trace(go.Scatter(
                x=age_range,
                y=pred_prices,
                mode='lines',
                name=f'Regression Line (@ {median_mileage:,.0f} miles)',
                line=dict(color='red', width=3)
            ))
            
            fig_age.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            st.markdown("**📉 Price vs Mileage (Linear Regression)**")
            
            # Create scatter plot
            fig_mileage = px.scatter(
                sample_df,
                x="mileage_clean",
                y="price_clean",
                opacity=0.5,
                labels={"mileage_clean": "Mileage", "price_clean": "Price (£)"},
                title=f"Price drops £{abs(viz_coefs['mileage_coef'] * 1000):,.0f} per 1k miles"
            )
            
            # Add regression line (holding age at median)
            median_age = clean_df["age_clean"].median()
            mileage_range = np.linspace(clean_df["mileage_clean"].min(), clean_df["mileage_clean"].max(), 100)
            pred_prices = viz_coefs['intercept'] + viz_coefs['age_coef'] * median_age + viz_coefs['mileage_coef'] * mileage_range
            
            fig_mileage.add_trace(go.Scatter(
                x=mileage_range,
                y=pred_prices,
                mode='lines',
                name=f'Regression Line (@ {median_age:.0f} years old)',
                line=dict(color='red', width=3)
            ))
            
            fig_mileage.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_mileage, use_container_width=True)
        
        # 3D Visualization with Regression Surface
        st.markdown("### 🎯 3D Visualization: Regression Surface")
        
        # Create grid for surface plot
        age_grid = np.linspace(clean_df["age_clean"].min(), clean_df["age_clean"].max(), 20)
        mileage_grid = np.linspace(clean_df["mileage_clean"].min(), clean_df["mileage_clean"].max(), 20)
        age_mesh, mileage_mesh = np.meshgrid(age_grid, mileage_grid)
        
        # Calculate predicted prices for the surface
        price_mesh = viz_coefs['intercept'] + viz_coefs['age_coef'] * age_mesh + viz_coefs['mileage_coef'] * mileage_mesh
        
        # Create 3D plot
        fig_3d = go.Figure()
        
        # Add scatter points (sample for performance)
        sample_3d = clean_df.sample(min(300, len(clean_df)))
        fig_3d.add_trace(go.Scatter3d(
            x=sample_3d["age_clean"],
            y=sample_3d["mileage_clean"],
            z=sample_3d["price_clean"],
            mode='markers',
            marker=dict(size=3, color='lightblue', opacity=0.6),
            name='Actual Prices',
            hovertemplate='Age: %{x:.0f} years<br>Mileage: %{y:,.0f}<br>Price: £%{z:,.0f}<extra></extra>'
        ))
        
        # Add regression surface
        fig_3d.add_trace(go.Surface(
            x=age_grid,
            y=mileage_grid,
            z=price_mesh,
            colorscale='Reds',
            opacity=0.7,
            name='Regression Surface',
            showscale=True,
            colorbar=dict(title="Predicted<br>Price (£)")
        ))
        
        fig_3d.update_layout(
            title="Linear Regression Model: Price = f(Age, Mileage)",
            scene=dict(
                xaxis_title="Age (Years)",
                yaxis_title="Mileage",
                zaxis_title="Price (£)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Heatmap showing predicted prices
        st.markdown("### 🗺️ Price Heatmap: Explore the Model")
        
        # Create a denser grid for heatmap
        age_heatmap = np.linspace(clean_df["age_clean"].min(), clean_df["age_clean"].max(), 30)
        mileage_heatmap = np.linspace(clean_df["mileage_clean"].min(), clean_df["mileage_clean"].max(), 30)
        age_hm, mileage_hm = np.meshgrid(age_heatmap, mileage_heatmap)
        price_hm = viz_coefs['intercept'] + viz_coefs['age_coef'] * age_hm + viz_coefs['mileage_coef'] * mileage_hm
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            x=age_heatmap,
            y=mileage_heatmap,
            z=price_hm,
            colorscale='RdYlGn',
            colorbar=dict(title="Predicted<br>Price (£)"),
            hovertemplate='Age: %{x:.1f} years<br>Mileage: %{y:,.0f}<br>Predicted Price: £%{z:,.0f}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title="Predicted Price by Age and Mileage",
            xaxis_title="Age (Years)",
            yaxis_title="Mileage",
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Interactive price calculator
        st.markdown("### 🧮 Interactive Price Estimator")
        st.info("Use the sliders below to see how age and mileage affect the predicted price according to the linear regression model.")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            slider_age = st.slider(
                "Age (Years)",
                min_value=int(clean_df["age_clean"].min()),
                max_value=int(clean_df["age_clean"].max()),
                value=int(clean_df["age_clean"].median()),
                step=1
            )
        
        with col2:
            slider_mileage = st.slider(
                "Mileage",
                min_value=int(clean_df["mileage_clean"].min()),
                max_value=int(clean_df["mileage_clean"].max()),
                value=int(clean_df["mileage_clean"].median()),
                step=1000
            )
        
        with col3:
            predicted_price_slider = viz_coefs['intercept'] + viz_coefs['age_coef'] * slider_age + viz_coefs['mileage_coef'] * slider_mileage
            st.markdown(f"""
            <div class="metric-card" style="margin-top: 1rem;">
                <h4>💰 Predicted Price</h4>
                <h2>£{predicted_price_slider:,.0f}</h2>
                <p>{slider_age} years, {slider_mileage:,} miles</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.error("❌ Could not fit model for visualization. Please check your data.")

with overview_tab3:
    st.markdown("### 💰 Depreciation Patterns & Insights")
    
    # Calculate and show depreciation patterns
    if models and 'linear' in models:
        demo_coefs = models['linear']['coefs']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📉 Depreciation Rate</h4>
                <p><strong>Per Year:</strong> £{abs(demo_coefs['age_coef']):,.0f}</p>
                <p><strong>Per 1,000 Miles:</strong> £{abs(demo_coefs['mileage_coef'] * 1000):,.0f}</p>
                <p><strong>Base Price:</strong> £{demo_coefs['intercept']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💡 Example Calculation</h4>
                <p>A 4-year-old Model 3 with 50,000 miles:</p>
                <p>Predicted Price: £{demo_coefs['intercept'] + demo_coefs['age_coef'] * 4 + demo_coefs['mileage_coef'] * 50000:,.0f}</p>
                <p>After 2 years + 10,000 miles:</p>
                <p>New Price: £{demo_coefs['intercept'] + demo_coefs['age_coef'] * 6 + demo_coefs['mileage_coef'] * 60000:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Could not fit demonstration model - not enough data")
    
    # Show depreciation by age
    st.markdown("### 📊 Depreciation by Age")
    fig_depr = px.box(
        clean_df, 
        x="age_clean", 
        y="price_clean", 
        title="Price Distribution by Age"
    )
    st.plotly_chart(fig_depr, use_container_width=True)
    
    # Depreciation curve visualization
    total_miles = miles_per_year * hold_years
    st.markdown(f"### 📉 Future Depreciation Projection")
    st.info(f"Estimating value loss over {hold_years} years with {miles_per_year:,} miles/year ({total_miles:,} total miles)")
    depr_fig = create_depreciation_visualization(clean_df, miles_per_year, hold_years)
    st.plotly_chart(depr_fig, use_container_width=True)

# Model Performance Metrics
st.markdown("### 📊 Model Performance Metrics")

if models and 'linear' in models:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        r2 = models['linear']['r2']
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Model R²</h4>
            <h2>{r2:.3f}</h2>
            <p>Explains {r2*100:.1f}% of price variation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mae = models['linear']['mae']
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 Mean Absolute Error</h4>
            <h2>£{mae:,.0f}</h2>
            <p>Average prediction error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        rmse = models['linear']['rmse']
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Root Mean Square Error</h4>
            <h2>£{rmse:,.0f}</h2>
            <p>Standard deviation of errors</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cv_score = models['linear']['cv_scores'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔄 Cross-Validation R²</h4>
            <h2>{cv_score:.3f}</h2>
            <p>Robustness score (5-fold CV)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show model coefficients
    coefs = models['linear']['coefs']
    st.markdown("#### 📊 Model Coefficients")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Base Price (Intercept)</h4>
            <h2>£{coefs['intercept']:,.0f}</h2>
            <p>Starting price for a new car</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Age Coefficient</h4>
            <h2>£{coefs['age_coef']:,.0f}</h2>
            <p>Price change per year of age</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🚗 Mileage Coefficient</h4>
            <h2>£{coefs['mileage_coef']:.2f}</h2>
            <p>Price change per mile</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional diagnostic visualizations
    st.markdown("### 📈 Additional Diagnostic Plots")
    
    col1, col2 = st.columns(2)
    with col1:
        # Scatter: Price vs Age colored by Mileage
        fig_scatter_age = px.scatter(
            clean_df,
            x="age_clean",
            y="price_clean",
            color="mileage_clean",
            hover_data=["mileage_clean"],
            labels={"age_clean": "Age (Years)", "price_clean": "Price (£)", "mileage_clean": "Mileage"},
            title="Price vs Age (colored by mileage)"
        )
        st.plotly_chart(fig_scatter_age, use_container_width=True)
    
    with col2:
        # Scatter: Price vs Mileage colored by Age
        fig_scatter_mileage = px.scatter(
            clean_df,
            x="mileage_clean",
            y="price_clean",
            color="age_clean",
            hover_data=["age_clean"],
            labels={"mileage_clean": "Mileage", "price_clean": "Price (£)", "age_clean": "Age (Years)"},
            title="Price vs Mileage (colored by age)"
        )
        st.plotly_chart(fig_scatter_mileage, use_container_width=True)

else:
    st.error("❌ Model training failed. Please check your data.")

# ============================================================================
# SECTION 3: 💎 BEST DEAL FINDER
# ============================================================================

st.markdown("""
<div class="section-header">
    <h2>💎 Section 3: Best Deal Finder</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>This section ranks all listings to identify the best deals based on value, depreciation, and other factors. 
    Use the sidebar to choose your ranking method and set your budget.</p>
</div>
""", unsafe_allow_html=True)

# Initialize prediction dataframe
pred_df = clean_df.copy()

# Use linear regression model for predictions
if models and 'linear' in models:
    model = models['linear']['model']
    X = pred_df[["mileage_clean", "age_clean"]].astype(float).values
    pred_df["predicted_price"] = model.predict(X)
    pred_df["value_gap"] = pred_df["predicted_price"] - pred_df["price_clean"]
    pred_df["value_pct"] = pred_df["value_gap"] / pred_df["predicted_price"]
    
    st.success(f"✅ Using Linear Regression model for predictions (R² = {models['linear']['r2']:.3f})")
else:
    st.error("❌ No model available for predictions")
    st.stop()

# Unified deal scoring & reasoning
if models and 'linear' in models:
    depreciation_model = models['linear']['model']
else:
    st.error("❌ No model available for depreciation calculations")
    st.stop()

# Compute deal metrics (depreciation and opportunity costs)
scored_df = compute_deal_metrics(pred_df, depreciation_model, miles_per_year, hold_years)
scored_df["reasoning"] = scored_df.apply(lambda r: build_reasoning(r, scored_df), axis=1)

# Apply price range filter to results (after model training and predictions)
scored_df = scored_df[
    (scored_df["price_clean"] >= min_budget) & 
    (scored_df["price_clean"] <= max_budget)
]

st.success(f"✅ Showing {len(scored_df)} cars within £{min_budget:,} - £{max_budget:,} budget range")

st.markdown("### 🎯 Ranking Options")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>💰 Lowest Total Monthly Cost</h4>
        <p>Sort by monthly cost including opportunity cost</p>
        <p><strong>Best for:</strong> Finding the most economical cars to own</p>
        <p><strong>Considers:</strong> Depreciation + foregone investment returns (5%)</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🏷️ Best Initial Value</h4>
        <p>Sort by biggest discount vs predicted price</p>
        <p><strong>Best for:</strong> Finding the best bargains at purchase</p>
        <p><strong>Focuses on:</strong> Immediate savings vs market value</p>
    </div>
    """, unsafe_allow_html=True)

# Apply ranking based on user selection
if 'ranking_method' in locals():
    if ranking_method == "Best Initial Value":
        view_df = scored_df.sort_values("value_gap", ascending=False).copy()
        st.info("🏆 **Active Ranking:** Best Initial Value (Biggest Discount vs Predicted Price)")
    else:  # Default: Lowest Total Monthly Cost
        view_df = scored_df.sort_values("monthly_cost_with_opportunity", ascending=True).copy()
        st.info("🏆 **Active Ranking:** Lowest Total Monthly Cost (Most Economical to Own)")
else:
    view_df = scored_df.sort_values("monthly_cost_with_opportunity", ascending=True).copy()

st.markdown("### 📊 Top Deals")

st.markdown("""
<div class="info-box">
    <p><strong>Understanding the Columns:</strong></p>
    <ul>
        <li><strong>Price:</strong> Current listing price</li>
        <li><strong>Predicted:</strong> What the model thinks it should cost based on age and mileage</li>
        <li><strong>Value Gap:</strong> How much below predicted (£) - positive = good deal, negative = overpriced</li>
        <li><strong>Value %:</strong> Discount percentage vs prediction</li>
        <li><strong>Total Cost:</strong> Total depreciation over {hold_years} years (purchase price - predicted resale value)</li>
        <li><strong>Annual Cost:</strong> Yearly cost of ownership (depreciation)</li>
        <li><strong>Monthly Cost:</strong> Monthly depreciation cost</li>
        <li><strong>Monthly Cost + Opp.:</strong> <strong>True economic cost</strong> - includes depreciation + opportunity cost (5% foregone investment returns)</li>
    </ul>
    <p><em>💡 Use "Monthly Cost + Opp." to compare the real economics of different cars. Lower-priced cars naturally have lower opportunity costs.</em></p>
</div>
""".format(hold_years=hold_years), unsafe_allow_html=True)

# Image column config (if available)
column_config = {}
if "main_image" in view_df.columns:
    column_config["main_image"] = st.column_config.ImageColumn("Photo", help="Listing main image", width="small")

# Enhanced display columns including both value and depreciation metrics
disp_cols = [
    "main_image", "price_clean", "predicted_price", "value_gap", "value_pct",
    "depreciation_est", "annual_depr_est", "monthly_cost_ownership", "monthly_cost_with_opportunity",
    "year_clean", "age_clean", "mileage_clean", "dealer_rating", "location", "reasoning", "url"
]

present_cols = [c for c in disp_cols if c in view_df.columns]

# Format the dataframe for better display
display_df = view_df[present_cols].copy()
if 'value_pct' in display_df.columns:
    display_df['value_pct'] = (display_df['value_pct'] * 100).round(1).astype(str) + '%'
if 'annual_depr_est' in display_df.columns:
    display_df['annual_depr_est'] = display_df['annual_depr_est'].round(0).astype(int)
if 'depreciation_est' in display_df.columns:
    display_df['depreciation_est'] = display_df['depreciation_est'].round(0).astype(int)
if 'monthly_cost_ownership' in display_df.columns:
    display_df['monthly_cost_ownership'] = display_df['monthly_cost_ownership'].round(0).astype(int)
if 'monthly_cost_with_opportunity' in display_df.columns:
    display_df['monthly_cost_with_opportunity'] = display_df['monthly_cost_with_opportunity'].round(0).astype(int)

# Add column configuration for better formatting
column_config = {}
if "main_image" in display_df.columns:
    column_config["main_image"] = st.column_config.ImageColumn("Photo", help="Listing main image", width="small")
if "price_clean" in display_df.columns:
    column_config["price_clean"] = st.column_config.NumberColumn("Price", help="Current listing price", format="£%d")
if "predicted_price" in display_df.columns:
    column_config["predicted_price"] = st.column_config.NumberColumn("Predicted", help="Model predicted price", format="£%d")
if "value_gap" in display_df.columns:
    column_config["value_gap"] = st.column_config.NumberColumn("Value Gap", help="How much below predicted price", format="£%d")
if "depreciation_est" in display_df.columns:
    column_config["depreciation_est"] = st.column_config.NumberColumn("Total Cost", help="Total cost of ownership over hold period", format="£%d")
if "annual_depr_est" in display_df.columns:
    column_config["annual_depr_est"] = st.column_config.NumberColumn("Annual Cost", help="Estimated annual cost of ownership", format="£%d")
if "monthly_cost_ownership" in display_df.columns:
    column_config["monthly_cost_ownership"] = st.column_config.NumberColumn("Monthly Cost", help="Monthly cost (depreciation only)", format="£%d")
if "monthly_cost_with_opportunity" in display_df.columns:
    column_config["monthly_cost_with_opportunity"] = st.column_config.NumberColumn("Monthly Cost + Opp.", help="Monthly cost including 5% opportunity cost on capital", format="£%d")
if "year_clean" in display_df.columns:
    column_config["year_clean"] = st.column_config.NumberColumn("Year", help="Registration year", format="%d")
if "age_clean" in display_df.columns:
    column_config["age_clean"] = st.column_config.NumberColumn("Age", help="Age in years", format="%d years")
if "mileage_clean" in display_df.columns:
    column_config["mileage_clean"] = st.column_config.NumberColumn("Mileage", help="Odometer mileage", format="%d")
if "value_pct" in display_df.columns:
    column_config["value_pct"] = st.column_config.TextColumn("Value Discount", help="Discount vs predicted price (%)")

st.dataframe(display_df.head(50), use_container_width=True, column_config=column_config)

csv_bytes = view_df[present_cols].to_csv(index=False).encode("utf-8")
st.download_button("📥 Download Best Deals CSV", data=csv_bytes, file_name="model3_best_deals.csv", mime="text/csv", help="Download the top deals for further analysis")

# ============================================================================
# SECTION 4: 🔍 INDIVIDUAL CAR ANALYSIS
# ============================================================================

st.markdown("""
<div class="section-header">
    <h2>🔍 Section 4: Individual Car Analysis</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>Have a specific Tesla Model 3 in mind? Enter its details below to see how it compares to the market 
    and get personalized depreciation estimates based on your usage patterns.</p>
</div>
""", unsafe_allow_html=True)

# Create input form
with st.form("car_comparison_form"):
    st.markdown("### 📝 Enter Car Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        input_year = st.number_input(
            "Registration Year",
            min_value=2017,
            max_value=2025,
            value=2022,
            step=1,
            help="Year the car was first registered"
        )
    
    with col2:
        input_mileage = st.number_input(
            "Mileage",
            min_value=0,
            max_value=500000,
            value=30000,
            step=1000,
            help="Current mileage in miles"
        )
    
    with col3:
        input_price = st.number_input(
            "Price (£)",
            min_value=1000,
            max_value=100000,
            value=25000,
            step=500,
            help="Asking price in pounds"
        )
    
    
    submitted = st.form_submit_button("🔍 Analyze This Car", type="primary")

# Process the comparison if form is submitted
if submitted and models and 'linear' in models:
    st.markdown("---")
    st.markdown("### 📊 Analysis Results")
    
    # Calculate age
    current_year = 2025
    input_age = current_year - input_year
    
    # Create a dataframe row for the input car
    input_car = pd.DataFrame({
        'year_clean': [input_year],
        'age_clean': [input_age],
        'mileage_clean': [input_mileage],
        'price_clean': [input_price]
    })
    
    # Get the trained model
    model = models['linear']['model']
    
    # Make prediction
    X_input = input_car[["mileage_clean", "age_clean"]].astype(float).values
    predicted_price = model.predict(X_input)[0]
    value_gap = predicted_price - input_price
    value_pct = value_gap / predicted_price if predicted_price > 0 else 0
    
    # Calculate depreciation estimates
    input_depr = estimate_depreciation(input_car, model, miles_per_year, hold_years)
    annual_depr = input_depr["annual_depr_est"].iloc[0]
    total_depr = input_depr["depreciation_est"].iloc[0]
    future_value = input_depr["pred_future_model"].iloc[0]
    monthly_cost = total_depr / (hold_years * 12)
    
    # Calculate opportunity cost
    annual_opportunity = input_price * 0.05
    monthly_opportunity = annual_opportunity / 12
    monthly_cost_with_opp = monthly_cost + monthly_opportunity
    
    # Display results in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Value Assessment</h4>
            <h2>{'🟢 Good Deal' if value_gap > 0 else '🔴 Overpriced'}</h2>
            <p>£{abs(value_gap):,.0f} {'below' if value_gap > 0 else 'above'} market</p>
            <p>({abs(value_pct)*100:.1f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Market Prediction</h4>
            <h2>£{predicted_price:,.0f}</h2>
            <p>vs £{input_price:,.0f} asking</p>
            <p>Model R²: {models['linear']['r2']:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Cost of Ownership</h4>
            <h2>£{monthly_cost_with_opp:,.0f}/mo</h2>
            <p>Depreciation: £{monthly_cost:,.0f}/mo</p>
            <p>+ Opportunity: £{monthly_opportunity:,.0f}/mo</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔮 Future Value</h4>
            <h2>£{future_value:,.0f}</h2>
            <p>after {hold_years} years</p>
            <p>({miles_per_year:,} mi/year)</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed analysis
    st.markdown("### 📋 Detailed Analysis")
    
    # Value comparison
    st.markdown("#### 💰 Value Comparison")
    if value_gap > 0:
        st.success(f"✅ **Good Deal!** This car is priced £{value_gap:,.0f} below what the market model predicts (£{predicted_price:,.0f} vs £{input_price:,.0f}). This represents a {value_pct*100:.1f}% discount.")
    else:
        st.warning(f"⚠️ **Overpriced** This car is priced £{abs(value_gap):,.0f} above what the market model predicts (£{predicted_price:,.0f} vs £{input_price:,.0f}). This represents a {abs(value_pct)*100:.1f}% premium.")
    
    # Depreciation analysis
    st.markdown("#### 📉 Cost of Ownership Analysis")
    st.info(f"""
    **Ownership Cost Breakdown:**
    - **Purchase Price:** £{input_price:,.0f}
    - **Predicted Resale Value:** £{future_value:,.0f} (after {hold_years} years)
    - **Total Depreciation:** £{total_depr:,.0f}
    - **Annual Cost (Depreciation):** £{annual_depr:,.0f}
    - **Monthly Cost (Depreciation):** £{monthly_cost:,.0f}
    
    **Including Opportunity Cost (5% foregone returns):**
    - **Monthly Opportunity Cost:** £{monthly_opportunity:,.0f} (what you could earn investing £{input_price:,.0f} at 5%/year)
    - **Total Monthly Cost:** £{monthly_cost_with_opp:,.0f} (depreciation + opportunity cost)
    
    *The opportunity cost represents the investment returns you're giving up by having capital tied up in a depreciating asset instead of earning 5% annual returns.*
    """)
    
    # Comparison to dataset
    st.markdown("#### 📊 How This Car Compares to the Market")
    
    # Find similar cars in the dataset
    similar_cars = clean_df[
        (clean_df["age_clean"] >= input_age - 1) & 
        (clean_df["age_clean"] <= input_age + 1) &
        (clean_df["mileage_clean"] >= input_mileage * 0.8) &
        (clean_df["mileage_clean"] <= input_mileage * 1.2)
    ]
    
    if len(similar_cars) > 0:
        similar_avg_price = similar_cars["price_clean"].mean()
        similar_median_price = similar_cars["price_clean"].median()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Similar Cars (n={len(similar_cars)})</h4>
                <p><strong>Average Price:</strong> £{similar_avg_price:,.0f}</p>
                <p><strong>Median Price:</strong> £{similar_median_price:,.0f}</p>
                <p><strong>Your Car:</strong> £{input_price:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vs_avg = input_price - similar_avg_price
            vs_median = input_price - similar_median_price
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 Price Comparison</h4>
                <p><strong>vs Average:</strong> £{vs_avg:,.0f} ({'below' if vs_avg < 0 else 'above'})</p>
                <p><strong>vs Median:</strong> £{vs_median:,.0f} ({'below' if vs_median < 0 else 'above'})</p>
                <p><strong>Percentile:</strong> {percentile_rank(similar_cars["price_clean"], input_price):.0f}th</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No similar cars found in the dataset for direct comparison.")
    
    # Visualization
    st.markdown("#### 📈 Market Position Visualization")
    
    # Create a scatter plot showing where this car fits
    fig_comparison = go.Figure()
    
    # Split market data into same-age vs other ages
    age_series = clean_df["age_clean"].round(0)
    same_age_mask = age_series == input_age
    market_same_age = clean_df.loc[same_age_mask]
    market_other_age = clean_df.loc[~same_age_mask]
    
    # Add market cars with the same age
    if len(market_same_age) > 0:
        fig_comparison.add_trace(go.Scatter(
            x=market_same_age["mileage_clean"],
            y=market_same_age["price_clean"],
            mode='markers',
            marker=dict(color='orange', size=7, opacity=0.8),
            name=f"Market (Age {input_age} yrs)",
            hovertemplate='Mileage: %{x:,.0f}<br>Price: £%{y:,.0f}<extra></extra>'
        ))
    
    # Add other market cars
    if len(market_other_age) > 0:
        fig_comparison.add_trace(go.Scatter(
            x=market_other_age["mileage_clean"],
            y=market_other_age["price_clean"],
            mode='markers',
            marker=dict(color='lightblue', size=6, opacity=0.45),
            name='Market (Other Ages)',
            hovertemplate='Mileage: %{x:,.0f}<br>Price: £%{y:,.0f}<extra></extra>'
        ))
    
    # Add the input car
    fig_comparison.add_trace(go.Scatter(
        x=[input_mileage],
        y=[input_price],
        mode='markers',
        marker=dict(color='red', size=14, symbol='star'),
        name='Your Car',
        hovertemplate=f'Your Car<br>Mileage: {input_mileage:,.0f}<br>Price: £{input_price:,.0f}<extra></extra>'
    ))
    
    # Add predicted price point
    fig_comparison.add_trace(go.Scatter(
        x=[input_mileage],
        y=[predicted_price],
        mode='markers',
        marker=dict(color='green', size=12, symbol='diamond'),
        name='Model Prediction',
        hovertemplate=f'Model Prediction<br>Mileage: {input_mileage:,.0f}<br>Predicted: £{predicted_price:,.0f}<extra></extra>'
    ))
    
    fig_comparison.update_layout(
        title=f"Your Car vs Market (Age: {input_age} years)",
        xaxis_title="Mileage",
        yaxis_title="Price (£)",
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Recommendations
    st.markdown("#### 💡 Recommendations")
    
    if value_gap > 1000:
        st.success("🎉 **Excellent Deal!** This car is significantly underpriced. Consider moving quickly if the condition is good.")
    elif value_gap > 0:
        st.info("👍 **Good Value** This car offers fair value compared to the market. Worth considering if it meets your needs.")
    elif value_gap > -1000:
        st.warning("⚠️ **Fair Price** This car is slightly overpriced but within reasonable range. You might want to negotiate.")
    else:
        st.error("❌ **Overpriced** This car is significantly overpriced. Consider looking for alternatives or negotiating hard.")
    
    # Cost of ownership advice (using total cost including opportunity)
    if monthly_cost_with_opp < 250:  # Low total monthly cost
        st.success(f"💰 **Low Total Cost of Ownership** At £{monthly_cost_with_opp:,.0f}/month (including opportunity cost), this is an economical choice. Depreciation: £{monthly_cost:,.0f}/mo + Opportunity: £{monthly_opportunity:,.0f}/mo.")
    elif monthly_cost_with_opp < 400:  # Moderate total monthly cost
        st.info(f"📊 **Moderate Total Cost of Ownership** At £{monthly_cost_with_opp:,.0f}/month (including opportunity cost), this is in line with typical Tesla Model 3 ownership costs. Depreciation: £{monthly_cost:,.0f}/mo + Opportunity: £{monthly_opportunity:,.0f}/mo.")
    else:
        st.warning(f"📉 **High Total Cost of Ownership** At £{monthly_cost_with_opp:,.0f}/month (including opportunity cost), this is expensive to own. Depreciation: £{monthly_cost:,.0f}/mo + Opportunity: £{monthly_opportunity:,.0f}/mo. Consider if this fits your budget.")

elif submitted:
    st.error("❌ Please ensure the model has been trained successfully before analyzing individual cars.")
