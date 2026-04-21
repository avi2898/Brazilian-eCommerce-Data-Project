"""
question1.py
------------
## Research Question 1 - Logistics Efficiency & Residual Analysis

How efficiently does the logistics network convert delivery distance into actual
delivery time, and which regions overperform or underperform relative to what
distance alone would predict?

Analytical framing:
- What does distance predict about actual delivery lead time?
- Which regions consistently deliver faster or slower than expected?
- Where are the operational bottlenecks and efficiency pockets?
- Which states should be prioritised for logistics intervention?

Output:
  outputs/question_1/geographic_delivery_map_and_regression.png

Methodology Note:
  This analysis measures logistics efficiency using actual lead time
  (order_delivered_customer_date - order_purchase_timestamp) rather than
  promised delivery dates. Estimated delivery dates are excluded because they
  likely contain inflated buffers and create misleading ceiling effects that
  obscure true operational performance.

  A linear regression is fit between log(distance + 1) and actual lead time.
  The residual — actual minus predicted lead time — serves as a region-level
  efficiency signal: positive residuals indicate operational friction (slower
  than distance would predict), while negative residuals indicate efficient
  logistics execution.

  Distance explains part of delivery time, but regional residual patterns
  reveal bottlenecks and efficiencies that geography alone cannot explain.
  Positive residual clusters indicate operational friction such as weak carrier
  performance, congestion, or fulfillment delays. Negative residual clusters
  indicate highly efficient logistics networks.
"""

from pathlib import Path
import urllib.request
import zipfile
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy import stats
import geopandas as gpd

from src.db_setup import query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
OUTPUT_DIR     = PROJECT_ROOT / "outputs" / "question_1"
OUTPUT_FILE    = OUTPUT_DIR / "geographic_delivery_map_and_regression.png"
GEO_CACHE_DIR  = PROJECT_ROOT / "data" / "naturalearth"
GEO_CACHE_FILE = GEO_CACHE_DIR / "ne_110m_admin_0_countries.shp"
GEO_URL        = (
    "https://naturalearth.s3.amazonaws.com/110m_cultural/"
    "ne_110m_admin_0_countries.zip"
)

# Theme colors
BG       = "#f8fafc"
LAND     = "#e2e8f0"
NEIGHBOR = "#cbd5e1"
EDGE     = "#94a3b8"
TEXT     = "#0f172a"

# Volume threshold for state-level prioritization
MIN_STATE_ORDERS = 100


# ---------------------------------------------------------------------------
# 0. World geodata — download once, cache locally
# ---------------------------------------------------------------------------
def get_world_geodata() -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of world country polygons.
    Downloads from Natural Earth on first run and caches to data/naturalearth/.
    """
    if not GEO_CACHE_FILE.exists():
        print(f"  Downloading Natural Earth shapefile from:\n  {GEO_URL}")
        GEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GEO_URL) as response:
            zip_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(GEO_CACHE_DIR)
        print(f"  Saved to: {GEO_CACHE_DIR}")
    return gpd.read_file(GEO_CACHE_FILE)


# ---------------------------------------------------------------------------
# 1. Load orders — actual lead time only (no estimated delivery date)
# ---------------------------------------------------------------------------
def load_orders() -> pd.DataFrame:
    """
    Load orders with purchase and actual delivery timestamps.
    Estimated delivery date is intentionally excluded.
    """
    sql = """
    SELECT
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date
    FROM orders o
    WHERE o.order_purchase_timestamp      IS NOT NULL
      AND o.order_delivered_customer_date IS NOT NULL
    """
    df = query(sql)
    df["order_purchase_timestamp"]      = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    return df


# ---------------------------------------------------------------------------
# 2. Compute actual lead time
# ---------------------------------------------------------------------------
def compute_lead_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add actual_lead_time_days = delivered - purchased (fractional days).
    Drop rows with missing or non-positive values.
    """
    df["actual_lead_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    df = df.dropna(subset=["actual_lead_time_days"])
    df = df[df["actual_lead_time_days"] > 0].copy()
    return df


# ---------------------------------------------------------------------------
# 3. Join customers
# ---------------------------------------------------------------------------
def join_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Attach customer zip prefix, city, and state."""
    sql = """
    SELECT
        customer_id,
        customer_zip_code_prefix,
        customer_city,
        customer_state
    FROM customers
    """
    customers = query(sql)
    return df.merge(customers, on="customer_id", how="inner")


# ---------------------------------------------------------------------------
# 4. Load geolocation averages
# ---------------------------------------------------------------------------
def load_geo() -> pd.DataFrame:
    """Average lat/lng per zip prefix."""
    sql = """
    SELECT
        geolocation_zip_code_prefix AS zip_prefix,
        AVG(geolocation_lat)        AS avg_lat,
        AVG(geolocation_lng)        AS avg_lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
    """
    return query(sql)


# ---------------------------------------------------------------------------
# 5. Haversine distance (vectorised)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0  # mean Earth radius used in Haversine formula


def haversine_km(lat1: np.ndarray, lng1: np.ndarray,
                 lat2: np.ndarray, lng2: np.ndarray) -> np.ndarray:
    """Return great-circle distance in km between two sets of coordinates."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# 6. Build order-level regression dataset with residuals
# ---------------------------------------------------------------------------
def build_regression_dataset(df: pd.DataFrame, geo: pd.DataFrame):
    """
    For each order compute seller-to-customer distance (km), actual lead time,
    then fit a global regression and compute residuals.

    Returns:
      df_reg   – order-level DataFrame with residual_days, log_distance, etc.
      slope    – regression slope
      intercept – regression intercept
    """
    sql_sellers = """
    SELECT
        oi.order_id,
        s.seller_zip_code_prefix
    FROM order_items oi
    JOIN sellers s ON oi.seller_id = s.seller_id
    """
    order_sellers = query(sql_sellers)

    seller_geo = geo.rename(columns={
        "zip_prefix": "seller_zip_code_prefix",
        "avg_lat":    "seller_lat",
        "avg_lng":    "seller_lng",
    })
    order_sellers = order_sellers.merge(seller_geo, on="seller_zip_code_prefix", how="inner")

    order_customers = (
        df[["order_id", "customer_zip_code_prefix", "customer_state",
            "actual_lead_time_days"]]
        .merge(
            geo.rename(columns={
                "zip_prefix": "customer_zip_code_prefix",
                "avg_lat":    "customer_lat",
                "avg_lng":    "customer_lng",
            }),
            on="customer_zip_code_prefix",
            how="inner",
        )
    )

    pairs = order_customers.merge(order_sellers, on="order_id", how="inner")
    pairs["distance_km"] = haversine_km(
        pairs["customer_lat"].values, pairs["customer_lng"].values,
        pairs["seller_lat"].values,   pairs["seller_lng"].values,
    )

    reg = (
        pairs.groupby("order_id")
        .agg(
            avg_distance_km          = ("distance_km",              "mean"),
            actual_lead_time_days    = ("actual_lead_time_days",    "first"),
            customer_state           = ("customer_state",            "first"),
            customer_zip_code_prefix = ("customer_zip_code_prefix", "first"),
        )
        .reset_index()
    )

    reg = reg.dropna(subset=["avg_distance_km", "actual_lead_time_days"])
    reg = reg[reg["avg_distance_km"] > 0].copy()

    # Fit global regression: log(distance + 1) → actual lead time
    # Distances are highly right-skewed.
    # log1p compresses long tails and models diminishing marginal impact of distance.
    reg["log_distance"] = np.log1p(reg["avg_distance_km"])
    x = reg["log_distance"].values
    y = reg["actual_lead_time_days"].values

    m, b = np.polyfit(x, y, 1)
    reg["predicted_days"] = m * x + b

    # Positive residual: slower than expected for given distance
    # Negative residual: faster than expected
    # Residuals proxy operational efficiency after controlling for geography
    reg["residual_days"]  = y - reg["predicted_days"].values

    return reg, float(m), float(b)


# ---------------------------------------------------------------------------
# 7. Model diagnostics — residual summary
# ---------------------------------------------------------------------------
def run_diagnostics(df_reg: pd.DataFrame) -> dict:
    """
    Print residual summary statistics and an informal heteroscedasticity check.

    Reports mean and standard deviation of residuals, and the Pearson correlation
    between fitted values and squared residuals. A strong positive correlation
    indicates that variance grows with distance — deliveries to far regions are
    not only slower but also less predictable.

    Returns
    -------
    dict with keys:
      mean_residual        – mean of residuals (should be ~0 for unbiased model)
      std_residual         – standard deviation of residuals
      corr_fitted_sq_resid – Pearson r between fitted values and squared residuals
    """
    res    = df_reg["residual_days"].values
    fitted = df_reg["predicted_days"].values

    print("\n  Residual diagnostics:")
    print(f"    Mean residual:  {res.mean():.4f} days  (should be ~0)")
    print(f"    Std  residual:  {res.std():.4f} days")

    # Pearson correlation between fitted values and squared residuals.
    # A positive correlation signals heteroscedasticity: variance rises with distance.
    corr_fs, _ = stats.pearsonr(fitted, res ** 2)
    print(f"    Corr(fitted, residual²): {corr_fs:.4f}  "
          f"{'— suggests heteroscedasticity' if abs(corr_fs) > 0.05 else ''}")

    return {
        "mean_residual":        float(res.mean()),
        "std_residual":         float(res.std()),
        "corr_fitted_sq_resid": float(corr_fs),
    }


# ---------------------------------------------------------------------------
# 8. Build geographic efficiency dataset (zip-prefix level residuals)
# ---------------------------------------------------------------------------
def build_geo_dataset(df_reg: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean residual days per customer zip prefix.
    Only prefixes with >= 10 orders are retained.
    """
    agg = (
        df_reg.groupby("customer_zip_code_prefix")
        .agg(
            mean_residual  = ("residual_days",  "mean"),
            num_orders     = ("order_id",        "count"),
            customer_state = ("customer_state",  "first"),
        )
        .reset_index()
        .rename(columns={"customer_zip_code_prefix": "zip_prefix"})
    )

    agg = agg[agg["num_orders"] >= 10]
    agg = agg.merge(geo, on="zip_prefix", how="inner")
    agg = agg.dropna(subset=["avg_lat", "avg_lng"])
    return agg


# ---------------------------------------------------------------------------
# 9. Impact Score Ranking — state-level bottleneck prioritisation
# ---------------------------------------------------------------------------
def run_prioritization(df_geo: pd.DataFrame) -> None:
    """
    Rank states by operational impact score = mean_residual × log1p(total_orders).
    High residual + high volume = greatest intervention opportunity.

    Impact Score Ranking:
    Residual inefficiency alone is not enough for prioritisation.
    We rank regions using both residual severity and order volume
    to identify bottlenecks with the greatest operational impact.
    """
    state_agg = (
        df_geo.groupby("customer_state")
        .agg(
            mean_residual_days = ("mean_residual", "mean"),
            total_orders       = ("num_orders",    "sum"),
        )
        .reset_index()
        .query(f"total_orders >= {MIN_STATE_ORDERS}")
    )

    # Impact score weights residual by log order volume
    # Combines severity (delay inefficiency) with scale (affected customers)
    # Helps prioritize operational interventions with highest business impact
    state_agg["impact_score"] = (
        state_agg["mean_residual_days"] * np.log1p(state_agg["total_orders"])
    )

    vol_median = state_agg["total_orders"].median()

    def _action(row: pd.Series) -> str:
        if row["mean_residual_days"] > 0 and row["total_orders"] >= vol_median:
            return "High-priority carrier / fulfillment review"
        elif row["mean_residual_days"] > 0:
            return "Investigate regional bottlenecks"
        else:
            return "Monitor — currently efficient"

    state_agg["suggested_action"] = state_agg.apply(_action, axis=1)

    # Top 5 bottleneck states (positive residual only)
    bottlenecks = (
        state_agg[state_agg["mean_residual_days"] > 0]
        .nlargest(5, "impact_score")
        .reset_index(drop=True)
    )

    print("\n  Top 5 bottleneck states (ranked by impact score):")
    print(f"  {'#':<3} {'State':<6} {'Residual (d)':>13} {'Orders':>8} "
          f"{'Impact':>8}  Suggested Action")
    print("  " + "-" * 75)
    for i, row in bottlenecks.iterrows():
        print(f"  {i+1:<3} {row['customer_state']:<6} "
              f"{row['mean_residual_days']:>12.2f}  "
              f"{int(row['total_orders']):>8,}  "
              f"{row['impact_score']:>7.2f}  "
              f"{row['suggested_action']}")

    # Top 5 most efficient states (negative residual)
    efficient = (
        state_agg[state_agg["mean_residual_days"] < 0]
        .nsmallest(5, "mean_residual_days")
        .reset_index(drop=True)
    )

    print("\n  Top 5 most efficient states (benchmark / best practice):")
    print(f"  {'#':<3} {'State':<6} {'Residual (d)':>13} {'Orders':>8}")
    print("  " + "-" * 35)
    for i, row in efficient.iterrows():
        print(f"  {i+1:<3} {row['customer_state']:<6} "
              f"{row['mean_residual_days']:>12.2f}  "
              f"{int(row['total_orders']):>8,}")

    return bottlenecks, efficient


# ---------------------------------------------------------------------------
# 10. Two-panel figure
# ---------------------------------------------------------------------------
def build_map_and_regression(
    df_geo: pd.DataFrame,
    df_reg: pd.DataFrame,
    slope: float,
    intercept: float,
) -> None:
    """
    Left panel  – Geographic efficiency map (mean residual by zip prefix).
                  Blue = faster than expected, Red = slower than expected.
    Right panel – Scatter of log(distance) vs actual lead time, points colored
                  by residual, with regression trendline.
    """
    fig, (ax_map, ax_reg) = plt.subplots(
        1, 2,
        figsize=(18, 9),
        gridspec_kw={"width_ratios": [1.25, 1]},
    )
    fig.patch.set_facecolor(BG)

    # Shared diverging norm centered on zero residual
    res_abs     = np.percentile(np.abs(df_geo["mean_residual"]), 95)
    shared_norm = mcolors.TwoSlopeNorm(vmin=-res_abs, vcenter=0, vmax=res_abs)
    div_cmap    = plt.get_cmap("RdBu_r")   # blue = efficient, red = bottleneck

    # ------------------------------------------------------------------ #
    # LEFT PANEL — Residual efficiency map                                 #
    # ------------------------------------------------------------------ #
    ax_map.set_facecolor(BG)

    world     = get_world_geodata()
    brazil    = world[world["NAME"] == "Brazil"]
    neighbors = world[
        world.geometry.intersects(brazil.geometry.union_all())
        & (world["NAME"] != "Brazil")
    ]
    neighbors.plot(ax=ax_map, color=NEIGHBOR, edgecolor=EDGE, linewidth=0.4)
    brazil.plot(ax=ax_map,    color=LAND,     edgecolor=EDGE, linewidth=0.8)

    ax_map.set_xlim(-75, -28)
    ax_map.set_ylim(-35,   6)

    sizes = (df_geo["num_orders"] / df_geo["num_orders"].max() * 80).clip(lower=4)

    ax_map.scatter(
        df_geo["avg_lng"], df_geo["avg_lat"],
        c=df_geo["mean_residual"],
        cmap=div_cmap, norm=shared_norm,
        s=sizes, alpha=0.82, linewidths=0, zorder=3,
    )

    cbar = plt.colorbar(
        ScalarMappable(norm=shared_norm, cmap=div_cmap),
        ax=ax_map, fraction=0.025, pad=0.02,
    )
    cbar.set_label("Residual Lead Time (days)", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    # Annotate top-8 states by absolute mean residual
    state_res = (
        df_geo.groupby("customer_state")["mean_residual"]
        .mean().abs().nlargest(8).index
    )
    for state in state_res:
        row = df_geo[df_geo["customer_state"] == state].nlargest(1, "num_orders").iloc[0]
        ax_map.annotate(
            state, xy=(row["avg_lng"], row["avg_lat"]),
            color=TEXT, fontsize=7, fontweight="bold",
            ha="center", va="bottom",
            xytext=(0, 5), textcoords="offset points",
        )

    ax_map.set_title(
        "Residual Logistics Efficiency by Customer Location\n"
        "(blue = faster than expected, red = bottleneck)",
        color=TEXT, fontsize=13, fontweight="bold", pad=12,
    )
    ax_map.set_xlabel("Longitude", color=TEXT, fontsize=9)
    ax_map.set_ylabel("Latitude",  color=TEXT, fontsize=9)
    ax_map.tick_params(colors=TEXT, labelsize=8)
    for spine in ax_map.spines.values():
        spine.set_edgecolor(EDGE)

    # ------------------------------------------------------------------ #
    # RIGHT PANEL — Log(distance) vs actual lead time                     #
    # ------------------------------------------------------------------ #
    ax_reg.set_facecolor(BG)

    x = df_reg["log_distance"].values
    y = df_reg["actual_lead_time_days"].values
    r = df_reg["residual_days"].values

    pearson_r, p_value = stats.pearsonr(x, y)
    r_squared = pearson_r ** 2

    # Scatter colored by residual using diverging norm
    res_abs_reg  = np.percentile(np.abs(r), 95)
    scatter_norm = mcolors.TwoSlopeNorm(vmin=-res_abs_reg, vcenter=0, vmax=res_abs_reg)

    ax_reg.scatter(
        x, y,
        c=r, cmap="RdBu_r", norm=scatter_norm,
        s=8, alpha=0.35, linewidths=0,
        label="Order",
    )

    # Regression trendline
    x_line = np.linspace(x.min(), x.max(), 300)
    ax_reg.plot(
        x_line, slope * x_line + intercept,
        color="#1e293b", linewidth=2, linestyle="--",
        label=f"Expected Lead Time  (r = {pearson_r:.3f})",
    )

    # Annotation box
    ax_reg.text(
        0.97, 0.97,
        f"Pearson r = {pearson_r:.3f}\n"
        f"R²  =  {r_squared:.3f}\n"
        f"n  =  {len(df_reg):,}\n"
        f"Variance increases with distance",
        transform=ax_reg.transAxes,
        color=TEXT, fontsize=8.5, ha="right", va="top",
        linespacing=1.6,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#f8fafc",
            edgecolor="#94a3b8",
        ),
    )

    ax_reg.set_title("Distance vs Actual Lead Time",
                     color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax_reg.set_xlabel("Log(Distance km + 1)", color=TEXT, fontsize=10)
    ax_reg.set_ylabel("Actual Lead Time (days)", color=TEXT, fontsize=10)
    ax_reg.tick_params(colors=TEXT, labelsize=8)
    ax_reg.grid(color=EDGE, linewidth=0.4, linestyle="--", alpha=0.6)
    for spine in ax_reg.spines.values():
        spine.set_edgecolor(EDGE)

    legend = ax_reg.legend(
        fontsize=9,
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        labelcolor="#0f172a",
    )
    legend.get_frame().set_alpha(0.9)

    # Subtle footnote on rising spread at longer distances
    ax_reg.text(
        0.5, -0.10,
        "Wider spread at higher distances suggests heteroscedasticity and rising delivery uncertainty.",
        transform=ax_reg.transAxes,
        ha="center", va="top", fontsize=7.5, color=EDGE, style="italic",
    )

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    fig.tight_layout(pad=2)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nFigure saved to: {OUTPUT_FILE}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Run the full logistics efficiency analysis.

    Can be called directly (`python -m src.question1`) or via `main.py`
    (`python main.py --question 1`).
    """
    print("Loading and preparing data...")
    df = load_orders()
    df = compute_lead_time(df)
    df = join_customers(df)

    print("Loading geolocation table...")
    geo = load_geo()

    print("Building order-level regression dataset...")
    df_reg, slope, intercept = build_regression_dataset(df, geo)
    print(f"  {len(df_reg):,} orders with valid distance + lead time")
    print(f"  Distance range:   {df_reg['avg_distance_km'].min():.0f} – "
          f"{df_reg['avg_distance_km'].max():.0f} km")
    print(f"  Lead time range:  {df_reg['actual_lead_time_days'].min():.1f} – "
          f"{df_reg['actual_lead_time_days'].max():.1f} days")
    print(f"  Residual range:   {df_reg['residual_days'].min():.1f} – "
          f"{df_reg['residual_days'].max():.1f} days")

    # Check residual mean/std and informal heteroscedasticity signal.
    run_diagnostics(df_reg)

    print("\nBuilding geographic efficiency dataset (zip-prefix level)...")
    df_geo = build_geo_dataset(df_reg, geo)
    print(f"  {len(df_geo):,} zip prefixes after filtering")

    # Impact Score Ranking:
    # Residual inefficiency alone is not enough for prioritisation.
    # We rank regions using both residual severity and order volume
    # to identify bottlenecks with the greatest operational impact.
    run_prioritization(df_geo)

    # Key Insights
    # 1. Distance has a moderate positive relationship with lead time,
    #    but explains only part of delivery performance.
    # 2. Lead-time variance increases at longer distances,
    #    indicating remote deliveries are less predictable.
    # 3. Regional residual clusters show that operational efficiency,
    #    infrastructure, and carrier execution matter beyond geography.

    print("\nRendering figure...")
    build_map_and_regression(df_geo, df_reg, slope, intercept)


if __name__ == "__main__":
    main()
