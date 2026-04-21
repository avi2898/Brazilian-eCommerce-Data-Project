"""
question2.py
------------
## Research Question 2 - Sales Growth & Seasonality

What is the pattern of sales growth over time, and which months exhibit seasonality?

Objectives:
  1. Compute monthly total sales (revenue) and order volume over time
  2. Measure month-over-month growth rate using SQL window functions
  3. Identify seasonal peaks and recurring patterns

Methodology:
  Orders are joined to order_items to capture both price and freight revenue.
  Monthly aggregation uses strftime('%Y-%m', order_purchase_timestamp).
  Window functions are used directly in SQL to avoid post-processing:
    - LAG()           -> previous month's sales for growth calculation
    - SUM() OVER()    -> cumulative sales (computed in SQL, not plotted)
  Growth rate = (current - previous) / previous, with safe division (NULL when
  previous is zero or missing).
  Seasonal peaks are identified as the top 3 months by total sales.
  Early months with very low sales are excluded from all visualizations
  to avoid distortions caused by partial data coverage.

Output:
  outputs/question_2/sales_trend.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

from src.db_setup import query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "question_2"
OUTPUT_FILE  = OUTPUT_DIR / "sales_trend.png"

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
BG_FIG  = "#0f172a"
BG_AX   = "#1e293b"
TEXT    = "#e2e8f0"
GRID    = "#334155"
EDGE    = "#475569"
BLUE    = "#3b82f6"    # primary sales line
BLUE_LT = "#60a5fa"   # dot markers
GREEN   = "#22c55e"   # positive growth bars
RED     = "#ef4444"   # negative growth bars / peak markers


# ---------------------------------------------------------------------------
# 1. Load monthly sales data via SQL window functions
# ---------------------------------------------------------------------------
def load_monthly_sales() -> pd.DataFrame:
    """
    Returns a DataFrame with one row per month containing:
      - month              : 'YYYY-MM' string
      - total_sales        : sum of price + freight_value
      - total_orders       : distinct order count
      - prev_month_sales   : previous month's total_sales (LAG)
      - growth_rate        : month-over-month growth (NULL for first month)
      - cumulative_sales   : running total of sales (SUM OVER)
    """
    sql = """
    WITH monthly AS (
        -- Step 1: aggregate to one row per month
        SELECT
            strftime('%Y-%m', o.order_purchase_timestamp)   AS month,
            SUM(oi.price + oi.freight_value)                AS total_sales,
            COUNT(DISTINCT o.order_id)                      AS total_orders
        FROM orders o
        JOIN order_items oi USING(order_id)
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY month
    )
    SELECT
        month,
        total_sales,
        total_orders,

        -- Previous month sales using LAG window function
        LAG(total_sales) OVER (ORDER BY month)              AS prev_month_sales,

        -- Month-over-month growth rate; NULL when no prior month exists or prev = 0
        CASE
            WHEN LAG(total_sales) OVER (ORDER BY month) IS NULL
              OR LAG(total_sales) OVER (ORDER BY month) = 0
            THEN NULL
            ELSE (total_sales - LAG(total_sales) OVER (ORDER BY month))
                 / LAG(total_sales) OVER (ORDER BY month)
        END                                                 AS growth_rate,

        -- Cumulative sales using SUM window function
        SUM(total_sales) OVER (ORDER BY month)              AS cumulative_sales

    FROM monthly
    ORDER BY month
    """
    df = query(sql)

    # Ensure proper types
    df["month"]            = pd.to_datetime(df["month"], format="%Y-%m")
    df["total_sales"]      = df["total_sales"].astype(float)
    df["total_orders"]     = df["total_orders"].astype(int)
    df["growth_rate"]      = pd.to_numeric(df["growth_rate"], errors="coerce")
    df["cumulative_sales"] = df["cumulative_sales"].astype(float)

    return df


# ---------------------------------------------------------------------------
# 2. Filter sales data to the stable analysis window
# ---------------------------------------------------------------------------
def filter_sales_window(
    df: pd.DataFrame,
    *,
    prev_min: float = 100_000,
    start: str = "2017-01-01",
    end: str = "2018-08-01",
) -> pd.DataFrame:
    """
    Apply the standard filtering and time-window restriction used in the analysis.

    Steps (applied in order):
      1. Remove early months where the previous month had < prev_min revenue
         (data coverage is sparse before mid-2016).
      2. Remove months with zero or missing total sales.
      3. Restrict to the stable data window (default: Jan 2017 – Aug 2018).
      4. Suppress MoM growth for the first month in the window (no valid prior period).

    Parameters
    ----------
    df       : raw DataFrame from load_monthly_sales()
    prev_min : minimum previous-month revenue to retain a row (default 100,000)
    start    : window start date string (inclusive)
    end      : window end date string (inclusive)

    Returns
    -------
    pd.DataFrame ready for build_plot()
    """
    df = df[df["prev_month_sales"] >= prev_min].copy()
    df = df[df["total_sales"] > 0]
    df = df[
        (df["month"] >= start) &
        (df["month"] <= end)
    ].copy()
    # Suppress MoM growth for the first month — no valid prior period in this window
    df.loc[df.index[:1], "growth_rate"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 3. Build dual-axis combo chart
# ---------------------------------------------------------------------------
def build_plot(df: pd.DataFrame) -> plt.Figure:
    """
    Single dual-axis combo chart:
      Primary axis (left)   – Monthly total sales line
      Secondary axis (right) – Month-over-month growth rate bars
    Both share the same x-axis (time).
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # Secondary y-axis for MoM growth
    ax2 = ax.twinx()
    ax2.set_facecolor(BG_AX)

    x = df["month"]

    # ------------------------------------------------------------------ #
    # MoM growth bars — plotted first so the sales line sits on top       #
    # ------------------------------------------------------------------ #
    growth = df[df["growth_rate"].notna()].copy()
    growth["growth_rate"] = growth["growth_rate"].clip(-2, 2)
    bar_colors = [GREEN if v >= 0 else RED for v in growth["growth_rate"]]

    ax2.bar(growth["month"], growth["growth_rate"] * 100,
            color=bar_colors, width=18, alpha=0.60, label="MoM Growth", zorder=2)
    ax2.axhline(0, color=EDGE, linewidth=1, zorder=3)

    ax2.set_ylabel("MoM Growth (%)", color=TEXT, fontsize=10, labelpad=8)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax2.tick_params(colors=TEXT, labelsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_edgecolor(EDGE)
    ax2.spines["right"].set_edgecolor(EDGE)
    ax2.spines["bottom"].set_edgecolor(EDGE)

    # ------------------------------------------------------------------ #
    # Sales line — primary axis                                           #
    # ------------------------------------------------------------------ #
    ax.plot(x, df["total_sales"],
            color=BLUE, linewidth=2.8, alpha=0.95, zorder=4,
            solid_capstyle="round", label="Sales")

    # Subtle dot markers
    ax.scatter(x, df["total_sales"],
               color=BLUE_LT, s=25, zorder=5, linewidths=0)

    # Mark top-3 peak months
    top3 = df.nlargest(3, "total_sales").sort_values("month")
    ax.scatter(top3["month"], top3["total_sales"],
               color=RED, s=90, zorder=6, label="Top 3 months", linewidths=0)

    offsets = [(0, 22), (-30, 32), (30, 32)]
    for (_, row), (dx, dy) in zip(top3.iterrows(), offsets):
        ax.annotate(
            f"{row['month'].strftime('%b %Y')}  R$ {row['total_sales']/1e6:.2f}M",
            xy=(row["month"], row["total_sales"]),
            xytext=(dx, dy), textcoords="offset points",
            ha="center", fontsize=8.5, color=RED, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8),
        )

    ax.set_ylabel("Total Sales (R$)", color=TEXT, fontsize=10, labelpad=8)
    ax.set_xlabel("Month", color=TEXT, fontsize=10, labelpad=8)
    ax.set_xticks(df["month"])

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"R$ {v/1e6:.1f}M")
    )
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_edgecolor(EDGE)
    ax.spines["right"].set_edgecolor(EDGE)
    ax.spines["bottom"].set_edgecolor(EDGE)

    # Grid on primary axis only to avoid doubling
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)
    ax.set_zorder(ax2.get_zorder() + 1)   # keep sales line above bars
    ax.patch.set_visible(False)            # let ax2 background show through

    # Title
    ax.set_title(
        "Monthly Sales Trend and Month-over-Month Growth",
        color=TEXT, fontsize=14, fontweight="bold", pad=14, loc="left",
    )

    # Custom legend with explicit handles so positive/negative growth bars
    # are distinguished rather than shown as a single combined entry.
    import matplotlib.patches as mpatches
    import matplotlib.lines  as mlines

    h_sales = mlines.Line2D(
        [], [], color=BLUE, linewidth=2.5, label="Monthly Sales"
    )
    h_pos = mpatches.Patch(
        facecolor=GREEN, alpha=0.70, label="Positive MoM Growth"
    )
    h_neg = mpatches.Patch(
        facecolor=RED, alpha=0.70, label="Negative MoM Growth"
    )
    h_peak = mlines.Line2D(
        [], [], marker="o", color="none", markerfacecolor=RED,
        markersize=7, label="Top 3 Months"
    )
    ax.legend(
        handles=[h_sales, h_pos, h_neg, h_peak],
        fontsize=9, facecolor=BG_FIG, edgecolor=EDGE, labelcolor=TEXT,
        framealpha=0.7, loc="lower right",
    )

    fig.autofmt_xdate(rotation=35, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nPlot saved to: {OUTPUT_FILE}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading monthly sales data...")
    df = load_monthly_sales()

    # Brief summary before window restriction (full dataset stats)
    df_prelim = df[df["prev_month_sales"] >= 100_000].copy()
    df_prelim = df_prelim[df_prelim["total_sales"] > 0]
    print(f"  {len(df_prelim)} months of data available  "
          f"({df_prelim['month'].min().strftime('%Y-%m')} to "
          f"{df_prelim['month'].max().strftime('%Y-%m')})")
    print(f"  Total revenue:  R$ {df_prelim['total_sales'].sum():,.0f}")
    print(f"  Total orders:   {df_prelim['total_orders'].sum():,}")
    print(f"  Avg MoM growth: {df_prelim['growth_rate'].mean() * 100:.1f}%")

    # Apply standard filtering and window restriction via reusable helper
    df_plot = filter_sales_window(df)

    print("Rendering plot...")
    build_plot(df_plot)


if __name__ == "__main__":
    main()
