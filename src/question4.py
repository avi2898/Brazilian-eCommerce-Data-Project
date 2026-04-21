"""
question4.py
------------
## Research Question 4 - Payment Methods: Risk vs Value Trade-off

How do payment methods differ in risk and value, and what trade-offs emerge
between reliability and revenue?

Objectives:
  1. Build an order-level dataset with payment type, status, and value
  2. Compute cancellation rate, AOV, order volume, and total revenue per method
  3. Visualise all four dimensions in a single Risk–Value bubble chart

Methodology:
  Orders with more than one distinct payment type are excluded so each order
  is attributed unambiguously to a single method.
  Cancellation is defined as order_status = 'canceled'. Orders with status
  'unavailable' are excluded from the cancellation metric because they likely
  reflect fulfillment or inventory issues rather than payment-related behavior,
  and would confound the payment-method comparison.
  Quadrant lines at the cross-method mean AOV and mean cancellation rate
  divide methods into four interpretable segments.

Output:
  outputs/question_4/risk_matrix.png

"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.db_setup import query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "question_4"
OUTPUT_FILE  = OUTPUT_DIR / "risk_matrix.png"

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
BG_FIG  = "#0f172a"
BG_AX   = "#1e293b"
TEXT    = "#e2e8f0"
GRID    = "#334155"
EDGE    = "#475569"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Return one row per order with:
      order_id       – unique order identifier
      order_status   – raw status string
      payment_type   – payment method (single type per order only)
      payment_value  – total amount paid

    Orders with more than one distinct payment type are excluded.
    """
    sql = """
    WITH payment_agg AS (
        SELECT
            order_id,
            COUNT(DISTINCT payment_type)  AS n_types,
            MAX(payment_type)             AS payment_type,
            SUM(payment_value)            AS payment_value
        FROM order_payments
        GROUP BY order_id
    )
    SELECT
        o.order_id,
        o.order_status,
        p.payment_type,
        p.payment_value
    FROM orders o
    JOIN payment_agg p USING(order_id)
    WHERE p.n_types = 1
      AND p.payment_type != 'not_defined'
    """
    df = query(sql)
    df["payment_value"] = df["payment_value"].astype(float)
    # Cancellation is defined strictly as order_status == 'canceled'.
    # 'unavailable' is excluded: it likely reflects fulfillment/inventory
    # issues rather than payment-related behavior, and would confound the
    # per-payment-method comparison.
    df["cancelled"] = (df["order_status"] == "canceled").astype(int)
    return df


# ---------------------------------------------------------------------------
# 2. Aggregate metrics
# ---------------------------------------------------------------------------
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-payment-type metrics:
      n_orders, cancellation_rate, aov, total_revenue
    """
    summary = (
        df.groupby("payment_type")
        .agg(
            n_orders      =("cancelled",     "count"),
            n_cancelled   =("cancelled",     "sum"),
            aov           =("payment_value", "mean"),
            total_revenue =("payment_value", "sum"),
        )
        .reset_index()
    )
    summary["cancellation_rate"] = (
        summary["n_cancelled"] / summary["n_orders"] * 100
    )
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Visualisation
# ---------------------------------------------------------------------------
def build_plot(summary: pd.DataFrame) -> None:
    """
    Risk–Value bubble chart:
      X-axis       – Average Order Value (AOV)
      Y-axis       – Cancellation Rate (%)
      Bubble size  – Number of orders
      Bubble color – Total revenue (viridis colormap)

    Quadrant lines at the cross-method mean AOV and mean cancellation rate
    divide the space into four labelled segments.
    """
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # Bubble sizes scaled to a visible range
    n = summary["n_orders"].to_numpy(dtype=float)
    sizes = 300 + (n - n.min()) / (n.max() - n.min() + 1e-9) * 2700

    revenue = summary["total_revenue"].to_numpy(dtype=float)
    norm    = plt.Normalize(vmin=revenue.min(), vmax=revenue.max())

    sc = ax.scatter(
        summary["aov"],
        summary["cancellation_rate"],
        s=sizes,
        c=revenue,
        cmap="viridis",
        norm=norm,
        alpha=0.75,
        edgecolors=EDGE,
        linewidths=1.2,
        zorder=3,
    )

    # Colorbar
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Total Revenue (R$)", color=TEXT, fontsize=9, labelpad=8)
    cbar.ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"R$ {v/1e6:.1f}M")
    )
    cbar.ax.tick_params(colors=TEXT, labelsize=8.5)
    cbar.outline.set_edgecolor(EDGE)

    # Quadrant reference lines
    mean_aov    = summary["aov"].mean()
    mean_cancel = summary["cancellation_rate"].mean()

    ax.axvline(mean_aov,    color=EDGE, linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(mean_cancel, color=EDGE, linewidth=1.2, linestyle="--", zorder=2)

    # Derive axis bounds for quadrant label placement
    x_min, x_max = (summary["aov"].min() * 0.85,
                    summary["aov"].max() * 1.12)
    y_min, y_max = (summary["cancellation_rate"].min() * 0.6,
                    summary["cancellation_rate"].max() * 1.2)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    quad_style = dict(
        fontsize=9,
        color="#cbd5e1",
        style="italic",
        ha="center",
        va="center",
        alpha=0.35,
    )
    f = 0.35

    x_left = mean_aov - f * (mean_aov - x_min)
    x_right = mean_aov + f * (x_max - mean_aov)
    y_top = mean_cancel + f * (y_max - mean_cancel)
    y_bottom = mean_cancel - f * (mean_cancel - y_min)

    ax.text(x_left, y_top, "High Risk\nLow Value", **quad_style)
    ax.text(x_right, y_top, "High Risk\nHigh Value", **quad_style)
    ax.text(x_left, y_bottom, "Low Risk\nLow Value", **quad_style)
    ax.text(x_right, y_bottom, "Low Risk\nHigh Value", **quad_style)

    # Bubble annotations with per-label offsets to avoid overlap.
    # debit_card uses a leader line because its bubble sits close to boleto;
    # the offset points left-and-down to clearly separate the two labels.
    offsets = {
        "credit_card": ( 24,  12),
        "boleto":      ( 10,  10),
        "voucher":     (-10,  10),
        "debit_card":  (-65, -9),
    }

    for _, row in summary.iterrows():
        ptype = row["payment_type"]
        dx, dy = offsets.get(ptype, (10, 10))
        ax.annotate(
            ptype.replace("_", " ").title(),
            xy=(row["aov"], row["cancellation_rate"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
            fontweight="bold",
        )

    # Axes formatting
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"R$ {v:,.0f}")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}%")
    )
    ax.set_xlabel("Average Order Value (AOV)", color=TEXT, fontsize=10, labelpad=8)
    ax.set_ylabel("Cancellation Rate (%)",     color=TEXT, fontsize=10, labelpad=8)

    ax.set_title(
        "Payment Methods: Risk–Value Tradeoff",
        color=TEXT,
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    fig.text(
        0.462, 0.89,
        "Olist E-commerce | Bubble size = order volume | Color = total revenue",
        ha="center",
        color="#94a3b8",
        fontsize=9,
    )

    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)

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
    print("Loading order and payment data...")
    df = load_data()
    print(f"  {len(df):,} orders with a single payment type")
    print(f"  Payment types: {sorted(df['payment_type'].unique())}")

    print("\nBuilding summary metrics...")
    summary = build_summary(df)

    print("\n  Metrics by payment type:")
    print(f"  {'Payment':<16} {'Orders':>8} {'Cancel%':>9}  {'AOV':>12}  {'Revenue':>14}")
    print("  " + "-" * 65)
    for _, row in summary.iterrows():
        print(f"  {row['payment_type']:<16} {row['n_orders']:>8,} "
              f"{row['cancellation_rate']:>8.2f}%  "
              f"R$ {row['aov']:>8,.0f}  "
              f"R$ {row['total_revenue']:>10,.0f}")

    print("\nRendering plot...")
    build_plot(summary)


if __name__ == "__main__":
    main()
