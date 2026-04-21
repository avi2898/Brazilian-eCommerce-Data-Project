"""
question3.py
------------
## Research Question 3 - Revenue Outlook for a New Seller

What can a new seller realistically expect to earn in a typical month,
and how uncertain is that outcome?

Objectives:
  1. Estimate the real distribution of monthly seller revenue from historical data
  2. Use bootstrap simulation (10,000 draws) to model expected outcomes
  3. Quantify uncertainty via a 95% confidence interval (2.5th–97.5th percentile)
  4. Compute practical probability metrics to ground the results in business terms

Bootstrap Approach:
  Rather than assuming a parametric distribution (e.g. normal), we treat the
  observed seller-month revenues as our empirical population and sample from it
  with replacement. Each of the 10,000 draws represents one hypothetical
  "typical seller month". The resulting distribution of simulated values gives
  a data-driven estimate of expected revenue and its uncertainty — without
  making any assumptions about the underlying distribution shape.

  Revenue is highly right-skewed: most sellers earn relatively modest amounts
  while a small number of high-performers drive the upper tail. In this context
  the median is a more reliable guide for a new seller than the mean.

Output:
  outputs/question_3/bootstrap_sim.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.db_setup import query

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "question_3"
OUTPUT_FILE  = OUTPUT_DIR / "bootstrap_sim.png"

# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------
BG_FIG  = "#0f172a"
BG_AX   = "#1e293b"
TEXT    = "#e2e8f0"
GRID    = "#334155"
EDGE    = "#475569"
BLUE    = "#3b82f6"
GREEN   = "#22c55e"
RED     = "#ef4444"
PURPLE  = "#a78bfa"

N_SIMULATIONS = 10_000
RANDOM_SEED   = 42


# ---------------------------------------------------------------------------
# 1. Load empirical seller-month revenue distribution
# ---------------------------------------------------------------------------
def load_seller_monthly_revenue() -> np.ndarray:
    """
    Query the database and return a 1-D NumPy array of seller monthly revenues.

    Each element represents the total revenue (price + freight_value) earned
    by one seller in one calendar month.
    """
    sql = """
    SELECT
        oi.seller_id,
        strftime('%Y-%m', o.order_purchase_timestamp)   AS month,
        SUM(oi.price + oi.freight_value)                AS seller_monthly_revenue
    FROM order_items oi
    JOIN orders o USING(order_id)
    WHERE o.order_purchase_timestamp IS NOT NULL
    GROUP BY oi.seller_id, month
    """
    df = query(sql)
    return df["seller_monthly_revenue"].astype(float).to_numpy()


# ---------------------------------------------------------------------------
# 2. Run bootstrap simulation
# ---------------------------------------------------------------------------
def run_simulation(revenue_values: np.ndarray) -> np.ndarray:
    """
    Bootstrap-sample with replacement from the empirical distribution N_SIMULATIONS times.

    Parameters
    ----------
    revenue_values : np.ndarray
        Empirical seller-month revenues from historical data.

    Returns
    -------
    np.ndarray
        Array of N_SIMULATIONS simulated revenue draws.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    return rng.choice(revenue_values, size=N_SIMULATIONS, replace=True)


# ---------------------------------------------------------------------------
# 3. Compute summary statistics
# ---------------------------------------------------------------------------
def compute_stats(simulated: np.ndarray) -> dict:
    """
    Return mean, median, 95% CI bounds, and practical probability metrics.

    Parameters
    ----------
    simulated : np.ndarray
        Array of simulated revenue draws.

    Returns
    -------
    dict with keys:
      mean, median, ci_low, ci_high,
      p_below_500, p_below_1000, p_above_5000
    """
    n = len(simulated)
    return {
        "mean":         float(np.mean(simulated)),
        "median":       float(np.median(simulated)),
        "ci_low":       float(np.percentile(simulated, 2.5)),
        "ci_high":      float(np.percentile(simulated, 97.5)),
        "p_below_500":  float(np.sum(simulated < 500)  / n * 100),
        "p_below_1000": float(np.sum(simulated < 1000) / n * 100),
        "p_above_5000": float(np.sum(simulated > 5000) / n * 100),
    }


# ---------------------------------------------------------------------------
# 4. Build visualization
# ---------------------------------------------------------------------------
def build_plot(simulated: np.ndarray, stats: dict) -> None:
    """
    Dark-themed histogram of simulated revenues with annotated reference lines
    and a probability summary box.

    Parameters
    ----------
    simulated : np.ndarray
        Array of simulated revenue draws.
    stats : dict
        Output of compute_stats().
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)

    # Clip to 99th percentile to limit the impact of extreme outliers
    p99      = np.percentile(simulated, 99)
    sim_plot = simulated[simulated <= p99]

    ax.hist(sim_plot, bins=80, color=BLUE, alpha=0.70, edgecolor="none", zorder=2)
    ax.set_xlim(0, p99)

    # Shaded 95% CI region
    ax.axvspan(stats["ci_low"], stats["ci_high"],
               color=BLUE, alpha=0.10, zorder=1, label="95% confidence interval")

    # Reference lines
    ax.axvline(stats["mean"],    color=TEXT,   linewidth=2,   linestyle="-",
               zorder=4, label=f"Mean    R$ {stats['mean']:,.0f}")
    ax.axvline(stats["median"],  color=PURPLE, linewidth=2,   linestyle=":",
               zorder=4, label=f"Median  R$ {stats['median']:,.0f}")
    ax.axvline(stats["ci_low"],  color=RED,    linewidth=1.6, linestyle="--",
               zorder=4, label=f"95% CI lower  R$ {stats['ci_low']:,.0f}")
    ax.axvline(stats["ci_high"], color=GREEN,  linewidth=1.6, linestyle="--",
               zorder=4, label=f"95% CI upper  R$ {stats['ci_high']:,.0f}")

    # Probability annotation box
    prob_text = (
        f"Prob. of earning < R$500:    {stats['p_below_500']:.1f}%\n"
        f"Prob. of earning < R$1,000:  {stats['p_below_1000']:.1f}%\n"
        f"Prob. of earning > R$5,000:  {stats['p_above_5000']:.1f}%"
    )
    ax.text(
        0.995, 0.6, prob_text,
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=8.5, color=TEXT,
        linespacing=1.7,
        bbox=dict(
            facecolor=BG_FIG,
            edgecolor=EDGE,
            boxstyle="round,pad=0.5",
            alpha=0.85
        ),
    )

    # Title and subtitle
    ax.set_title(
        "Revenue Outlook for a New Seller",
        color=TEXT, fontsize=14, fontweight="bold", pad=22, loc="left",
    )
    fig.text(
        0.0675, 0.87,
        "Bootstrap simulation of monthly seller revenue"
        f"  |  {N_SIMULATIONS:,} draws with replacement  |  seed = {RANDOM_SEED}",
        ha="left",
        color="#94a3b8",
        fontsize=9,
    )

    # Axes formatting
    ax.set_xlabel("Simulated Monthly Revenue (R$)", color=TEXT, fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency",                      color=TEXT, fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"R$ {v/1e3:.0f}k" if v >= 1000 else f"R$ {v:.0f}")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--", alpha=0.8)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(EDGE)

    ax.legend(fontsize=9, facecolor=BG_FIG, edgecolor=EDGE, labelcolor=TEXT,
              framealpha=0.9, loc="upper right")
    fig.text(
        0.98, 0.02,
        "*Display capped at 99th percentile for readability (extreme outliers excluded)",
        ha="right",
        color="#94a3b8",
        fontsize=8,
        alpha=0.85,
    )
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
    print("Loading seller monthly revenue data...")
    revenue_values = load_seller_monthly_revenue()
    print(f"  Empirical observations: {len(revenue_values):,} seller-month pairs")
    print(f"  Revenue range: R$ {revenue_values.min():,.0f} – R$ {revenue_values.max():,.0f}")

    print(f"\nRunning bootstrap simulation ({N_SIMULATIONS:,} draws)...")
    simulated = run_simulation(revenue_values)
    stats = compute_stats(simulated)

    print(f"\n  Mean revenue:    R$ {stats['mean']:>10,.2f}")
    print(f"  Median revenue:  R$ {stats['median']:>10,.2f}")
    print(f"  95% CI lower:    R$ {stats['ci_low']:>10,.2f}")
    print(f"  95% CI upper:    R$ {stats['ci_high']:>10,.2f}")

    print(f"\n  Practical probability estimates:")
    print(f"    Earning below R$500:    {stats['p_below_500']:.1f}%")
    print(f"    Earning below R$1,000:  {stats['p_below_1000']:.1f}%")
    print(f"    Earning above R$5,000:  {stats['p_above_5000']:.1f}%")

    print(
        f"\n  Business insight: A new seller can realistically expect to earn around "
        f"R$ {stats['median']:,.0f} per month (median).\n"
        f"  Most outcomes fall between R$ {stats['ci_low']:,.0f} and "
        f"R$ {stats['ci_high']:,.0f} (95% CI).\n"
        f"  There is a {stats['p_above_5000']:.1f}% chance of exceeding R$ 5,000 "
        f"and a {stats['p_below_500']:.1f}% chance of earning under R$ 500."
    )

    print("\nRendering plot...")
    build_plot(simulated, stats)


if __name__ == "__main__":
    main()
