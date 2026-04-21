# Codebase Guide — Brazilian E-Commerce Analytics Project

A deep-dive reference for understanding the architecture, code, and analytical decisions
behind this project.

---

## Table of Contents

1. [Project Architecture Overview](#1-project-architecture-overview)
2. [Execution Flow](#2-execution-flow)
3. [File-by-File Breakdown](#3-file-by-file-breakdown)
   - [db_setup.py](#db_setuppy)
   - [main.py](#mainpy)
   - [question1.py — Logistics Efficiency](#question1py--logistics-efficiency--residual-analysis)
   - [question2.py — Sales Trends](#question2py--sales-trends--seasonality)
   - [question3.py — Bootstrap Simulation](#question3py--seller-revenue-bootstrap-simulation)
   - [question4.py — Payment Methods](#question4py--payment-methods-risk-vs-value)
4. [Line-by-Line Deep Dives](#4-line-by-line-deep-dives)
   - [db_setup.py explained](#db_setuppy-explained)
   - [main.py explained](#mainpy-explained)
5. [Key Analytical Decisions](#5-key-analytical-decisions)

---

## 1. Project Architecture Overview

```
project_root/
├── main.py                      ← entry point; runs any/all questions
├── README.md                    ← project overview and analytical walkthrough
├── requirements.txt             ← pinned dependency list
├── project_walkthrough.ipynb    ← full analytical notebook with outputs
├── .gitignore                   ← excludes data, venv, compiled files
│
├── src/
│   ├── __init__.py      ← marks src/ as a Python package
│   ├── db_setup.py      ← builds SQLite database; exposes query() helper
│   ├── question1.py     ← Q1: logistics efficiency map + residual regression
│   ├── question2.py     ← Q2: sales trends + MoM growth (dual-axis chart)
│   ├── question3.py     ← Q3: seller revenue bootstrap simulation
│   └── question4.py     ← Q4: payment methods risk–value bubble chart
│
├── data/                ← raw Olist CSVs (git-ignored, too large to commit)
├── outputs/
│   ├── question_1/      ← charts saved here by question1.py
│   ├── question_2/
│   ├── question_3/
│   └── question_4/
│
└── docs/
    └── codebase_guide.md ← this file
```

**Key design principles:**
- **Single database, many queries** — all question files share one SQLite file (`ecommerce.db`) accessed through a shared `query()` helper. No file reads after setup.
- **Separation of concerns** — data loading, computation, and visualisation live in separate functions within each question file.
- **Zero duplication of infrastructure** — `db_setup.py` centralises all database logic. Question files stay focused on their specific analysis.
- **Self-contained outputs** — each question saves its chart to `outputs/question_N/` and prints a confirmation path. No side effects on other modules.

---

## 2. Execution Flow

```
User runs: python main.py --question 2
                │
                ▼
          main.py::main()
          - parses CLI args
          - creates outputs/question_N/ dirs
          - calls run_question(2)
                │
                ▼
          importlib.import_module("src.question2")
          mod.main()
                │
         ┌──────┴──────────────────────────────────┐
         ▼                                          ▼
   load_monthly_sales()                       build_plot(df)
   - calls query(sql)                         - creates Matplotlib figure
     └─► db_setup.query()                     - saves to outputs/question_2/
         └─► sqlite3 → pd.DataFrame           - plt.show()
```

Or for the full run (`python main.py`): questions 1 → 4 execute in sequence, each printing timing.

**Database setup** (one-time, separate step):
```
python -m src.db_setup
```
This reads 9 CSV files from `data/`, loads them into `ecommerce.db` as SQLite tables, and prints row counts. After this, `main.py` can be run any number of times without touching CSV files again.

---

## 3. File-by-File Breakdown

### db_setup.py

**What it does:** Reads Olist CSV files, creates/replaces SQLite tables, exposes the `query()` helper.

**Key objects:**
| Name | Type | Purpose |
|------|------|---------|
| `DB_PATH` | `Path` | Location of `ecommerce.db` |
| `TABLES` | `dict` | Maps table name → CSV filename |
| `DATE_COLUMNS` | `dict` | Columns to parse as datetime per table |
| `query(sql)` | function | Run SQL, return DataFrame |
| `build_database()` | function | Load all CSVs into SQLite |

**Why SQLite instead of reading CSVs directly?**
Once the database is built, all queries use SQL — which is faster to write for joins and aggregations, avoids re-reading 9 large CSVs on every run. SQLite requires no server setup and stores the whole database as a single file.

---

### main.py

**What it does:** CLI entry point. Accepts `--question N` to run one question or runs all four in sequence.

**Key design choices:**
- Uses `importlib.import_module()` instead of static imports — this means question modules are only loaded when needed, and adding a new question requires no changes to import blocks.
- Creates output subdirectories (`outputs/question_N/`) before running any question, so question files never need to worry about directory existence.
- Times each question and prints a summary.

---

### question1.py — Logistics Efficiency & Residual Analysis

**Research question:** Which Brazilian states over- or underperform delivery expectations relative to the distance alone?

**Pipeline:**
```
load_orders()              ← actual delivery timestamps only
    │
compute_lead_time()        ← (delivered - purchased) / 86400 seconds
    │
join_customers()           ← attach customer state/zip
    │
load_geo()                 ← average lat/lng per zip prefix
    │
build_regression_dataset() ← Haversine distance + log-linear OLS regression
    │                         residual = actual − predicted
    ├── run_diagnostics()  ← residual mean/std + Pearson correlation (heteroscedasticity check)
    │
build_geo_dataset()        ← aggregate to zip-prefix level
    │
run_prioritization()       ← rank states by impact = residual × log(volume)
    │
build_map_and_regression() ← two-panel chart: Brazil map + scatter plot
```

**Why Haversine distance?** Latitude/longitude coordinates are not on a flat plane. A straight Euclidean distance would underestimate long routes and give incorrect results near the poles. The Haversine formula computes the great-circle distance — the shortest path over a spherical Earth. It is implemented as a vectorised NumPy operation in `question1.py`, so it runs on all rows simultaneously without a Python loop.

**Why log(distance)?** Distance values are right-skewed — a few very long orders (e.g. Amazon to remote north) would dominate a linear fit. `log1p()` compresses the tail so the regression line fits the bulk of orders better.

**What is a residual here?** After fitting `actual_days ~ log(distance)`, each order's residual is `actual − predicted`. A positive residual means the region took *longer* than its distance would predict — a bottleneck signal. A negative residual means it was *faster than expected* — an efficiency signal.

**Output:** Two-panel PNG: left = Brazil geospatial scatter plot (coloured by mean residual per zip), right = scatter plot (log distance vs lead time, coloured by residual).

---

### question2.py — Sales Trends & Seasonality

**Research question:** What does monthly revenue growth look like over time, and which months are peak months?

**Pipeline:**
```
load_monthly_sales()   ← SQL window functions: LAG + SUM OVER
    │
main()                 ← filter to 2017-01 → 2018-08 stable window
    │                     suppress first MoM (no prior period)
    │
build_plot()           ← dual-axis combo chart
                          primary axis:   total sales line
                          secondary axis: MoM growth bars
```

**SQL window functions used:**
- `LAG(total_sales) OVER (ORDER BY month)` — the previous month's sales, used to compute month-over-month growth in one SQL pass with no Python post-processing.
- `SUM(total_sales) OVER (ORDER BY month)` — cumulative running total computed in SQL (available in the DataFrame but not plotted, to keep the chart focused).

**Why a dual-axis chart?** Sales (R$) and growth rate (%) live on incompatible scales. A secondary y-axis lets both be shown on the same time axis without distortion. The key technical trick is `ax.set_zorder(ax2.get_zorder() + 1)` combined with `ax.patch.set_visible(False)` — this forces the sales line to render in front of the growth bars while still letting the background show through.

**Why exclude months below R$100,000 in prior sales?** The early months have only a handful of orders and no realistic prior-month comparison. Their MoM growth rates are enormous (e.g. +2000%) and statistically meaningless. Filtering on `prev_month_sales >= 100,000` cleanly removes them without hard-coding dates.

---

### question3.py — Seller Revenue Bootstrap Simulation

**Research question:** What can a new seller realistically expect to earn per month, and how uncertain is that outcome?

**Pipeline:**
```
load_seller_monthly_revenue()   ← one value per (seller_id, month) pair
    │
run_simulation()                ← 10,000 bootstrap draws with replacement
    │
compute_stats()                 ← mean, median, 95% CI, probability thresholds
    │
build_plot()                    ← dark histogram with reference lines + annotation box
```

**Why bootstrap sampling instead of just reporting the mean?**
Revenue is highly right-skewed — a handful of top sellers earn 10–100x the median. Reporting the mean alone is misleading. By drawing 10,000 samples from the empirical distribution (with replacement), we simulate what a "random seller month" looks like across many hypothetical scenarios, and the resulting histogram makes the skew visible. No parametric distribution (e.g. normal, lognormal) is assumed.

**What is `np.random.default_rng(RANDOM_SEED)`?** This creates a reproducible random number generator using NumPy's modern Generator API (preferred over `np.random.seed()` since NumPy 1.17). Setting `RANDOM_SEED = 42` means the same 10,000 draws are produced every time the script runs — results are reproducible.

**Why clip display at the 99th percentile?** A small number of seller-months earn extremely high revenues (outliers). Showing the full x-axis would compress most of the histogram into a tiny region. Clipping at p99 shows the meaningful distribution without distortion, with a footnote that explains the exclusion.

**Output:** Dark-themed histogram with mean/median/CI reference lines and a probability annotation box showing P(< R$500), P(< R$1,000), P(> R$5,000).

---

### question4.py — Payment Methods Risk vs Value

**Research question:** Do payment methods differ in cancellation risk and average order value, and what trade-offs exist?

**Pipeline:**
```
load_data()         ← SQL CTE: filter to single-payment-type orders only
    │               ← label cancelled = (order_status == 'canceled')
    │
build_summary()     ← groupby payment_type: n_orders, cancel rate, AOV, revenue
    │
build_plot()        ← Risk–Value bubble chart
                       X = AOV, Y = cancel rate
                       size = order volume, color = total revenue (viridis)
                       quadrant lines at cross-method means
```

**Why exclude multi-payment orders?** Some orders use a combination of methods (e.g. voucher + credit card). Attributing those orders to a single payment type would be ambiguous — the cancellation or value might be driven by either method. Restricting to single-type orders gives clean, unambiguous attribution.

**Why exclude `unavailable` from the cancellation metric?** Cancellation is defined strictly as `order_status = 'canceled'`. Orders marked `unavailable` are dropped from the cancellation count because they likely reflect fulfillment or inventory issues on the seller side rather than payment-related behavior — including them would confound the payment-method comparison, which is specifically about how payment choice affects cancellation behavior.

**What is a Risk–Value bubble chart?** A scatter plot where each point is a category (payment method), with two additional dimensions encoded visually: bubble size encodes order volume (so you see how many orders each method accounts for) and colour encodes total revenue. The quadrant lines divide the space into four interpretable segments: High Risk/Low Value, High Risk/High Value, Low Risk/Low Value, Low Risk/High Value.

**Why viridis colormap?** Viridis is perceptually uniform (equal visual distance per unit of data change), accessible to colour-blind readers, and looks good on dark backgrounds. Unlike rainbow or jet colormaps, it doesn't create false peaks.

---

## 4. Line-by-Line Deep Dives

### db_setup.py explained

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```
`__file__` is the path to `db_setup.py` itself. `.resolve()` converts it to an absolute path. `.parent` goes up one level (from `db_setup.py` to `src/`). `.parent` again goes up to `project_root/`. This pattern means the code works regardless of where you run it from — no hardcoded paths.

```python
DB_PATH = PROJECT_ROOT / "ecommerce.db"
```
The `/` operator on `Path` objects is overloaded to join path segments. Equivalent to `os.path.join()` but more readable.

```python
TABLES = {
    "orders": "olist_orders_dataset.csv",
    ...
}
```
A registry of all table names and their source files. When `build_database()` loops over this, a new CSV only requires one line here — no other code changes.

```python
DATE_COLUMNS = {
    "orders": ["order_purchase_timestamp", ...],
}
```
Tells pandas which columns to parse as datetimes during CSV loading. Only defined for tables that actually have date columns, avoiding unnecessary overhead.

```python
def query(sql: str, params=None) -> pd.DataFrame:
    with _get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)
```
This is the entire interface that all question files use. The `with` block ensures the connection is closed after each query even if an exception is raised. `params` supports parameterised queries (prevents SQL injection, though here all SQL is static).

```python
def _load_csv(table: str, filename: str) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    parse_dates = [c for c in DATE_COLUMNS.get(table, []) if c in header]
    return pd.read_csv(csv_path, parse_dates=parse_dates)
```
First reads just the header row (`nrows=0`) to check which columns exist before reading the full file. This guards against schema changes upstream — if a column is renamed in a future Olist version, the load won't crash.

```python
df.to_sql(table, conn, if_exists="replace", index=False)
```
`if_exists="replace"` drops and recreates the table if it already exists, making `build_database()` idempotent — safe to re-run. `index=False` prevents pandas from adding an extra auto-increment column.

---

### main.py explained

```python
import importlib

mod = importlib.import_module(module_path)
mod.main()
```
Instead of `from src import question2; question2.main()` at the top of the file, this dynamically imports the module at runtime. Benefit: if you add a question 5, you only need to update the `module_map` dict. No new import statements. The module is only loaded when actually requested.

```python
for i in range(1, 5):
    (root / "outputs" / f"question_{i}").mkdir(parents=True, exist_ok=True)
```
`parents=True` creates the full directory tree if needed (e.g. if `outputs/` itself doesn't exist yet). `exist_ok=True` means it won't raise an error if the folder already exists. This runs before any question, so every question file can safely write to its subfolder without checking.

```python
parser.add_argument("--question", "-q", type=int, choices=[1, 2, 3, 4], default=None)
```
`type=int` automatically converts the string argument to an integer. `choices=[1, 2, 3, 4]` makes argparse validate and reject invalid inputs before the script does anything. `default=None` means omitting the flag runs all questions.

---

## 5. Key Analytical Decisions

### Why exclude estimated delivery dates (question1)?
The `order_estimated_delivery_date` column contains dates that Olist commits to customers. These are not measured delivery times — they are pre-set business promises, often padded by days or weeks to manage expectations. Using them as a ceiling would bias the analysis toward measuring "did the platform meet its promise" rather than "how efficient is the logistics operation". By using only `order_delivered_customer_date − order_purchase_timestamp`, we measure true operational speed.

### Why log-transform distance (question1)?
Distance in km is right-skewed: most orders are within 500 km, but some cross the full width of Brazil (3,000+ km). A linear model on raw distance would be pulled toward fitting the extreme long-distance tail. `log1p(distance)` compresses the scale so the model weights all observations more evenly. It also captures the likely *diminishing marginal* effect of distance: the difference between 100 km and 200 km probably matters more than the difference between 2,000 km and 2,100 km.

### Why impact score = residual × log(volume) (question1)?
A state with a large positive residual (slow deliveries) but only 10 orders is less actionable than a state with a moderate residual and 10,000 orders. Multiplying residual by `log1p(total_orders)` combines severity and scale. The log dampens the effect of very high volumes so that a state with 50,000 orders doesn't automatically dominate one with 5,000.

### Why bootstrap sampling instead of a confidence interval formula (question3)?
Classical confidence interval formulas assume a specific distribution (usually normal). Seller revenue is strongly right-skewed — the normal assumption is false. Bootstrap sampling makes no distributional assumption: it treats the observed seller-month revenues as the population and re-samples from them. The resulting simulation reflects the actual shape of the distribution, including the long right tail.

### Why filter orders to single payment type (question4)?
A split payment (e.g. 60% voucher + 40% credit card) cannot be cleanly attributed to either method for the purpose of cancellation rate or AOV analysis. Including split-payment orders would contaminate both groups. Excluding them means the analysis is clean, though it slightly underrepresents voucher usage (vouchers are often used as partial payment).

