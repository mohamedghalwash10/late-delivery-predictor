"""Stage 3b: exploratory data analysis. Saves figures and prints the numbers behind them."""
import matplotlib

matplotlib.use("Agg")  # draw straight to files, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.build_dataset import NUMERIC_FEATURES, TARGET, load_base_table
from src.config import FIGURES_DIR, PROJECT_ROOT

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

BLUE = "#4C72B0"
REPORT_PATH = PROJECT_ROOT / "reports" / "stage3_eda_findings.txt"
_lines: list[str] = []


def log(text="") -> None:
    text = str(text)
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def save_fig(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)
    log(f"  [figure] {name}")


def late_rate_table(df, col, min_orders):
    g = df.groupby(col, observed=True)[TARGET].agg(orders="size", late_rate_pct="mean")
    g["late_rate_pct"] = (g["late_rate_pct"] * 100).round(2)
    return g[g["orders"] >= min_orders]


def quantile_bins(series, q):
    """Return 1..q bin numbers (1 = lowest values) and the bin edges."""
    bins, edges = pd.qcut(series, q, labels=False, retbins=True, duplicates="drop")
    return (bins + 1).astype("Int64"), edges


def plot_rate(table, title, name, overall, ordered=False, top=None):
    """Bar chart of late rate. ordered=True keeps the natural order (time, bins)."""
    if ordered:
        t = table.sort_index()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x=t.index.astype(str), y=t["late_rate_pct"], color=BLUE, ax=ax)
        ax.axhline(overall, color="red", ls="--", label=f"overall {overall:.1f}%")
        ax.set_ylabel("Late rate (%)")
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
    else:
        t = table.sort_values("late_rate_pct", ascending=False)
        if top:
            t = t.head(top)
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(t) + 1.5)))
        sns.barplot(x=t["late_rate_pct"], y=t.index.astype(str),
                    color=BLUE, ax=ax, orient="h")
        ax.axvline(overall, color="red", ls="--", label=f"overall {overall:.1f}%")
        ax.set_xlabel("Late rate (%)")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, name)


# --------------------------------------------------------------------------
def univariate(df) -> None:
    section("1. UNIVARIATE ANALYSIS")

    counts = df[TARGET].value_counts().sort_index()
    log(f"Target: on-time {counts[0]:,} ({counts[0] / len(df) * 100:.2f}%), "
        f"late {counts[1]:,} ({counts[1] / len(df) * 100:.2f}%)")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=["On-time (0)", "Late (1)"], y=counts.values, color=BLUE, ax=ax)
    ax.set_title("Target distribution")
    ax.set_ylabel("Orders")
    save_fig(fig, "01_target_distribution.png")

    numeric = ["estimated_delivery_days", "distance_km", "total_price",
               "total_freight", "freight_ratio", "total_weight_g",
               "total_volume_cm3", "shipping_limit_days"]
    log("\nNumeric features (summary):")
    log(df[numeric].describe(percentiles=[.01, .25, .5, .75, .95, .99])
        .round(2).T.to_string())

    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    for ax, col in zip(axes.ravel(), numeric):
        s = df[col].dropna()
        s = s[s <= s.quantile(0.99)]
        sns.histplot(s, bins=40, ax=ax, color=BLUE)
        ax.set_title(col)
    fig.suptitle("Numeric features (values above the 99th percentile hidden)")
    save_fig(fig, "02_numeric_distributions.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    s = df["days_vs_estimate"].clip(-40, 40)
    sns.histplot(s, bins=80, ax=ax, color=BLUE)
    ax.axvline(0, color="red", ls="--", label="estimated date (right of line = late)")
    ax.set_title("Actual minus estimated delivery date, in days (clipped at +/-40)")
    ax.legend()
    save_fig(fig, "03_days_vs_estimate.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (col, n) in zip(axes.ravel(), [("customer_state", 10), ("seller_state", 10),
                                           ("product_category", 12), ("payment_type", 6)]):
        vc = df[col].value_counts().head(n)
        sns.barplot(x=vc.values, y=vc.index.astype(str), color=BLUE, ax=ax, orient="h")
        ax.set_title(f"Top {len(vc)}: {col}")
    save_fig(fig, "04_categorical_counts.png")


# --------------------------------------------------------------------------
def bivariate(df, overall) -> None:
    section("2. BIVARIATE ANALYSIS: late rate by factor")

    for col, min_n, top, fname in [
        ("customer_state", 200, None, "05_late_by_customer_state.png"),
        ("seller_state", 100, None, "06_late_by_seller_state.png"),
        ("product_category", 300, 15, "07_late_by_category.png"),
    ]:
        tab = late_rate_table(df, col, min_n)
        log(f"\nLate rate by {col} (groups with >= {min_n} orders), top 10:")
        log(tab.sort_values("late_rate_pct", ascending=False).head(10).to_string())
        plot_rate(tab, f"Late rate by {col}", fname, overall, top=top)

    # ---- time ----
    tab = late_rate_table(df, "purchase_year_month", 200)
    log("\nLate rate by purchase month (months with >= 200 orders):")
    log(tab.to_string())
    plot_rate(tab, "Late rate by purchase month", "08_late_by_year_month.png",
              overall, ordered=True)

    for col, title, fname in [
        ("purchase_month", "Late rate by calendar month", "09_late_by_month.png"),
        ("purchase_dayofweek", "Late rate by weekday (0 = Monday)", "10_late_by_weekday.png"),
        ("purchase_hour", "Late rate by purchase hour", "11_late_by_hour.png"),
    ]:
        tab = late_rate_table(df, col, 1)
        log(f"\nLate rate by {col}:")
        log(tab.to_string())
        plot_rate(tab, title, fname, overall, ordered=True)

    # ---- numeric factors, in deciles ----
    n = 12
    for col, fname in [
        ("distance_km", "distance"), ("estimated_delivery_days", "estimated_days"),
        ("total_freight", "freight"), ("total_weight_g", "weight"),
        ("shipping_limit_days", "shipping_limit"),
    ]:
        bins, edges = quantile_bins(df[col], 10)
        tab = late_rate_table(df.assign(_bin=bins), "_bin", 1)
        log(f"\nLate rate by {col} decile (1 = lowest). Bin edges: "
            f"{np.round(edges, 1).tolist()}")
        log(tab.to_string())
        plot_rate(tab, f"Late rate by {col} decile (1 = lowest)",
                  f"{n}_late_by_{fname}.png", overall, ordered=True)
        n += 1

    # ---- other factors ----
    tmp = df.assign(n_items_capped=df["n_items"].clip(upper=5),
                    n_sellers_capped=df["n_sellers"].clip(upper=3))
    for col, title, fname, ordered in [
        ("payment_type", "Late rate by payment type", "17_late_by_payment.png", False),
        ("same_state", "Late rate: seller and customer in same state (1 = yes)",
         "18_late_by_same_state.png", True),
        ("n_sellers_capped", "Late rate by number of sellers (3 = 3 or more)",
         "19_late_by_n_sellers.png", True),
        ("n_items_capped", "Late rate by number of items (5 = 5 or more)",
         "20_late_by_n_items.png", True),
    ]:
        tab = late_rate_table(tmp, col, 30)
        log(f"\nLate rate by {col}:")
        log(tab.to_string())
        plot_rate(tab, title, fname, overall, ordered=ordered)

    # ---- distributions split by target ----
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, col in zip(axes, ["estimated_delivery_days", "distance_km",
                              "total_freight", "total_weight_g"]):
        sns.boxplot(data=df, x=TARGET, y=col, showfliers=False, color=BLUE, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("is_late")
    save_fig(fig, "21_boxplots_by_target.png")

    # ---- customer impact (review scores exist only AFTER delivery) ----
    r = df.dropna(subset=["review_score"])
    log("\nCustomer impact: review score by delivery outcome (analysis only):")
    log(r.groupby(TARGET)["review_score"].agg(orders="size", mean_score="mean")
        .round(2).to_string())
    low = r.assign(low=r["review_score"] <= 2).groupby(TARGET)["low"].mean() * 100
    log("Share of reviews with 1-2 stars (%):")
    log(low.round(2).to_string())


# --------------------------------------------------------------------------
def multivariate(df, overall) -> None:
    section("3. MULTIVARIATE ANALYSIS")

    corr = df[NUMERIC_FEATURES + [TARGET]].corr(numeric_only=True)
    log("Correlation of each numeric feature with is_late:")
    log(corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        .round(3).to_string())
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", annot_kws={"size": 7},
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation matrix (purchase-time numeric features + target)")
    save_fig(fig, "22_correlation_heatmap.png")

    # distance x promised delivery window
    d_bin, d_edges = quantile_bins(df["distance_km"], 5)
    e_bin, e_edges = quantile_bins(df["estimated_delivery_days"], 5)
    log(f"\nDistance quintile edges (km): {np.round(d_edges, 0).tolist()}")
    log(f"Estimated-days quintile edges: {np.round(e_edges, 0).tolist()}")
    pivot = df.assign(dist_q=d_bin, est_q=e_bin).pivot_table(
        index="dist_q", columns="est_q", values=TARGET, aggfunc="mean") * 100
    log("\nLate rate (%) by distance quintile (rows) x promised-days quintile (columns):")
    log(pivot.round(2).to_string())
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("Promised delivery days (quintile, 1 = shortest)")
    ax.set_ylabel("Distance (quintile, 1 = closest)")
    ax.set_title("Late rate (%)")
    save_fig(fig, "23_distance_x_promised_days.png")

    # seller state -> customer state routes (cells with < 30 orders hidden)
    top_c = df["customer_state"].value_counts().head(8).index
    top_s = df["seller_state"].value_counts().head(8).index
    sub = df[df["customer_state"].isin(top_c) & df["seller_state"].isin(top_s)]
    counts = pd.crosstab(sub["seller_state"], sub["customer_state"])
    rate = pd.crosstab(sub["seller_state"], sub["customer_state"],
                       values=sub[TARGET], aggfunc="mean") * 100
    rate = rate.where(counts >= 30)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(rate, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    ax.set_title("Late rate (%) by route (cells with < 30 orders hidden)")
    save_fig(fig, "24_route_heatmap.png")

    # sellers
    seller = df.groupby("seller_id")[TARGET].agg(orders="size", late_orders="sum")
    seller["late_rate_pct"] = (seller["late_orders"] / seller["orders"] * 100).round(2)
    big = seller[seller["orders"] >= 30]
    log(f"\nSellers in total: {len(seller):,}; with >= 30 orders: {len(big):,}")
    log("Seller late rate (%), sellers with >= 30 orders:")
    log(big["late_rate_pct"].describe().round(2).to_string())
    log("\nTen highest late rates among sellers with >= 30 orders:")
    worst = big.sort_values("late_rate_pct", ascending=False).head(10).copy()
    worst.index = worst.index.str[:8] + "..."
    log(worst.to_string())
    share = big["late_orders"].sum() / df[TARGET].sum() * 100
    log(f"\nSellers with >= 30 orders account for {share:.1f}% of all late orders")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(big["late_rate_pct"], bins=30, color=BLUE, ax=ax)
    ax.axvline(overall, color="red", ls="--", label=f"overall {overall:.1f}%")
    ax.set_title("Seller late rate (sellers with >= 30 orders)")
    ax.set_xlabel("Late rate (%)")
    ax.legend()
    save_fig(fig, "25_seller_late_rate_distribution.png")


def main() -> None:
    df = load_base_table()
    overall = df[TARGET].mean() * 100
    log(f"Orders analysed: {len(df):,} | overall late rate: {overall:.2f}%")
    univariate(df)
    bivariate(df, overall)
    multivariate(df, overall)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    log(f"\nFindings saved to: {REPORT_PATH}")
    log(f"Figures saved in:  {FIGURES_DIR}")


if __name__ == "__main__":
    main()