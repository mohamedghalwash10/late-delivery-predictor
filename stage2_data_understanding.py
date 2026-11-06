"""Stage 2: understand the Olist data. Prints and saves a report; builds no features."""
import pandas as pd

from src.config import PROJECT_ROOT
from src.data_loader import load_all

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

REPORT_PATH = PROJECT_ROOT / "reports" / "stage2_data_understanding.txt"
_lines: list[str] = []


def log(text="") -> None:
    """Print a line and keep it for the saved report."""
    text = str(text)
    print(text)
    _lines.append(text)


def section(title: str) -> None:
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


def overview(tables) -> None:
    section("1. TABLE OVERVIEW")
    rows = [
        {
            "table": name,
            "rows": len(df),
            "columns": df.shape[1],
            "memory_MB": round(df.memory_usage(deep=True).sum() / 1_048_576, 1),
            "duplicate_rows": int(df.duplicated().sum()),
        }
        for name, df in tables.items()
    ]
    log(pd.DataFrame(rows).to_string(index=False))


def column_details(tables) -> None:
    section("2. COLUMNS, DATA TYPES AND MISSING VALUES")
    for name, df in tables.items():
        log(f"\n--- {name} ({len(df):,} rows) ---")
        info = pd.DataFrame({
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        })
        log(info.to_string())


def order_status(orders) -> None:
    section("3. ORDER STATUS AND DATE RANGE")
    counts = orders["order_status"].value_counts()
    log(pd.DataFrame({
        "orders": counts,
        "pct": (counts / len(orders) * 100).round(2),
    }).to_string())
    first = orders["order_purchase_timestamp"].min()
    last = orders["order_purchase_timestamp"].max()
    log(f"\nPurchase dates: {first} -> {last}")


def key_coverage(child, child_col, parent, parent_col, label) -> None:
    matched = child[child_col].dropna().isin(parent[parent_col]).mean() * 100
    unique = parent[parent_col].is_unique
    log(f"{label:<58} matched {matched:6.2f}%   parent key unique: {unique}")


def relationships(t) -> None:
    section("4. TABLE RELATIONSHIPS (do the join keys really match?)")
    key_coverage(t["orders"], "customer_id", t["customers"], "customer_id",
                 "orders.customer_id -> customers.customer_id")
    key_coverage(t["items"], "order_id", t["orders"], "order_id",
                 "items.order_id -> orders.order_id")
    key_coverage(t["payments"], "order_id", t["orders"], "order_id",
                 "payments.order_id -> orders.order_id")
    key_coverage(t["reviews"], "order_id", t["orders"], "order_id",
                 "reviews.order_id -> orders.order_id")
    key_coverage(t["items"], "product_id", t["products"], "product_id",
                 "items.product_id -> products.product_id")
    key_coverage(t["items"], "seller_id", t["sellers"], "seller_id",
                 "items.seller_id -> sellers.seller_id")
    key_coverage(t["products"], "product_category_name",
                 t["category_translation"], "product_category_name",
                 "products.category -> category_translation")
    key_coverage(t["customers"], "customer_zip_code_prefix",
                 t["geolocation"], "geolocation_zip_code_prefix",
                 "customers.zip_prefix -> geolocation.zip_prefix")
    key_coverage(t["sellers"], "seller_zip_code_prefix",
                 t["geolocation"], "geolocation_zip_code_prefix",
                 "sellers.zip_prefix -> geolocation.zip_prefix")


def cardinality(t) -> None:
    section("5. CARDINALITY (rows per order)")
    orders, items, customers = t["orders"], t["items"], t["customers"]
    log(f"orders.order_id unique: {orders['order_id'].is_unique}")
    log(f"customers.customer_id unique: {customers['customer_id'].is_unique}")
    log(f"distinct customer_unique_id: {customers['customer_unique_id'].nunique():,} "
        f"of {len(customers):,} customer rows")
    no_items = int((~orders["order_id"].isin(items["order_id"])).sum())
    log(f"orders with no item rows: {no_items:,}")

    per_order = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
    )
    log("\nItems per order:")
    log(per_order["n_items"].describe().round(2).to_string())
    multi = per_order["n_sellers"] > 1
    log(f"\nOrders with more than one seller: {int(multi.sum()):,} "
        f"({multi.mean() * 100:.2f}% of orders that have items)")
    log(f"Max payment rows for one order: {t['payments'].groupby('order_id').size().max()}")
    log(f"Max review rows for one order:  {t['reviews'].groupby('order_id').size().max()}")


def target_and_balance(orders) -> None:
    section("6. TARGET VARIABLE AND CLASS BALANCE")
    delivered = orders[orders["order_status"] == "delivered"]
    log(f"Orders in total:      {len(orders):,}")
    log(f"Delivered orders:     {len(delivered):,}")

    date_cols = ["order_delivered_customer_date", "order_estimated_delivery_date"]
    no_date = int(delivered["order_delivered_customer_date"].isna().sum())
    log(f"Delivered but no delivery date (cannot be labelled): {no_date:,}")

    labelled = delivered.dropna(subset=date_cols).copy()
    log(f"Labelled orders (used for modelling): {len(labelled):,}")

    delivered_day = labelled["order_delivered_customer_date"].dt.normalize()
    estimated_day = labelled["order_estimated_delivery_date"].dt.normalize()
    late_by_day = delivered_day > estimated_day
    late_by_time = (labelled["order_delivered_customer_date"]
                    > labelled["order_estimated_delivery_date"])

    log("\nClass balance under two possible definitions:")
    for label, series in [("Calendar-day comparison (chosen)", late_by_day),
                          ("Exact timestamp comparison", late_by_time)]:
        n_late = int(series.sum())
        n_on_time = len(series) - n_late
        log(f"  {label}")
        log(f"    on-time (0): {n_on_time:>7,}  ({n_on_time / len(series) * 100:5.2f}%)")
        log(f"    late    (1): {n_late:>7,}  ({n_late / len(series) * 100:5.2f}%)")

    days_diff = (delivered_day - estimated_day).dt.days
    log("\nDays between actual and estimated delivery (negative = early):")
    log(days_diff.describe().round(2).to_string())

    section("7. DATE SANITY CHECKS (on labelled orders)")
    before_purchase = (labelled["order_delivered_customer_date"]
                       < labelled["order_purchase_timestamp"]).sum()
    before_carrier = (labelled["order_delivered_customer_date"]
                      < labelled["order_delivered_carrier_date"]).sum()
    est_before_purchase = (labelled["order_estimated_delivery_date"]
                           < labelled["order_purchase_timestamp"]).sum()
    log(f"Delivered before purchase:           {int(before_purchase):,}")
    log(f"Delivered before carrier pickup:     {int(before_carrier):,}")
    log(f"Estimated date before purchase date: {int(est_before_purchase):,}")


def main() -> None:
    tables = load_all()
    overview(tables)
    column_details(tables)
    order_status(tables["orders"])
    relationships(tables)
    cardinality(tables)
    target_and_balance(tables["orders"])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(_lines), encoding="utf-8")
    log(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()