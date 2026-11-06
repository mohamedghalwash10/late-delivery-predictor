"""Stage 3a: build the order-level table, validate it and save it."""
import pandas as pd

from src.build_dataset import (
    ANALYSIS_ONLY, BASE_PATH, FEATURE_COLUMNS, TARGET, build_order_table,
)

pd.set_option("display.width", 200)


def main() -> None:
    df = build_order_table()
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(BASE_PATH, index=False)

    print("\n" + "=" * 70 + "\nORDER-LEVEL TABLE\n" + "=" * 70)
    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"order_id unique: {df['order_id'].is_unique}")
    counts = df[TARGET].value_counts().sort_index()
    print(f"On-time (0): {counts[0]:,}   Late (1): {counts[1]:,}   "
          f"Late rate: {df[TARGET].mean() * 100:.2f}%")

    print("\n" + "=" * 70 + "\nMISSING VALUES IN PURCHASE-TIME FEATURES\n" + "=" * 70)
    miss = pd.DataFrame({
        "missing": df[FEATURE_COLUMNS].isna().sum(),
        "missing_pct": (df[FEATURE_COLUMNS].isna().mean() * 100).round(2),
    })
    print(miss[miss["missing"] > 0].to_string()
          if (miss["missing"] > 0).any() else "None")

    print("\n" + "=" * 70 + "\nLEAKAGE GUARD\n" + "=" * 70)
    print(f"Allowed model features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Analysis-only, never features ({len(ANALYSIS_ONLY)}): {ANALYSIS_ONLY}")
    overlap = set(FEATURE_COLUMNS) & set(ANALYSIS_ONLY)
    print(f"Overlap between the two groups: {overlap or 'none'}")

    size_mb = BASE_PATH.stat().st_size / 1_048_576
    print(f"\nSaved: {BASE_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()