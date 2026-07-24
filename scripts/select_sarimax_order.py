"""AIC/BIC-based (p,d,q) order search for SARIMAX, per vegetable.

This is the piece that was actually missing: the SARIMAX order in configs/config.yaml
was fixed to (1,1,1) from literature precedent, not chosen via AIC. This performs the
standard Box-Jenkins order search - fit every candidate (p,1,q) on the full series
(seasonal order held fixed at the already-stability-tested (1,0,1,52) from
configs/config.yaml), compare AIC/BIC/HQIC, and report the AIC-minimizing order per
vegetable. d=1 is held fixed (regular differencing was already established as
necessary - see CLAUDE.md's SARIMAX gotchas).

Usage:
    python scripts/select_sarimax_order.py
"""

import sys
import warnings
from itertools import product
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.sarimax.train import _fit
from src.utils.io import PROJECT_ROOT, get_processed_path, load_config

P_RANGE = (0, 1, 2)
Q_RANGE = (0, 1, 2)
D_FIXED = 1


def search_vegetable(vegetable: str, config: dict) -> pd.DataFrame:
    df = pd.read_csv(get_processed_path(vegetable, config), parse_dates=["date"])
    df = df.rename(columns={"retail_price": "target"}) if "target" not in df.columns else df
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])

    rows = []
    for p, q in product(P_RANGE, Q_RANGE):
        order = (p, D_FIXED, q)
        try:
            res = _fit(df, config, order=order, seasonal_order=seasonal_order)
            rows.append(
                {
                    "vegetable": vegetable,
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "aic": res.aic,
                    "bic": res.bic,
                    "hqic": res.hqic,
                    "converged": bool(res.mle_retvals.get("converged", False)),
                }
            )
            print(f"  {vegetable} order={order}: AIC={res.aic:.1f} BIC={res.bic:.1f}")
        except Exception as e:
            print(f"  {vegetable} order={order}: FAILED ({e})")
    return pd.DataFrame(rows)


def main():
    config = load_config()
    all_rows = []
    for vegetable in config["vegetables"]:
        print(f"=== {vegetable} ===")
        all_rows.append(search_vegetable(vegetable, config))

    results = pd.concat(all_rows, ignore_index=True)
    out_path = PROJECT_ROOT / "results" / "tables" / "sarimax_order_selection.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\nSaved {len(results)} rows to {out_path}")

    best = results.loc[results.groupby("vegetable")["aic"].idxmin()]
    print("\nAIC-selected order per vegetable:")
    print(best[["vegetable", "order", "seasonal_order", "aic", "bic"]].to_string(index=False))

    configured_order = str(tuple(config["models"]["sarimax"]["order"]))
    print(f"\nCurrently configured order (configs/config.yaml): {configured_order}")
    mismatches = best[best["order"] != configured_order]
    if len(mismatches):
        print(f"{len(mismatches)}/{len(best)} vegetables have an AIC-preferred order different from the configured one:")
        print(mismatches[["vegetable", "order", "aic"]].to_string(index=False))
    else:
        print("All vegetables' AIC-preferred order matches the configured order.")


if __name__ == "__main__":
    main()
