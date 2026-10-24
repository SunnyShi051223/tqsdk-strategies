#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export historical main-contract mappings for the multi-factor strategy set."""

import csv
import re
from datetime import date
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth
from strategies.config_local import TQ_ACCOUNT, TQ_PASSWORD


START = date(2019, 1, 1)
END = date(2026, 10, 10)
MAX_HISTORY_DAYS = 3000
STRATEGIES = [
    "27_elder_triple_screen.py", "27_multi_factor_ranking.py", "46_multi_factor.py",
    "48_sector_rotation_multi_factor.py", "49_mean_variance_portfolio.py",
    "50_volatility_skew.py", "52_multi_factor_ai_prediction.py",
    "54_multi_asset_long_short_hedge.py", "55_momentum_value_factor_composite.py",
    "56_cross_section_multi_factor_ranking.py", "58_money_flow_rotation.py",
    "59_vol_momentum_composite.py", "60_cross_section_ml_ranking.py",
    "62_macro_factor_rotation.py", "79_cross_section_risk_parity.py",
    "80_momentum_carry_composite.py", "81_term_structure_momentum_rotation.py",
    "82_liquidity_adjusted_momentum.py", "83_regime_switching_allocation.py",
    "91_volatility_target_position.py", "93_adaptive_position_sizing.py",
]
CONTRACT_RE = re.compile(r"\b(SHFE|DCE|CZCE|INE|CFFEX)\.([A-Za-z]+)\d{2,4}\b")
PRODUCT_ALIASES = {
    # A few examples in the repository use the wrong/obsolete exchange prefix.
    "CZCE.JD": "DCE.jd",
    "CZCE.P": "DCE.p",
    "CZCE.cs": "DCE.cs",
    "CZCE.mo": "CZCE.OI",  # Repository comment identifies this as rapeseed oil.
    "CZCE.LK": "DCE.l",  # Repository sample uses LK for DCE LLDPE.
}


def requested_products(strategy_dir: Path):
    products = set()
    for filename in STRATEGIES:
        source = (strategy_dir / filename).read_text(encoding="utf-8")
        products.update(f"{exchange}.{product}" for exchange, product in CONTRACT_RE.findall(source))
    return sorted(products)


def main():
    root = Path(__file__).resolve().parent
    original_products = requested_products(root / "strategies")
    products_by_key = {}
    for product in original_products:
        canonical = PRODUCT_ALIASES.get(product, product)
        exchange, product_id = canonical.split(".", 1)
        products_by_key[(exchange.upper(), product_id.upper())] = canonical
    products = sorted(products_by_key.values())
    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
    try:
        all_conts = api.query_cont_quotes()
        cont_by_product = {}
        for symbol in all_conts:
            # In the installed SDK, query_cont_quotes() returns the current
            # delivery-month representative (e.g. SHFE.rb2610), while
            # query_his_cont_quotes() expects the corresponding KQ.m code.
            match = re.fullmatch(r"([^.]+)\.([A-Za-z]+)\d+", symbol)
            if match:
                cont_by_product[(match.group(1).upper(), match.group(2).upper())] = (
                    f"KQ.m@{match.group(1)}.{match.group(2)}"
                )

        matches = {}
        missing = []
        for key, product in products_by_key.items():
            cont = cont_by_product.get(key)
            if cont:
                # Normalize to the exact exchange/product spelling used by the
                # historical continuous-contract endpoint.
                exact_product = cont.removeprefix("KQ.m@")
                matches[exact_product] = cont
            else:
                missing.append(product)
        if not matches:
            raise RuntimeError("No strategy products matched TqSdk continuous contracts")

        unique_conts = sorted(set(matches.values()))
        frame = api.query_his_cont_quotes(unique_conts, n=MAX_HISTORY_DAYS)
    finally:
        api.close()

    date_col = pd.to_datetime(frame["date"]).dt.date
    mask = (date_col >= START) & (date_col <= END)
    frame = frame.loc[mask].copy()
    date_col = date_col.loc[mask]
    rows = []
    coverage = []
    for product, cont in matches.items():
        values = frame[cont]
        valid = values.astype(str).str.len().gt(0)
        mapped_dates = date_col.loc[valid]
        coverage.append({
            "product": product,
            "mainline": cont,
            "mapped_days": int(valid.sum()),
            "first_mapped_date": mapped_dates.iloc[0].isoformat() if len(mapped_dates) else "",
            "last_mapped_date": mapped_dates.iloc[-1].isoformat() if len(mapped_dates) else "",
            "contracts": int(values.loc[valid].nunique()),
        })
        for dt, contract in zip(date_col, values):
            contract = str(contract)
            if contract:
                rows.append((dt.isoformat(), product, cont, contract))

    out_dir = root / "backtest_results" / "_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = out_dir / "historical_main_contracts_2019-2026-10-10.csv"
    coverage_file = out_dir / "historical_main_contract_coverage.csv"
    missing_file = out_dir / "unmatched_strategy_products.txt"
    with mapping_file.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trading_date", "product", "mainline", "actual_contract"])
        writer.writerows(rows)
    pd.DataFrame(coverage).to_csv(coverage_file, index=False, encoding="utf-8-sig")
    missing_keys = {tuple(part.upper() for part in product.split(".", 1)) for product in missing}
    unmatched_original = sorted({
        p for p in original_products
        if tuple(part.upper() for part in PRODUCT_ALIASES.get(p, p).split(".", 1)) in missing_keys
    })
    missing_file.write_text("\n".join(unmatched_original) + ("\n" if unmatched_original else ""), encoding="utf-8")

    print(f"products_requested={len(original_products)} canonical_products={len(products)} products_mapped={len(matches)} products_unmatched={len(unmatched_original)}")
    print(f"mapping_rows={len(rows)} dates={date_col.iloc[0] if len(date_col) else ''}..{date_col.iloc[-1] if len(date_col) else ''}")
    print(f"mapping_file={mapping_file}")
    print(f"coverage_file={coverage_file}")
    if unmatched_original:
        print("unmatched=" + ",".join(unmatched_original))
    expected_first = date(2019, 1, 2)  # Jan 1 was a holiday
    expected_last = date(2026, 10, 9)  # Oct 10 was Saturday
    for item in coverage:
        if not item["mapped_days"] or item["first_mapped_date"] > expected_first.isoformat() or item["last_mapped_date"] < expected_last.isoformat():
            print("coverage_gap=" + repr(item))


if __name__ == "__main__":
    main()
