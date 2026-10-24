#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export current futures products and their day-by-day main-contract mapping."""

from __future__ import annotations

import argparse
import csv
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth

from strategies.config_local import TQ_ACCOUNT, TQ_PASSWORD


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "future_list.csv"
HISTORY_PATH = ROOT / "Backtest" / "_data" / "main_contract_history.csv"
CONTRACT_RE = re.compile(r"^([^.]+)\.([A-Za-z]+?)\d{2,4}$")
CATALOG_FIELDS = [
    "product",
    "exchange",
    "product_id",
    "mainline_symbol",
    "representative_contract",
    "first_mapped_date",
    "last_mapped_date",
    "contract_count",
    "updated_at",
]
HISTORY_FIELDS = ["trading_date", "product", "mainline_symbol", "actual_contract"]


def _canonical_product(value: str) -> str:
    exchange, product_id = value.split(".", 1)
    return f"{exchange.upper()}.{product_id}"


def _parse_catalog(contracts) -> list[dict[str, str]]:
    products: dict[str, dict[str, str]] = {}
    for contract in contracts:
        match = CONTRACT_RE.fullmatch(str(contract))
        if not match:
            continue
        exchange, product_id = match.groups()
        product = _canonical_product(f"{exchange}.{product_id}")
        products[product] = {
            "product": product,
            "exchange": exchange.upper(),
            "product_id": product_id,
            "mainline_symbol": f"KQ.m@{exchange}.{product_id}",
            "representative_contract": str(contract),
            "first_mapped_date": "",
            "last_mapped_date": "",
            "contract_count": "",
        }
    return [products[key] for key in sorted(products, key=str.upper)]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _history_length(start: date, end: date, configured: int | None) -> int:
    # query_his_cont_quotes returns the most recent N trading days, so account
    # for the days between the requested start and today as well as the range.
    today = date.today()
    required = math.ceil(max(0, (today - start).days) * 5 / 7) + 100
    return max(configured or 0, required, 200)


def export_catalog_and_history(
    *, products: list[str], start: date, end: date, history_days: int | None = None
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if start > end:
        raise ValueError("--start must be earlier than or equal to --end")
    if not TQ_ACCOUNT.strip() or not TQ_PASSWORD:
        raise RuntimeError("请先在 strategies/config_local.py 中填写快期账号配置。")

    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
    try:
        catalog = _parse_catalog(api.query_cont_quotes())
        if not catalog:
            raise RuntimeError("TqSdk 没有返回可用期货主连品种。")

        by_product = {row["product"].upper(): row for row in catalog}
        selected = catalog if "ALL" in {p.upper() for p in products} else []
        if not selected:
            for requested in products:
                key = _canonical_product(requested).upper()
                row = by_product.get(key)
                if row is None:
                    raise ValueError(f"future_list 中没有找到期货品种: {requested}")
                selected.append(row)

        symbols = [row["mainline_symbol"] for row in selected]
        n = _history_length(start, end, history_days)
        frame = api.query_his_cont_quotes(symbols, n=n)
    finally:
        api.close()

    trading_dates = pd.to_datetime(frame["date"]).dt.date
    date_mask = (trading_dates >= start) & (trading_dates <= end)
    frame = frame.loc[date_mask].reset_index(drop=True)
    trading_dates = trading_dates.loc[date_mask].reset_index(drop=True)

    history_rows: list[dict[str, str]] = []
    selected_by_symbol = {row["mainline_symbol"]: row for row in selected}
    for symbol, catalog_row in selected_by_symbol.items():
        values = frame[symbol].fillna("").astype(str).str.strip()
        valid = values.ne("")
        mapped_dates = trading_dates.loc[valid]
        catalog_row["first_mapped_date"] = mapped_dates.min().isoformat() if len(mapped_dates) else ""
        catalog_row["last_mapped_date"] = mapped_dates.max().isoformat() if len(mapped_dates) else ""
        catalog_row["contract_count"] = str(values.loc[valid].nunique())
        for trading_date, contract in zip(trading_dates, values):
            if contract:
                history_rows.append(
                    {
                        "trading_date": trading_date.isoformat(),
                        "product": catalog_row["product"],
                        "mainline_symbol": symbol,
                        "actual_contract": contract,
                    }
                )

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    existing_catalog = {row["product"].upper(): row for row in _read_csv(CATALOG_PATH)}
    for row in catalog:
        old = existing_catalog.get(row["product"].upper(), {})
        for field in ("first_mapped_date", "last_mapped_date", "contract_count"):
            if not row[field]:
                row[field] = old.get(field, "")
        row["updated_at"] = now
    _write_csv(CATALOG_PATH, CATALOG_FIELDS, catalog)

    selected_products = {row["product"].upper() for row in selected}
    preserved = [
        row
        for row in _read_csv(HISTORY_PATH)
        if row.get("product", "").upper() not in selected_products
    ]
    merged_history = preserved + history_rows
    merged_history.sort(key=lambda row: (row["product"].upper(), row["trading_date"]))
    _write_csv(HISTORY_PATH, HISTORY_FIELDS, merged_history)

    return catalog, history_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 future_list.csv 和按交易日变化的主连—实际合约映射。"
    )
    parser.add_argument(
        "--product",
        default="all",
        help="品种代码，例如 DCE.m；默认 all。",
    )
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="最低请求交易日数量；默认按开始日期自动计算。",
    )
    args = parser.parse_args()

    catalog, history = export_catalog_and_history(
        products=[args.product],
        start=args.start,
        end=args.end,
        history_days=args.history_days,
    )
    print(f"future_list={CATALOG_PATH} products={len(catalog)}")
    print(f"main_contract_history={HISTORY_PATH} rows_updated={len(history)}")
    if args.product.lower() != "all":
        product = _canonical_product(args.product).upper()
        matched = [row for row in catalog if row["product"].upper() == product]
        if matched:
            row = matched[0]
            print(
                f"product={row['product']} mainline={row['mainline_symbol']} "
                f"coverage={row['first_mapped_date']}..{row['last_mapped_date']} "
                f"contracts={row['contract_count']}"
            )


if __name__ == "__main__":
    main()
