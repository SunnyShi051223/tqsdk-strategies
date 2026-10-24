#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cache daily mainline bars and their historical tradable-contract mapping."""

from datetime import date
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth
from strategies.config_local import TQ_ACCOUNT, TQ_PASSWORD


START = date(2019, 1, 1)
END = date(2026, 10, 10)
DATA_LENGTH = 3000
MAPPING_PATH = Path(__file__).resolve().parent / "backtest_results" / "_data" / "historical_main_contracts_2019-2026-10-10.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "backtest_results" / "_data" / "daily_mainline_bars_2019-2026-10-10.csv"
FIELDS = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]


def main():
    mapping = pd.read_csv(MAPPING_PATH, dtype={"product": str, "mainline": str, "actual_contract": str})
    symbols = sorted(mapping["mainline"].unique())
    api = TqApi(auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD))
    try:
        series = {symbol: api.get_kline_serial(symbol, 86400, data_length=DATA_LENGTH) for symbol in symbols}
        while not all(api.is_serial_ready(frame) for frame in series.values()):
            api.wait_update()
        frames = []
        for symbol, frame in series.items():
            current = frame.loc[frame["datetime"].gt(0), ["datetime"] + FIELDS].copy()
            current["trading_date"] = pd.to_datetime(current.pop("datetime"), unit="ns", utc=True).dt.tz_convert("Asia/Shanghai").dt.date.astype(str)
            current = current[(current["trading_date"] >= START.isoformat()) & (current["trading_date"] <= END.isoformat())]
            current["mainline"] = symbol
            frames.append(current)
    finally:
        api.close()

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.merge(mapping[["trading_date", "mainline", "actual_contract"]], on=["trading_date", "mainline"], how="left", validate="one_to_one")
    bars = bars.sort_values(["mainline", "trading_date"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    gaps = bars["actual_contract"].isna().groupby(bars["mainline"]).sum()
    print(f"symbols={len(symbols)} rows={len(bars)} first={bars.trading_date.min()} last={bars.trading_date.max()}")
    print(f"mapping_gaps={int(gaps.sum())} output={OUTPUT_PATH}")
    if gaps.any():
        print("gaps_by_mainline=" + repr(gaps[gaps.gt(0)].to_dict()))


if __name__ == "__main__":
    main()
