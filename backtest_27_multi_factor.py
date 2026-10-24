#!/usr/bin/env python3
"""Offline historical backtest adapter for strategies/27_multi_factor_ranking.py."""

import json
from datetime import date

import numpy as np
from tqsdk import BacktestFinished, TqApi, TqAuth, TqBacktest, TqSim, TargetPosTask
from strategies.config_local import TQ_ACCOUNT, TQ_PASSWORD


SYMBOLS = [
    "SHFE.rb2501", "SHFE.hc2501", "DCE.i2501", "DCE.jm2501", "DCE.j2501"
]
KLINE_DUR = 86400
LOOKBACK = 20
REBALANCE_BARS = 10
TOP_N = 1
BOT_N = 1
VOLUME = 1
FACTOR_WEIGHTS = {"momentum": 0.4, "volatility": 0.3, "trend": 0.3}
BACKTEST_START = date(2024, 1, 1)
BACKTEST_END = date(2025, 1, 31)


def calc_momentum(kl, period):
    close = kl["close"].iloc[:-1]
    if len(close) < period + 1:
        return np.nan
    return (close.iloc[-1] - close.iloc[-period]) / close.iloc[-period]


def calc_volatility(kl, period):
    close = kl["close"].iloc[:-1]
    if len(close) < period + 1:
        return np.nan
    rets = close.pct_change().dropna()
    if len(rets) < period:
        return np.nan
    vol = rets.iloc[-period:].std()
    return np.nan if vol == 0 else -1.0 / vol


def calc_trend_strength(kl, period):
    close = kl["close"].iloc[:-1]
    if len(close) < period + 1:
        return np.nan
    window = close.iloc[-period:]
    mean = window.mean()
    return np.nan if mean == 0 else ((window - mean) / mean).std()


def normalize(values):
    valid = [v for v in values.values() if v is not None and not np.isnan(v)]
    if len(valid) < 2:
        return {k: 0.5 for k in values}
    mean, std = np.mean(valid), np.std(valid)
    if std == 0:
        return {k: 0.5 for k in values}
    return {k: (v - mean) / std if v is not None and not np.isnan(v) else 0.5 for k, v in values.items()}


def main():
    api = TqApi(
        account=TqSim(),
        backtest=TqBacktest(start_dt=BACKTEST_START, end_dt=BACKTEST_END),
        auth=TqAuth(TQ_ACCOUNT, TQ_PASSWORD),
    )
    klines = {s: api.get_kline_serial(s, KLINE_DUR, data_length=LOOKBACK + 5) for s in SYMBOLS}
    # The source strategy uses manual orders. For a historical run, target
    # positions make the same long/short intent while allowing TqSdk to manage
    # close/open sequencing and valid active prices across session boundaries.
    target_tasks = {s: TargetPosTask(api, s, price="ACTIVE") for s in SYMBOLS}
    long_set, short_set = set(), set()
    last_day = None
    bars_seen = 0
    rebalance_count = 0
    ranking_log = []
    try:
        while True:
            api.wait_update()
            if not any(api.is_changing(klines[s].iloc[-1], "datetime") for s in SYMBOLS):
                continue
            completed_days = [klines[s]["datetime"].iloc[-2] for s in SYMBOLS]
            # A fixed historical contract can stop updating before the others expire.
            # Use the newest completed bar to let the backtest reach its requested end.
            day = max(completed_days)
            if last_day == day:
                continue
            last_day = day
            bars_seen += 1
            if bars_seen < LOOKBACK or bars_seen % REBALANCE_BARS:
                continue
            factor_scores = {}
            for sym, kl in klines.items():
                factor_scores[sym] = {
                    "momentum": calc_momentum(kl, LOOKBACK),
                    "volatility": calc_volatility(kl, LOOKBACK),
                    "trend": calc_trend_strength(kl, LOOKBACK),
                }
            mom = normalize({s: v["momentum"] for s, v in factor_scores.items()})
            vol = normalize({s: v["volatility"] for s, v in factor_scores.items()})
            trend = normalize({s: v["trend"] for s, v in factor_scores.items()})
            composite = {
                s: FACTOR_WEIGHTS["momentum"] * mom[s]
                + FACTOR_WEIGHTS["volatility"] * vol[s]
                + FACTOR_WEIGHTS["trend"] * trend[s]
                for s in SYMBOLS
            }
            ranked = sorted(composite.items(), key=lambda x: x[1], reverse=True)
            new_long = {s for s, _ in ranked[:TOP_N]}
            new_short = {s for s, _ in ranked[-BOT_N:]}
            ranking_log.append({"date": int(day), "ranking": ranked, "long": sorted(new_long), "short": sorted(new_short)})
            for sym in SYMBOLS:
                target = VOLUME if sym in new_long else -VOLUME if sym in new_short else 0
                target_tasks[sym].set_target_volume(target)
            long_set, short_set = new_long, new_short
            rebalance_count += 1
    except BacktestFinished:
        account_info = api.get_account()
        result = {
            "strategy": "27_multi_factor_ranking",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "symbols": SYMBOLS,
            "rebalance_count": rebalance_count,
            "ranking_log": ranking_log,
            "balance": float(account_info.balance),
            "available": float(account_info.available),
            "float_profit": float(account_info.float_profit),
            "position_profit": float(account_info.position_profit),
            "risk_ratio": float(account_info.risk_ratio),
        }
        print("BACKTEST_RESULT=" + json.dumps(result, ensure_ascii=False))
    finally:
        api.close()


if __name__ == "__main__":
    main()
