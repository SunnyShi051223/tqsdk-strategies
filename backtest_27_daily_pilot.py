#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline daily research backtest for strategy 27 using cached mainline bars."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "backtest_results" / "_data" / "daily_mainline_bars_2019-2026-10-10.csv"
OUT = ROOT / "backtest_results" / "27_multi_factor_ranking"
START = "2019-01-01"
END = "2026-10-10"
PRODUCTS = ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j"]
WEIGHTS = {"momentum": 0.4, "volatility": 0.3, "trend": 0.3}
LOOKBACK = 20
REBALANCE_BARS = 10
TOP_N = 1
BOT_N = 1
BASELINE_COST_BPS = 5.0  # per one-way notional traded; generic proxy, not exchange fee schedule


def zscore_cross_section(values: pd.Series) -> pd.Series:
    valid = values.dropna()
    out = pd.Series(0.5, index=values.index, dtype=float)
    if len(valid) < 2:
        return out
    std = valid.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return out
    out.loc[valid.index] = (valid - valid.mean()) / std
    return out


def factor_values(close: pd.DataFrame, day_index: int) -> pd.DataFrame:
    """Use data through the signal close only; formulas follow strategy 27."""
    rows = {}
    for symbol in close.columns:
        hist = close[symbol].iloc[: day_index + 1].dropna()
        if len(hist) < LOOKBACK + 1:
            rows[symbol] = {name: np.nan for name in WEIGHTS}
            continue
        # The source strategy measures the close against the close 20 bars earlier.
        momentum = hist.iloc[-1] / hist.iloc[-(LOOKBACK + 1)] - 1
        returns = hist.pct_change().dropna().iloc[-LOOKBACK:]
        vol = returns.std(ddof=1)
        volatility = np.nan if not np.isfinite(vol) or vol == 0 else -1.0 / vol
        window = hist.iloc[-LOOKBACK:]
        mean = window.mean()
        trend = np.nan if mean == 0 else ((window - mean) / mean).std(ddof=1)
        rows[symbol] = {"momentum": momentum, "volatility": volatility, "trend": trend}
    return pd.DataFrame.from_dict(rows, orient="index").reindex(close.columns)


def rank_and_weight(factors: pd.DataFrame, omit_factor: str | None = None) -> tuple[dict[str, float], dict]:
    normalized = {name: zscore_cross_section(factors[name]) for name in WEIGHTS}
    active = {name: weight for name, weight in WEIGHTS.items() if name != omit_factor}
    total_weight = sum(active.values())
    score = sum(normalized[name] * (weight / total_weight) for name, weight in active.items())
    score = score.replace([np.inf, -np.inf], np.nan).dropna()
    if len(score) < TOP_N + BOT_N:
        return {}, {"scores": score.to_dict(), "normalized": {n: s.to_dict() for n, s in normalized.items()}}
    ranked = sorted(score.items(), key=lambda item: (-item[1], item[0]))
    longs = [symbol for symbol, _ in ranked[:TOP_N]]
    shorts = [symbol for symbol, _ in ranked[-BOT_N:]]
    positions = {symbol: 0.0 for symbol in factors.index}
    for symbol in longs:
        positions[symbol] = 0.5 / len(longs)
    for symbol in shorts:
        positions[symbol] = -0.5 / len(shorts)
    return positions, {
        "ranking": ranked,
        "long": longs,
        "short": shorts,
        "raw_factors": factors.to_dict(orient="index"),
        "normalized": {name: series.to_dict() for name, series in normalized.items()},
    }


def load_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bars = pd.read_csv(DATA, dtype={"trading_date": str, "mainline": str, "actual_contract": str})
    bars = bars[(bars.trading_date >= START) & (bars.trading_date <= END)]
    bars = bars[bars.mainline.isin([f"KQ.m@{product}" for product in PRODUCTS])].copy()
    bars["product"] = bars.mainline.str.replace("KQ.m@", "", regex=False)
    bars = bars.sort_values(["product", "trading_date"])
    idx = pd.Index(sorted(bars.trading_date.unique()), name="trading_date")
    def panel(col: str) -> pd.DataFrame:
        return bars.pivot(index="trading_date", columns="product", values=col).reindex(index=idx, columns=PRODUCTS)
    raw_open, raw_close = panel("open"), panel("close")
    contracts = panel("actual_contract")
    adjusted_open = raw_open.copy()
    adjusted_close = raw_close.copy()
    for product in PRODUCTS:
        scale = 1.0
        for i in range(len(idx)):
            if i and contracts[product].iloc[i] != contracts[product].iloc[i - 1]:
                # Chain the new contract at its open to the previous adjusted close.
                scale = adjusted_close[product].iloc[i - 1] / raw_open[product].iloc[i]
            adjusted_open.iloc[i, adjusted_open.columns.get_loc(product)] = raw_open[product].iloc[i] * scale
            adjusted_close.iloc[i, adjusted_close.columns.get_loc(product)] = raw_close[product].iloc[i] * scale
    return adjusted_open, adjusted_close, contracts, panel("volume")


def build_targets(close: pd.DataFrame, omit_factor: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    targets = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current = {symbol: 0.0 for symbol in close.columns}
    logs = []
    for i, day in enumerate(close.index):
        bar_count = i + 1
        if bar_count >= LOOKBACK and bar_count % REBALANCE_BARS == 0:
            factors = factor_values(close, i)
            new, details = rank_and_weight(factors, omit_factor=omit_factor)
            if new:
                current = new
                logs.append({"date": day, **details})
        targets.loc[day] = pd.Series(current)
    return targets, logs


def backtest(open_: pd.DataFrame, close: pd.DataFrame, contracts: pd.DataFrame,
             targets: pd.DataFrame, cost_bps: float = BASELINE_COST_BPS,
             keep_contributions: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    daily = []
    contrib_rows = []
    fee_rate = cost_bps / 10000.0
    previous_position = pd.Series(0.0, index=close.columns)
    previous_equity = 1.0
    previous_date = None
    for i, day in enumerate(close.index):
        position = targets.iloc[i - 1].copy() if i > 0 else pd.Series(0.0, index=close.columns)
        if i == 0:
            overnight = pd.Series(0.0, index=close.columns)
            intraday = pd.Series(0.0, index=close.columns)
            turnover_cost = pd.Series(0.0, index=close.columns)
        else:
            prev_close = close.iloc[i - 1]
            today_open = open_.iloc[i]
            today_close = close.iloc[i]
            same_contract = contracts.iloc[i].eq(contracts.iloc[i - 1]) & contracts.iloc[i].notna()
            overnight = (today_open / prev_close - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            overnight = overnight.where(same_contract, 0.0)
            intraday = (today_close / today_open - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            old_contract = contracts.iloc[i - 1]
            changed = contracts.iloc[i].ne(old_contract) | contracts.iloc[i].isna() | old_contract.isna()
            turnover = (position - previous_position).abs().where(~changed, position.abs() + previous_position.abs())
            turnover_cost = turnover.fillna(0.0) * fee_rate
        asset_contrib = previous_position * overnight + position * intraday - turnover_cost
        gross_ret = float((previous_position * overnight + position * intraday).sum())
        fees = float(turnover_cost.sum())
        net_ret = gross_ret - fees
        previous_equity *= max(0.0, 1.0 + net_ret)
        daily.append({
            "trading_date": day,
            "gross_return": gross_ret,
            "cost": fees,
            "net_return": net_ret,
            "equity": previous_equity,
            "gross_exposure": float(position.abs().sum()),
            "long_exposure": float(position.clip(lower=0).sum()),
            "short_exposure": float(-position.clip(upper=0).sum()),
        })
        if keep_contributions:
            contrib_rows.append(pd.Series(asset_contrib, name=day))
        previous_position = position
        previous_date = day
    result = pd.DataFrame(daily).set_index("trading_date")
    contrib = pd.DataFrame(contrib_rows).fillna(0.0) if keep_contributions else None
    if contrib is not None:
        contrib.index.name = "trading_date"
    return result, contrib


def drawdown(equity: pd.Series) -> float:
    path = pd.concat([pd.Series([1.0]), equity.reset_index(drop=True)], ignore_index=True)
    return float((path / path.cummax() - 1).min())


def metrics(daily: pd.DataFrame) -> dict:
    r = daily.net_return.astype(float)
    if len(r) == 0:
        return {"days": 0, "total_return": 0.0, "cagr": 0.0, "annualized_volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "total_cost": 0.0, "average_gross_exposure": 0.0}
    equity = (1 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else 0.0
    return {
        "days": int(len(r)),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown(equity),
        "win_rate": float((r > 0).mean()),
        "total_cost": float(daily.cost.sum()),
        "average_gross_exposure": float(daily.gross_exposure.mean()),
    }


def period_table(daily: pd.DataFrame) -> pd.DataFrame:
    periods = [("2019-2021开发期", "2019-01-01", "2021-12-31"),
               ("2022-2023验证期", "2022-01-01", "2023-12-31"),
               ("2024-2026样本外期", "2024-01-01", "2026-10-09")]
    out = []
    dates = pd.to_datetime(daily.index)
    for name, start, end in periods:
        sub = daily.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        out.append({"period": name, **metrics(sub)})
    return pd.DataFrame(out)


def factor_ic(close: pd.DataFrame, targets: pd.DataFrame, logs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for item in logs:
        i = close.index.get_loc(item["date"])
        end_i = min(i + REBALANCE_BARS, len(close.index) - 1)
        if end_i <= i:
            continue
        for symbol in close.columns:
            f = item.get("raw_factors", {}).get(symbol, {})
            fwd = close[symbol].iloc[end_i] / close[symbol].iloc[i] - 1
            for factor, value in f.items():
                if np.isfinite(value) and np.isfinite(fwd):
                    rows.append({"signal_date": item["date"], "symbol": symbol, "factor": factor, "factor_value": value, "forward_return": fwd})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    by_asset = []
    for (symbol, factor), group in detail.groupby(["symbol", "factor"]):
        ic = spearmanr(group.factor_value, group.forward_return).statistic if len(group) >= 3 else np.nan
        by_asset.append({"product": symbol, "factor": factor, "observations": len(group), "spearman_ic": float(ic) if pd.notna(ic) else np.nan,
                         "ic_positive": bool(ic > 0) if pd.notna(ic) else None})
    summary = []
    for factor, group in detail.groupby("factor"):
        per_dates = group.groupby("signal_date").apply(lambda x: spearmanr(x.factor_value, x.forward_return).statistic if len(x) >= 3 else np.nan)
        summary.append({"factor": factor, "rebalances": int(per_dates.notna().sum()), "mean_cross_sectional_ic": float(per_dates.mean()),
                        "positive_ic_fraction": float((per_dates.dropna() > 0).mean()) if per_dates.notna().any() else np.nan})
    return pd.DataFrame(summary), pd.DataFrame(by_asset)


def main() -> None:
    open_, close, contracts, _volume = load_panel()
    if close.empty or close.index.min() > "2019-01-02" or close.index.max() < "2026-10-09":
        raise RuntimeError(f"Daily mainline cache is incomplete: {close.index.min()}..{close.index.max()}")
    if close.isna().any().any() or open_.isna().any().any():
        raise RuntimeError("Required pilot products contain missing daily OHLC values")
    OUT.mkdir(parents=True, exist_ok=True)

    targets, logs = build_targets(close)
    daily, contributions = backtest(open_, close, contracts, targets, keep_contributions=True)
    base_metrics = metrics(daily)

    # Drop one factor at a time to estimate its incremental value in this factor set.
    ablations = [{"model": "all_factors", **base_metrics}]
    for factor in WEIGHTS:
        ab_targets, _ = build_targets(close, omit_factor=factor)
        ab_daily, _ = backtest(open_, close, contracts, ab_targets)
        ab_metric = metrics(ab_daily)
        ablations.append({"model": f"without_{factor}", "omitted_factor": factor, **ab_metric,
                          "return_contribution_pp": (base_metrics["total_return"] - ab_metric["total_return"]) * 100,
                          "max_drawdown_change_pp": (ab_metric["max_drawdown"] - base_metrics["max_drawdown"]) * 100})
    ablation_df = pd.DataFrame(ablations)

    period_df = period_table(daily)
    ic_summary, ic_by_asset = factor_ic(close, targets, logs)

    asset_rows = []
    for symbol in contributions.columns:
        series = contributions[symbol]
        asset_rows.append({
            "product": symbol,
            "weighted_contribution_pp": float(series.sum() * 100),
            "standalone_contribution_mdd": drawdown((1 + series).cumprod()),
            "active_days": int(targets[symbol].abs().sum() > 0) and int((targets[symbol].shift(1).fillna(0).abs() > 0).sum()),
            "long_days": int((targets[symbol].shift(1).fillna(0) > 0).sum()),
            "short_days": int((targets[symbol].shift(1).fillna(0) < 0).sum()),
        })
    asset_df = pd.DataFrame(asset_rows).sort_values("weighted_contribution_pp", ascending=False)

    daily_out = daily.copy()
    daily_out.insert(0, "trading_date", daily_out.index)
    daily_out.to_csv(OUT / "daily_equity.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(OUT / "target_weights_at_close.csv", encoding="utf-8-sig")
    ablation_df.to_csv(OUT / "factor_contribution.csv", index=False, encoding="utf-8-sig")
    period_df.to_csv(OUT / "period_metrics.csv", index=False, encoding="utf-8-sig")
    asset_df.to_csv(OUT / "product_attribution.csv", index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    ic_by_asset.to_csv(OUT / "factor_ic_by_product.csv", index=False, encoding="utf-8-sig")

    signal_rows = []
    for item in logs:
        row = {"signal_date": item["date"], "long": ",".join(item["long"]), "short": ",".join(item["short"])}
        for rank, (symbol, score) in enumerate(item["ranking"], start=1):
            row[f"rank_{rank}"] = symbol
            row[f"score_{rank}"] = score
        signal_rows.append(row)
    pd.DataFrame(signal_rows).to_csv(OUT / "rebalance_signals.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(daily.index), daily.equity, label="Strategy 27 (net)", linewidth=1.5)
    plt.axhline(1.0, color="gray", linewidth=0.7)
    plt.title("Strategy 27: Daily Mainline Backtest")
    plt.xlabel("Date")
    plt.ylabel("Equity (initial = 1.0)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "equity_curve.png", dpi=160)
    plt.close()

    report = {
        "strategy": "27_multi_factor_ranking",
        "strategy_source": "strategies/27_multi_factor_ranking.py",
        "method": "offline daily research simulation using cached mainline bars; signal after close, execution at next open; OHLC is back-adjusted at main-contract changes and cross-contract roll gaps are excluded from PnL",
        "start_requested": START,
        "end_requested": END,
        "first_market_day": str(close.index.min()),
        "last_market_day": str(close.index.max()),
        "products": PRODUCTS,
        "lookback_bars": LOOKBACK,
        "rebalance_bars": REBALANCE_BARS,
        "long_count": TOP_N,
        "short_count": BOT_N,
        "factor_weights": WEIGHTS,
        "gross_exposure_target": 1.0,
        "cost_assumption_one_way_bps": BASELINE_COST_BPS,
        "cost_note": "Generic basis-point proxy for commission plus slippage; exact historical fee schedule and tick-size fills are unavailable in the cached OHLCV data.",
        "mainline_roll_note": "At a contract change, link the new contract at its open to the prior adjusted close for indicators; do not credit the raw cross-contract gap to PnL and charge close/reopen turnover.",
        "rebalance_count": len(logs),
        "metrics": base_metrics,
        "output_files": ["daily_equity.csv", "equity_curve.png", "rebalance_signals.csv", "factor_contribution.csv", "factor_ic_summary.csv", "factor_ic_by_product.csv", "product_attribution.csv", "period_metrics.csv", "target_weights_at_close.csv"],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(OUT), "rebalance_count": len(logs), "metrics": base_metrics,
                      "periods": period_df.to_dict(orient="records"), "factor_contribution": ablation_df.to_dict(orient="records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
