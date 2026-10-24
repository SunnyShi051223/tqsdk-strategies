#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Daily research backtests for the README multi-factor/rotation strategy group."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

import backtest_27_daily_pilot as common


ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "strategies"
DATA = ROOT / "backtest_results" / "_data" / "daily_mainline_bars_2019-2026-10-10.csv"
RESULTS = ROOT / "backtest_results"
START = "2019-01-01"
END = "2026-10-10"
COST_BPS = 5.0

# Pools and parameters are read from the example sources. Product aliases repair
# exchange/product typos present in some examples; see the report notes per file.
ALIASES = {
    "CZCE.MO": "CZCE.OI",
    "CZCE.CS": "DCE.cs",
    "CZCE.JD": "DCE.jd",
    "CZCE.P": "DCE.p",
    "CZCE.LK": "DCE.l",
}
POOLS = {
    "27_elder_triple_screen": ["SHFE.rb"],
    "27_multi_factor_ranking": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j"],
    "46_multi_factor": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.j", "DCE.jm", "CZCE.mo", "CZCE.rm", "CZCE.cs", "DCE.m", "DCE.a"],
    "48_sector_rotation_multi_factor": ["SHFE.rb", "SHFE.hc", "DCE.i", "SHFE.cu", "SHFE.al", "SHFE.zn", "DCE.m", "DCE.c", "CZCE.rm", "INE.sc", "SHFE.fu", "DCE.eg"],
    "49_mean_variance_portfolio": ["SHFE.rb", "SHFE.cu", "DCE.i", "DCE.m", "INE.sc", "SHFE.au", "DCE.jm", "SHFE.al"],
    "50_volatility_skew": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.j", "DCE.jm", "SHFE.cu", "SHFE.al", "SHFE.zn", "SHFE.ni", "DCE.m", "DCE.c", "CZCE.rm", "CZCE.cs", "CZCE.sr", "INE.sc", "SHFE.fu"],
    "52_multi_factor_ai_prediction": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.j", "DCE.jm", "SHFE.cu", "SHFE.al", "DCE.m", "CZCE.rm", "CZCE.cs", "CZCE.sr", "INE.sc"],
    "54_multi_asset_long_short_hedge": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j", "SHFE.cu", "SHFE.al", "SHFE.zn", "DCE.m", "DCE.y", "CZCE.MA", "CZCE.TA"],
    "55_momentum_value_factor_composite": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j", "SHFE.cu", "SHFE.al", "CZCE.MA", "CZCE.TA"],
    "56_cross_section_multi_factor_ranking": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j"],
    "58_money_flow_rotation": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j", "CZCE.MA", "CZCE.TA", "INE.sc"],
    "59_vol_momentum_composite": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.jm", "DCE.j"],
    "60_cross_section_ml_ranking": ["CZCE.RM", "CZCE.SR", "CZCE.CF", "CZCE.MA", "CZCE.JD", "CZCE.LK", "DCE.y", "DCE.m", "DCE.p", "DCE.a", "SHFE.rb", "SHFE.hc"],
    "62_macro_factor_rotation": ["SHFE.rb", "DCE.i", "DCE.j", "DCE.jm", "SHFE.hc", "DCE.c", "DCE.cs", "DCE.m", "CZCE.rm", "CZCE.OI", "CZCE.P", "DCE.y", "SHFE.cu", "SHFE.al", "SHFE.zn", "SHFE.au", "SHFE.ag", "DCE.pp", "DCE.l", "SHFE.fu"],
    "79_cross_section_risk_parity": ["SHFE.rb", "SHFE.hc", "DCE.i", "DCE.j", "DCE.jm"],
    "80_momentum_carry_composite": ["SHFE.rb", "SHFE.cu", "SHFE.al", "DCE.m", "CZCE.SR"],
    "81_term_structure_momentum_rotation": ["SHFE.rb", "INE.sc", "DCE.y", "CZCE.CF", "SHFE.au"],
    "82_liquidity_adjusted_momentum": ["SHFE.rb", "SHFE.cu", "DCE.i", "DCE.m", "CZCE.MA"],
    "83_regime_switching_allocation": ["SHFE.rb", "SHFE.cu", "INE.sc", "DCE.m", "SHFE.au"],
    "91_volatility_target_position": ["SHFE.cu"],
    "93_adaptive_position_sizing": ["SHFE.au"],
}

SOURCE_FILES = {name: name + ".py" for name in POOLS}
GENERIC_FILES = {
    "79_cross_section_risk_parity": "79_cross_section_risk_parity.py",
    "80_momentum_carry_composite": "80_momentum_carry_composite.py",
    "81_term_structure_momentum_rotation": "81_term_structure_momentum_rotation.py",
    "82_liquidity_adjusted_momentum": "82_liquidity_adjusted_momentum.py",
    "83_regime_switching_allocation": "83_regime_switching_allocation.py",
    "91_volatility_target_position": "91_volatility_target_position.py",
    "93_adaptive_position_sizing": "93_adaptive_position_sizing.py",
}


def key(product: str) -> str:
    exchange, product_id = product.split(".", 1)
    return f"{exchange.upper()}.{product_id.upper()}"


def resolve_product(product: str, available: dict[str, str]) -> str | None:
    product = ALIASES.get(key(product), product)
    return available.get(key(product))


def load_market() -> dict[str, Any]:
    bars = pd.read_csv(DATA, dtype={"trading_date": str, "mainline": str, "actual_contract": str})
    bars["product"] = bars.mainline.str.replace("KQ.m@", "", regex=False)
    dates = pd.Index(sorted(bars.trading_date.unique()), name="trading_date")
    products = sorted(bars["product"].unique())
    available = {key(p): p for p in products}

    def panel(column: str) -> pd.DataFrame:
        return bars.pivot(index="trading_date", columns="product", values=column).reindex(index=dates, columns=products)

    raw = {name: panel(name) for name in ("open", "high", "low", "close", "volume", "open_oi", "close_oi")}
    contracts = panel("actual_contract")
    adjusted = {name: raw[name].copy() for name in ("open", "high", "low", "close")}
    for product in products:
        factor = 1.0
        col = adjusted["close"].columns.get_loc(product)
        for i in range(len(dates)):
            if i and contracts[product].iloc[i] != contracts[product].iloc[i - 1]:
                factor = adjusted["close"][product].iloc[i - 1] / raw["open"][product].iloc[i]
            for field in adjusted:
                adjusted[field].iloc[i, col] = raw[field][product].iloc[i] * factor
    frames = {}
    for product in products:
        frames[product] = pd.DataFrame({
            **{field: adjusted[field][product] for field in adjusted},
            "volume": raw["volume"][product],
            "open_oi": raw["open_oi"][product],
            "close_oi": raw["close_oi"][product],
            "actual_contract": contracts[product],
        }, index=dates)
    return {"dates": dates, "available": available, "products": products, "frames": frames,
            "contracts": contracts, "open": adjusted["open"], "close": adjusted["close"]}


def load_generic_modules() -> dict[str, Any]:
    modules = {}
    for strategy, filename in GENERIC_FILES.items():
        path = STRATEGY_DIR / filename
        spec = importlib.util.spec_from_file_location("offline_" + strategy, path)
        if not spec or not spec.loader:
            raise RuntimeError(f"Could not load strategy source: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[strategy] = module
    return modules


def finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def roll(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.iloc[max(0, len(df) - n):]


def simple_return(series: pd.Series, bars_back: int) -> float:
    if len(series) < bars_back + 1 or series.iloc[-bars_back - 1] == 0:
        return np.nan
    return finite(series.iloc[-1] / series.iloc[-bars_back - 1] - 1)


def rank01(values: dict[str, float]) -> dict[str, float]:
    valid = {k: finite(v) for k, v in values.items() if np.isfinite(finite(v))}
    ordered = sorted(valid.values())
    if not ordered:
        return {}
    return {k: ordered.index(v) / max(len(ordered) - 1, 1) for k, v in valid.items()}


def zscores(values: dict[str, float], population: bool = False) -> dict[str, float]:
    valid = pd.Series({k: finite(v) for k, v in values.items()}).dropna()
    if len(valid) < 2:
        return {k: 0.5 for k in values}
    sd = valid.std(ddof=0 if population else 1)
    if not np.isfinite(sd) or sd == 0:
        return {k: 0.5 for k in values}
    return {k: float((v - valid.mean()) / sd) if k in valid.index else 0.5 for k, v in values.items()}


def atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period + 1:
        return np.nan
    prev = df.close.shift(1)
    tr = pd.concat([df.high - df.low, (df.high - prev).abs(), (df.low - prev).abs()], axis=1).max(axis=1)
    return finite(tr.iloc[-period:].mean())


def adx_di(df: pd.DataFrame, period: int = 14) -> tuple[float, float, float]:
    if len(df) < period * 2 + 1:
        return np.nan, np.nan, np.nan
    tr_values, plus_values, minus_values = [], [], []
    for i in range(len(df) - period, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        tr_values.append(max(row.high - row.low, abs(row.high - prev.close), abs(row.low - prev.close)))
        up, down = row.high - prev.high, prev.low - row.low
        plus_values.append(up if up > down and up > 0 else 0.0)
        minus_values.append(down if down > up and down > 0 else 0.0)
    tr_avg = np.mean(tr_values)
    if tr_avg == 0:
        return np.nan, np.nan, np.nan
    plus_di = np.mean(plus_values) / tr_avg * 100
    minus_di = np.mean(minus_values) / tr_avg * 100
    total = plus_di + minus_di
    return (abs(plus_di - minus_di) / total * 100 if total else np.nan), plus_di, minus_di


def source_schedule(strategy: str, i: int, date: str, warmed: bool, state: dict[str, Any]) -> bool:
    count = i + 1
    if strategy == "27_elder_triple_screen":
        return i >= 39
    if strategy == "27_multi_factor_ranking":
        return count >= 20 and count % 10 == 0
    if strategy in {"48_sector_rotation_multi_factor", "49_mean_variance_portfolio", "50_volatility_skew"}:
        step = 20
        return count % step == 0 and warmed
    if strategy == "52_multi_factor_ai_prediction":
        return pd.Timestamp(date).weekday() == 0 and warmed
    if strategy == "55_momentum_value_factor_composite":
        # Correct the source's weekday-number comparison so this weekly signal
        # can be tested as intended on daily bars.
        return pd.Timestamp(date).weekday() == 0 and warmed
    if strategy in {"56_cross_section_multi_factor_ranking", "58_money_flow_rotation", "59_vol_momentum_composite"}:
        return count % 5 == 1 and warmed
    if strategy == "60_cross_section_ml_ranking":
        return count % 10 == 1 and warmed
    if strategy == "62_macro_factor_rotation":
        return count % 5 == 0 and warmed
    if strategy in GENERIC_FILES:
        return warmed
    if strategy == "46_multi_factor":
        return warmed
    if strategy == "54_multi_asset_long_short_hedge":
        return warmed
    return False


def available_pool(strategy: str, market: dict) -> list[str]:
    resolved = []
    for product in POOLS[strategy]:
        canonical = resolve_product(product, market["available"])
        if canonical and canonical not in resolved:
            resolved.append(canonical)
    return resolved


def calc_snapshot(strategy: str, i: int, market: dict, pool: list[str], modules: dict) -> dict | None:
    factors: dict[str, dict[str, float]] = {}
    details: dict[str, Any] = {}
    if strategy == "27_multi_factor_ranking":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            c = df.close
            if len(c) < 21:
                continue
            mom = c.iloc[-1] / c.iloc[-21] - 1
            vol = c.pct_change().dropna().iloc[-20:].std()
            trend = ((c.iloc[-20:] - c.iloc[-20:].mean()) / c.iloc[-20:].mean()).std()
            factors[p] = {"momentum": finite(mom), "volatility": finite(-1 / vol if vol else np.nan), "trend": finite(trend)}
        return {"factors": factors}

    if strategy == "27_elder_triple_screen":
        p = pool[0]
        df = market["frames"][p].iloc[:i + 1]
        c = df.close
        if len(c) < 40:
            return None
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        hist = macd - macd.ewm(span=9).mean()
        lowest, highest = df.low.rolling(9).min(), df.high.rolling(9).max()
        stoch = 100 * (c - lowest) / (highest - lowest).replace(0, np.nan)
        fast, slow = c.rolling(5).mean(), c.rolling(10).mean()
        if any(pd.isna(x) for x in (hist.iloc[-1], hist.iloc[-2], stoch.iloc[-1], fast.iloc[-1], fast.iloc[-2], slow.iloc[-1], slow.iloc[-2])):
            return None
        factors[p] = {"macd_slope": finite(hist.iloc[-1] - hist.iloc[-2]), "stochastic_k": finite(stoch.iloc[-1]),
                      "ma_cross": 1.0 if fast.iloc[-2] <= slow.iloc[-2] and fast.iloc[-1] > slow.iloc[-1] else -1.0 if fast.iloc[-2] >= slow.iloc[-2] and fast.iloc[-1] < slow.iloc[-1] else 0.0}
        details[p] = {"long": bool(hist.iloc[-1] > hist.iloc[-2] and stoch.iloc[-1] < 30 and factors[p]["ma_cross"] > 0),
                      "short": bool(hist.iloc[-1] < hist.iloc[-2] and stoch.iloc[-1] > 70 and factors[p]["ma_cross"] < 0),
                      "exit_long": bool(hist.iloc[-1] < hist.iloc[-2] and factors[p]["ma_cross"] < 0),
                      "exit_short": bool(hist.iloc[-1] > hist.iloc[-2] and factors[p]["ma_cross"] > 0)}
        return {"factors": factors, "details": details}

    if strategy == "46_multi_factor":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            c = df.close
            if len(df) < 60:
                continue
            ma5, ma20, ma60 = c.tail(5).mean(), c.tail(20).mean(), c.tail(60).mean()
            a = atr(df, 14)
            vol_ratio = a / c.iloc[-1] if c.iloc[-1] else np.nan
            volavg = df.volume.tail(20).mean()
            factors[p] = {"momentum": finite(c.iloc[-1] / c.iloc[-20] - 1),
                          "trend": float((ma5 > ma20) + (ma20 > ma60)) / 2,
                          "volatility": finite(1 / (1 + vol_ratio * 100)),
                          "volume": finite(min(df.volume.iloc[-1] / volavg / 2, 1) if volavg else np.nan)}
        return {"factors": factors}

    if strategy == "48_sector_rotation_multi_factor":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 22:
                continue
            c, v = df.close, df.volume
            ret5 = np.log(c.tail(6)).diff().dropna()
            ret20 = np.log(c.tail(21)).diff().dropna()
            rv5, rv20 = ret5.std(ddof=0), ret20.std(ddof=0)
            factors[p] = {"momentum": finite(c.iloc[-1] / c.iloc[-20] - 1),
                          "low_volatility_ratio": finite(-rv5 / rv20 if rv20 else np.nan),
                          "volume_ratio": finite(v.tail(5).mean() / v.tail(20).mean() if v.tail(20).mean() else np.nan),
                          "trend": finite(c.tail(5).mean() / c.tail(20).mean() - 1)}
        return {"factors": factors}

    if strategy == "49_mean_variance_portfolio":
        if len(pool) < 3 or i < 60:
            return None
        rets = {}
        for p in pool:
            c = market["frames"][p].close.iloc[:i + 1]
            r = np.log(c).diff().dropna().tail(60)
            if len(r) < 60:
                return None
            rets[p] = r.to_numpy()
        matrix = np.column_stack([rets[p] for p in pool])
        cov = np.cov(matrix.T) * 252
        mu = matrix[-20:].mean(axis=0) * 252
        return {"factors": {p: {"expected_return": finite(mu[k])} for k, p in enumerate(pool)},
                "extras": {"mu": mu, "cov": cov, "pool": pool}}

    if strategy == "50_volatility_skew":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 25:
                continue
            c, v = df.close, df.volume
            r = np.log(c.tail(20)).diff().dropna()
            v20 = r.std(ddof=0) * np.sqrt(252)
            v10, v5 = v.tail(10).mean(), v.tail(5).mean()
            factors[p] = {"skew": finite(r.skew()), "volatility": finite(v20),
                          "momentum": finite(c.iloc[-1] / c.iloc[-20] - 1),
                          "volume_change": finite((v5 - v10) / v10 if v10 else np.nan)}
        return {"factors": factors}

    if strategy == "52_multi_factor_ai_prediction":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            c, v = df.close, df.volume
            if len(c) < 26:
                continue
            mom5, mom20 = c.iloc[-1] / c.iloc[-6] - 1, c.iloc[-1] / c.iloc[-21] - 1
            ma_cross = 1.0 if c.tail(5).mean() > c.tail(20).mean() else -1.0
            e12, e26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
            macd_hist = finite((e12.iloc[-1] - e26.iloc[-1]) / c.iloc[-1] * 100)
            volume_change = finite(v.iloc[-1] / v.tail(10).mean() - 1 if v.tail(10).mean() else 0)
            factors[p] = {"mom_5": finite(mom5), "mom_20": finite(mom20), "ma_cross": ma_cross,
                          "macd_hist": macd_hist, "volatility": finite(np.log(c.tail(21)).diff().std() * np.sqrt(252)),
                          "volume_change": volume_change}
        return {"factors": factors}

    if strategy == "54_multi_asset_long_short_hedge":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 21:
                continue
            volavg = df.volume.tail(20).mean()
            factors[p] = {"momentum": finite(df.close.iloc[-1] / df.close.iloc[-21] - 1),
                          "volatility": finite(df.close.pct_change().tail(20).std() * np.sqrt(252)),
                          "trend_filter": float(df.close.iloc[-1] > df.close.tail(20).mean()),
                          "liquidity_filter": finite(df.volume.iloc[-1] / volavg if volavg else np.nan)}
        return {"factors": factors}

    if strategy == "55_momentum_value_factor_composite":
        for p in pool:
            c = market["frames"][p].close.iloc[:i + 1]
            if len(c) < 21:
                continue
            # In the source, both helper calls use the same close/period proxy.
            proxy = finite(c.iloc[-1] / c.iloc[-20] - 1)
            factors[p] = {"momentum": proxy, "value_proxy": proxy}
        return {"factors": factors}

    if strategy == "56_cross_section_multi_factor_ranking":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 33:
                continue
            c = df.close
            mom = c.pct_change(20).iloc[-1]
            mom_short = c.pct_change(10).iloc[-1]
            mom_prev = c.pct_change(10).iloc[-11]
            a = atr(df, 14)
            _, plus_di, minus_di = adx_di(df, 14)
            factors[p] = {"momentum": finite(mom), "low_volatility": finite(-a / c.iloc[-1]),
                          "directional_strength": finite(plus_di - minus_di), "momentum_acceleration": finite(mom_short - mom_prev)}
        return {"factors": factors}

    if strategy == "58_money_flow_rotation":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 25:
                continue
            hi, lo, c, v, oi = df.high, df.low, df.close, df.volume, df.close_oi
            mult = ((c - lo) - (hi - c)) / (hi - lo + 1e-10)
            cmf_den = v.tail(20).sum()
            cmf = (mult.tail(20) * v.tail(20)).sum() / cmf_den if cmf_den else 0.0
            price_delta = c.iloc[-1] - c.iloc[-5]
            oi_delta = oi.iloc[-1] - oi.iloc[-5]
            if abs(oi_delta) < 1e-6:
                oi_div = 0.0
            elif price_delta > 0 and oi_delta > 0:
                oi_div = 2.0
            elif price_delta > 0:
                oi_div = -1.0
            elif oi_delta > 0:
                oi_div = -2.0
            else:
                oi_div = 1.0
            vwap = (c.tail(20) * v.tail(20)).sum() / v.tail(20).sum() if v.tail(20).sum() else c.iloc[-1]
            factors[p] = {"cmf": finite(cmf), "oi_divergence": finite(oi_div), "vwap_deviation": finite(c.iloc[-1] / vwap - 1 if vwap else 0)}
        return {"factors": factors}

    if strategy == "59_vol_momentum_composite":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 33:
                continue
            c = df.close
            r = c.pct_change().dropna()
            ma = c.tail(20).mean()
            sd = c.tail(20).std()
            boll_z = finite((c.iloc[-1] - ma) / sd if sd else 0.0)
            delta = c.diff()
            gain = delta.clip(lower=0).tail(14).mean()
            loss = (-delta.clip(upper=0)).tail(14).mean()
            rsi = 100 - 100 / (1 + gain / loss) if loss else 100.0
            temporal = boll_z * 0.5 + (rsi - 50) / 50 * 0.5
            adx, _, _ = adx_di(df, 14)
            vol = r.tail(14).std()
            vol_adj = c.pct_change(20).iloc[-1] / vol if vol and np.isfinite(vol) else 0.0
            factors[p] = {"temporal_signal": finite(temporal), "vol_adjusted_momentum": finite(vol_adj), "adx": finite(adx)}
        return {"factors": factors}

    if strategy == "60_cross_section_ml_ranking":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            if len(df) < 25:
                continue
            c, v, o = df.close, df.volume, df.open
            typical = c
            vwap = (typical * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
            money_flow = (vwap.diff() * v).rolling(20).sum().iloc[-1]
            factors[p] = {"momentum": finite(c.iloc[-1] / c.iloc[-20] - 1),
                          "volatility": finite(c.pct_change().rolling(20).std().iloc[-1]),
                          "trend": finite(c.tail(5).mean() / c.tail(20).mean() - 1),
                          "volume": finite(v.iloc[-1] / v.tail(20).mean() if v.tail(20).mean() else 0),
                          "money_flow": finite(money_flow),
                          "overnight_gap": finite(o.iloc[-1] / c.iloc[-2] - 1)}
        return {"factors": factors}

    if strategy == "62_macro_factor_rotation":
        for p in pool:
            df = market["frames"][p].iloc[:i + 1]
            c, v, oi = df.close, df.volume, df.close_oi
            if len(c) < 60:
                continue
            recent, earlier = c.iloc[-5:], c.iloc[-10:-5]
            strength = (recent.iloc[-1] / recent.iloc[0]) / (earlier.iloc[-1] / earlier.iloc[0] + 1e-8) if len(earlier) else 0.0
            vol = c.pct_change().dropna().tail(20).std()
            factors[p] = {"momentum_20d": finite(c.iloc[-1] / c.iloc[-20] - 1),
                          "momentum_60d": finite(c.iloc[-1] / c.iloc[-60] - 1),
                          "low_volatility": finite(1 / (vol + 1e-8)),
                          "liquidity": finite(v.iloc[-1] / oi.iloc[-1] if oi.iloc[-1] > 0 else 0),
                          "strength": finite(strength)}
        return {"factors": factors}

    if strategy in GENERIC_FILES:
        module = modules[strategy]
        for p in pool:
            df = market["frames"][p].iloc[max(0, i - 179):i + 1].drop(columns="actual_contract")
            if len(df) < 80:
                continue
            features = module.calculate_features(df)
            factors[p] = {name: finite(value) for name, value in features.items()}
        return {"factors": factors}
    return None


def nudge_rank_weights(factors: dict[str, dict[str, float]], names: list[str], weights: dict[str, float], omit: str | None) -> dict[str, float]:
    names = [name for name in names if name != omit]
    ranks = {name: rank01({p: f.get(name, np.nan) for p, f in factors.items()}) for name in names}
    scores = {}
    for p in factors:
        score = sum(weights[name] * ranks.get(name, {}).get(p, 0.5) for name in names)
        scores[p] = finite(score)
    return scores


def normalize_lots(lots: dict[str, float]) -> dict[str, float]:
    gross = sum(abs(x) for x in lots.values())
    return {p: value / gross for p, value in lots.items()} if gross else {p: 0.0 for p in lots}


def sorted_edges(scores: dict[str, float], long_n: int, short_n: int = 0, long_only: bool = False) -> tuple[dict[str, float], list[str], list[str]]:
    valid = [(p, v) for p, v in scores.items() if np.isfinite(v)]
    valid.sort(key=lambda pair: (-pair[1], pair[0]))
    if not valid:
        return {}, [], []
    long_symbols = [p for p, _ in valid[:min(long_n, len(valid))]]
    short_symbols = [] if long_only else [p for p, _ in valid[-min(short_n, len(valid)):]]
    short_symbols = [p for p in short_symbols if p not in long_symbols]
    lots = {p: 0.0 for p, _ in valid}
    for p in long_symbols:
        lots[p] = 1.0
    for p in short_symbols:
        lots[p] = -1.0
    return lots, long_symbols, short_symbols


def mvo_weights(snapshot: dict, omit: str | None = None) -> dict[str, float]:
    extra = snapshot["extras"]
    pool, mu, cov = extra["pool"], extra["mu"], extra["cov"].copy()
    if omit == "expected_return":
        return {p: 1.0 / len(pool) for p in pool}
    if omit == "covariance":
        cov = np.diag(np.diag(cov))
    n = len(mu)
    max_weight = 0.30
    rf = 0.02 / 252
    def objective(w):
        vol = np.sqrt(w @ cov @ w + 1e-10)
        return -float((w @ mu - rf) / vol)
    result = minimize(objective, np.ones(n) / n, method="SLSQP", bounds=[(-max_weight, max_weight)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: np.abs(w).sum() - 1.0}],
                      options={"maxiter": 250, "ftol": 1e-7})
    w = result.x if result.success else np.ones(n) / n
    daily_cov = cov / 252
    port_vol = np.sqrt(w @ daily_cov @ w * 252)
    if port_vol > 1e-6:
        w = w * min(0.10 / port_vol, 2.0)
    return {p: float(w[k]) for k, p in enumerate(pool)}


def create_positions(strategy: str, snapshot: dict, omit: str | None, modules: dict, state: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
    factors = snapshot["factors"]
    detail: dict[str, Any] = {"scores": {}, "long": [], "short": []}

    if strategy == "27_elder_triple_screen":
        p = next(iter(factors), None)
        if p is None:
            return state, detail
        f, d = factors[p], snapshot["details"][p]
        # Dropping a screen removes that confirmation condition, then retains the original entry/exit action.
        long_ok = (omit == "macd_slope" or f["macd_slope"] > 0) and (omit == "stochastic_k" or f["stochastic_k"] < 30) and (omit == "ma_cross" or f["ma_cross"] > 0)
        short_ok = (omit == "macd_slope" or f["macd_slope"] < 0) and (omit == "stochastic_k" or f["stochastic_k"] > 70) and (omit == "ma_cross" or f["ma_cross"] < 0)
        if long_ok:
            state[p] = 1.0
        elif short_ok:
            state[p] = -1.0
        elif (omit == "macd_slope" or f["macd_slope"] < 0) and (omit == "ma_cross" or f["ma_cross"] < 0):
            state[p] = 0.0
        elif (omit == "macd_slope" or f["macd_slope"] > 0) and (omit == "ma_cross" or f["ma_cross"] > 0):
            state[p] = 0.0
        detail["scores"] = {p: float(f["macd_slope"])}
        detail["long"] = [p] if state[p] > 0 else []
        detail["short"] = [p] if state[p] < 0 else []
        return state, detail

    if strategy == "27_multi_factor_ranking":
        weights = {"momentum": 0.4, "volatility": 0.3, "trend": 0.3}
        names = [n for n in weights if n != omit]
        norm = {n: zscores({p: f[n] for p, f in factors.items()}, population=True) for n in weights}
        scores = {p: sum(weights[n] * norm[n][p] for n in names) for p in factors}
        lots, longs, shorts = sorted_edges(scores, 1, 1)
    elif strategy == "46_multi_factor":
        weights = {"momentum": 0.3, "trend": 0.3, "volatility": 0.2, "volume": 0.2}
        z_mom = {p: float(np.clip(f["momentum"] * 10 + 0.5, 0, 1)) for p, f in factors.items()}
        scores = {p: sum(weights[n] * (z_mom[p] if n == "momentum" else factors[p][n]) for n in weights if n != omit) for p in factors}
        lots, longs, shorts = sorted_edges(scores, 3, long_only=True)
    elif strategy == "48_sector_rotation_multi_factor":
        weights = {"momentum": 0.35, "low_volatility_ratio": 0.25, "volume_ratio": 0.20, "trend": 0.20}
        scores = nudge_rank_weights(factors, list(weights), weights, omit)
        lots, longs, shorts = sorted_edges(scores, 2, 2)
    elif strategy == "49_mean_variance_portfolio":
        lots = mvo_weights(snapshot, omit)
        scores = {p: lots[p] for p in lots}
        longs = [p for p, v in lots.items() if v > 0]
        shorts = [p for p, v in lots.items() if v < 0]
    elif strategy == "50_volatility_skew":
        vols = [f["volatility"] for f in factors.values() if np.isfinite(f["volatility"])]
        threshold = np.percentile(vols, 80) if vols else np.inf
        valid = {p: f for p, f in factors.items() if np.isfinite(f["volatility"]) and f["volatility"] <= threshold}
        weights = {"skew": 0.35, "low_volatility": 0.25, "momentum": 0.25, "volume_change": 0.15}
        ranked_factors = {"skew": rank01({p: f["skew"] for p, f in valid.items()}),
                          "low_volatility": rank01({p: -f["volatility"] for p, f in valid.items()}),
                          "momentum": rank01({p: f["momentum"] for p, f in valid.items()}),
                          "volume_change": rank01({p: f["volume_change"] for p, f in valid.items()})}
        names = [n for n in weights if n != omit]
        scores = {p: sum(weights[n] * ranked_factors[n].get(p, 0.5) * (0.01 if n == "volume_change" else 1.0) for n in names) for p in valid}
        lots, longs, shorts = sorted_edges(scores, 2, 2)
    elif strategy == "52_multi_factor_ai_prediction":
        scores = {}
        for p, f in factors.items():
            momscore = sum(1 if f[k] > threshold else -1 if f[k] < -threshold else 0 for k, threshold in (("mom_5", 0.02), ("mom_20", 0.05)) if k != omit)
            trend = 0 if omit == "ma_cross" else f["ma_cross"]
            volume = 0 if omit == "volume_change" else (1 if f["volume_change"] > 0 else -1 if f["volume_change"] < -0.3 else 0)
            score = momscore * 0.4 + trend * 0.4 + volume * 0.2
            confidence = min(abs(score) / 3.0, 1.0)
            scores[p] = score
            if abs(score) > 1 and confidence >= 0.60:
                state[p] = float(np.sign(score))
            else:
                state[p] = 0.0
        lots = state.copy()
        longs = [p for p, v in lots.items() if v > 0][:3]
        shorts = [p for p, v in lots.items() if v < 0][:3]
        lots = {p: (1.0 if p in longs else -1.0 if p in shorts else 0.0) for p in factors}
    elif strategy == "54_multi_asset_long_short_hedge":
        valid = {p: f for p, f in factors.items() if (omit == "liquidity_filter" or f["liquidity_filter"] >= 0.5)}
        if len(valid) < 4:
            valid = {}
        ordered = sorted(valid, key=lambda p: (-valid[p]["momentum"], p))
        n_long = max(1, int(len(ordered) * 0.2)) if ordered else 0
        n_short = max(1, int(len(ordered) * 0.2)) if ordered else 0
        longs = ordered[:n_long]
        shorts = ordered[-n_short:] if n_short else []
        longs = [p for p in longs if omit == "trend_filter" or factors[p]["trend_filter"] > 0]
        shorts = [p for p in shorts if omit == "trend_filter" or factors[p]["trend_filter"] <= 0]
        lots = {p: (1.0 if p in longs else -1.0 if p in shorts else 0.0) for p in factors}
        scores = {p: f["momentum"] for p, f in factors.items()}
    elif strategy == "55_momentum_value_factor_composite":
        weights = {"momentum": 0.6, "value_proxy": 0.4}
        ranks = {n: rank01({p: f[n] for p, f in factors.items()}) for n in weights}
        scores = {p: sum(weights[n] * (2 * ranks[n].get(p, 0.5) - 1) for n in weights if n != omit) for p in factors}
        n_edge = max(1, len(scores) // 3)
        lots, longs, shorts = sorted_edges(scores, n_edge, n_edge)
    elif strategy == "56_cross_section_multi_factor_ranking":
        names = ["momentum", "low_volatility", "directional_strength"]
        ranks = {n: rank01({p: f[n] for p, f in factors.items()}) for n in names}
        scores = {p: sum(ranks[n].get(p, 0.5) for n in names if n != omit) / 3 for p in factors}
        n_edge = max(1, int(len(scores) * 0.25))
        lots, longs, shorts = sorted_edges(scores, n_edge, n_edge)
    elif strategy == "58_money_flow_rotation":
        weights = {"cmf": 0.5, "oi_divergence": 0.3, "vwap_deviation": 0.2}
        ranks = {n: rank01({p: f[n] for p, f in factors.items()}) for n in weights}
        scores = {p: sum(weights[n] * ranks[n].get(p, 0.5) for n in weights if n != omit) for p in factors}
        n_edge = max(1, int(len(scores) * 0.25))
        lots, longs, shorts = sorted_edges(scores, n_edge, n_edge)
    elif strategy == "59_vol_momentum_composite":
        mom_rank = rank01({p: f["vol_adjusted_momentum"] for p, f in factors.items()})
        scores = {}
        for p, f in factors.items():
            temporal = 0.0 if omit == "temporal_signal" else f["temporal_signal"] * 0.6
            cross = 0.0 if omit == "vol_adjusted_momentum" else (mom_rank.get(p, 0.5) - 0.5) * 2 * 0.4
            scores[p] = temporal + cross
        avg_adx = np.nanmean([f.get("adx", 25) if np.isfinite(f.get("adx", np.nan)) else 25 for f in factors.values()]) if factors else 0
        n_edge = max(1, int(len(scores) * 0.25))
        lots, longs, shorts = sorted_edges(scores, n_edge, n_edge)
        scale = 1.0 if avg_adx >= 40 else 0.5 if avg_adx >= 25 else 0.0
        if omit == "adx_regime":
            scale = 1.0
        lots = {p: value * scale for p, value in lots.items()}
        if scale == 0:
            longs, shorts = [], []
    elif strategy == "60_cross_section_ml_ranking":
        weights = {"momentum": 0.25, "volatility": -0.10, "trend": 0.20, "volume": 0.10, "money_flow": 0.15, "overnight_gap": 0.20}
        z = {n: zscores({p: f[n] for p, f in factors.items()}) for n in weights}
        scores = {p: sum(weights[n] * z[n].get(p, 0) for n in weights if n != omit) for p in factors}
        lots, longs, shorts = sorted_edges(scores, min(4, len(scores)), long_only=True)
    elif strategy == "62_macro_factor_rotation":
        weights = {"momentum_20d": 0.30, "momentum_60d": 0.15, "low_volatility": 0.20, "liquidity": 0.15, "strength": 0.20}
        z = {n: zscores({p: f[n] for p, f in factors.items()}) for n in weights}
        scores = {p: sum(weights[n] * z[n].get(p, 0.5) for n in weights if n != omit) for p in factors}
        lots, longs, shorts = sorted_edges(scores, min(5, len(scores)), long_only=True)
    elif strategy in GENERIC_FILES:
        module = modules[strategy]
        weights = dict(module.FACTOR_WEIGHTS)
        names = [n for n in weights if n != omit]
        scores = {p: float(module.clamp(sum(weights[n] * f.get(n, 0.0) for n in names))) for p, f in factors.items()}
        if module.MODE == "single":
            p = next(iter(factors), None)
            if p is None:
                lots = {}
            else:
                score = scores[p]
                if abs(score) < module.EXIT_SCORE:
                    state[p] = 0.0
                elif abs(score) >= module.ENTRY_SCORE:
                    state[p] = float(np.sign(score))
                lots = state.copy()
            longs = [p for p, v in lots.items() if v > 0]; shorts = [p for p, v in lots.items() if v < 0]
        else:
            ordered = sorted(scores, key=lambda p: (-scores[p], p))
            longs = [ordered[0]] if ordered and scores[ordered[0]] >= module.ENTRY_SCORE else []
            shorts = [ordered[-1]] if ordered and scores[ordered[-1]] <= -module.ENTRY_SCORE else []
            lots = {p: (1.0 if p in longs else -1.0 if p in shorts else 0.0) for p in factors}
    else:
        scores, lots, longs, shorts = {}, {}, [], []

    detail["scores"] = {p: finite(v) for p, v in scores.items()}
    detail["long"] = longs
    detail["short"] = shorts
    return lots, detail


def create_targets(strategy: str, market: dict, pool: list[str], modules: dict, omit: str | None = None,
                   snapshots_override: dict[str, dict] | None = None) -> tuple[pd.DataFrame, list[dict], dict[str, dict]]:
    dates = market["dates"]
    target = pd.DataFrame(0.0, index=dates, columns=pool)
    state = {p: 0.0 for p in pool}
    schedule_state: dict[str, Any] = {}
    snapshots: dict[str, dict] = {}
    events = []
    for i, day in enumerate(dates):
        date = str(day)
        warmed = i >= (79 if strategy in GENERIC_FILES else 59 if strategy == "62_macro_factor_rotation" else 32 if strategy in {"56_cross_section_multi_factor_ranking", "59_vol_momentum_composite"} else 24 if strategy in {"50_volatility_skew", "58_money_flow_rotation"} else 20 if strategy in {"27_multi_factor_ranking", "48_sector_rotation_multi_factor", "54_multi_asset_long_short_hedge", "60_cross_section_ml_ranking"} else 0)
        if source_schedule(strategy, i, date, warmed, schedule_state):
            snapshot = snapshots_override.get(date) if snapshots_override is not None else calc_snapshot(strategy, i, market, pool, modules)
            if snapshot:
                snapshots[date] = snapshot
                if strategy == "49_mean_variance_portfolio":
                    raw, detail = create_positions(strategy, snapshot, omit, modules, state)
                elif strategy == "27_elder_triple_screen" or strategy in GENERIC_FILES:
                    raw, detail = create_positions(strategy, snapshot, omit, modules, state)
                else:
                    raw, detail = create_positions(strategy, snapshot, omit, modules, state)
                state = {p: raw.get(p, 0.0) for p in pool}
                events.append({"signal_date": date, "long": detail.get("long", []), "short": detail.get("short", []),
                               "scores": detail.get("scores", {}), "factors": snapshot.get("factors", {})})
        if strategy == "52_multi_factor_ai_prediction" and i >= 20:
            for p in pool:
                pos = state[p]
                c = market["frames"][p].close.iloc[:i + 1]
                ma20 = c.tail(20).mean()
                if pos > 0 and c.iloc[-1] < ma20 * 0.98:
                    state[p] = 0.0
                elif pos < 0 and c.iloc[-1] > ma20 * 1.02:
                    state[p] = 0.0
        if strategy == "59_vol_momentum_composite":
            target.loc[date] = state
        else:
            target.loc[date] = normalize_lots(state)
    return target, events, snapshots


def forward_roll_adjusted_return(market: dict, product: str, signal_i: int, horizon: int = 10) -> float:
    end = min(signal_i + horizon, len(market["dates"]) - 1)
    if end <= signal_i:
        return np.nan
    total = 1.0
    df = market["frames"][product]
    for i in range(signal_i + 1, end + 1):
        if df.actual_contract.iloc[i] == df.actual_contract.iloc[i - 1]:
            overnight = df.open.iloc[i] / df.close.iloc[i - 1] - 1
        else:
            overnight = 0.0
        intraday = df.close.iloc[i] / df.open.iloc[i] - 1
        total *= (1 + overnight) * (1 + intraday)
    return total - 1


def factor_ic_tables(market: dict, events: list[dict], pool: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    date_to_i = {str(d): i for i, d in enumerate(market["dates"])}
    for event in events:
        i = date_to_i[event["signal_date"]]
        for p in pool:
            for factor, value in event["factors"].get(p, {}).items():
                fwd = forward_roll_adjusted_return(market, p, i)
                if np.isfinite(finite(value)) and np.isfinite(fwd):
                    rows.append({"signal_date": event["signal_date"], "product": p, "factor": factor, "factor_value": finite(value), "forward_10d_return": fwd})
    data = pd.DataFrame(rows)
    if data.empty:
        return pd.DataFrame(columns=["factor", "observations", "mean_cross_sectional_ic"]), pd.DataFrame(columns=["product", "factor", "observations", "spearman_ic"])
    by_product = []
    for (product, factor), group in data.groupby(["product", "factor"]):
        ic = spearmanr(group.factor_value, group.forward_10d_return).statistic if len(group) >= 3 else np.nan
        by_product.append({"product": product, "factor": factor, "observations": int(len(group)), "spearman_ic": finite(ic), "ic_positive": bool(ic > 0) if np.isfinite(ic) else None})
    by_factor = []
    for factor, group in data.groupby("factor"):
        values = []
        for _, one_date in group.groupby("signal_date"):
            if len(one_date) >= 3:
                values.append(spearmanr(one_date.factor_value, one_date.forward_10d_return).statistic)
        by_factor.append({"factor": factor, "observations": int(len(group)), "rebalances_with_cross_section": int(len(values)),
                          "mean_cross_sectional_ic": float(np.nanmean(values)) if values else np.nan,
                          "positive_ic_fraction": float(np.mean(np.asarray(values) > 0)) if values else np.nan})
    return pd.DataFrame(by_factor), pd.DataFrame(by_product)


def segment_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, start, end in [("development_2019_2021", "2019-01-01", "2021-12-31"),
                             ("validation_2022_2023", "2022-01-01", "2023-12-31"),
                             ("out_of_sample_2024_2026-10-09", "2024-01-01", "2026-10-09")]:
        dates = pd.to_datetime(daily.index)
        sub = daily.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
        rows.append({"period": name, **common.metrics(sub)})
    return pd.DataFrame(rows)


def write_strategy_result(strategy: str, market: dict, pool: list[str], modules: dict) -> dict:
    output = RESULTS / strategy
    output.mkdir(parents=True, exist_ok=True)
    targets, events, snapshots = create_targets(strategy, market, pool, modules)
    open_, close, contracts = (market[name][pool] for name in ("open", "close", "contracts"))
    daily, contributions = common.backtest(open_, close, contracts, targets, cost_bps=COST_BPS, keep_contributions=True)
    base = common.metrics(daily)

    factor_names = sorted({name for snapshot in snapshots.values() for f in snapshot.get("factors", {}).values() for name in f})
    if strategy == "49_mean_variance_portfolio":
        factor_names = ["expected_return", "covariance"]
    elif strategy == "27_elder_triple_screen":
        factor_names = ["macd_slope", "stochastic_k", "ma_cross"]
    elif strategy in GENERIC_FILES:
        factor_names = list(modules[strategy].FACTOR_WEIGHTS)
    elif strategy == "54_multi_asset_long_short_hedge":
        factor_names = ["momentum", "trend_filter", "liquidity_filter"]
    ablation_rows = [{"model": "all_factors", **base}]
    for factor in factor_names:
        ab_targets, _, _ = create_targets(strategy, market, pool, modules, omit=factor, snapshots_override=snapshots)
        ab_daily, _ = common.backtest(open_, close, contracts, ab_targets, cost_bps=COST_BPS)
        ab = common.metrics(ab_daily)
        ablation_rows.append({"model": f"without_{factor}", "omitted_factor": factor, **ab,
                              "return_contribution_pp": (base["total_return"] - ab["total_return"]) * 100,
                              "max_drawdown_change_pp": (ab["max_drawdown"] - base["max_drawdown"]) * 100})
    ablations = pd.DataFrame(ablation_rows)
    periods = segment_metrics(daily)
    ic_summary, ic_by_product = factor_ic_tables(market, events, pool)

    asset_rows = []
    for product in pool:
        series = contributions[product]
        active = targets[product].shift(1).fillna(0)
        asset_rows.append({"product": product, "weighted_contribution_pp": float(series.sum() * 100),
                           "contribution_max_drawdown": common.drawdown((1 + series).cumprod()),
                           "active_days": int((active.abs() > 0).sum()), "long_days": int((active > 0).sum()), "short_days": int((active < 0).sum())})
    asset_table = pd.DataFrame(asset_rows).sort_values("weighted_contribution_pp", ascending=False)
    cost_rows = []
    for bps in (0.0, COST_BPS, 10.0):
        cost_daily, _ = common.backtest(open_, close, contracts, targets, cost_bps=bps)
        cost_rows.append({"one_way_cost_bps": bps, **common.metrics(cost_daily)})
    cost_table = pd.DataFrame(cost_rows)

    daily_out = daily.copy()
    daily_out.insert(0, "trading_date", daily_out.index)
    daily_out.to_csv(output / "daily_equity.csv", index=False, encoding="utf-8-sig")
    targets.to_csv(output / "target_weights_at_close.csv", encoding="utf-8-sig")
    ablations.to_csv(output / "factor_contribution.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(output / "period_metrics.csv", index=False, encoding="utf-8-sig")
    asset_table.to_csv(output / "product_attribution.csv", index=False, encoding="utf-8-sig")
    cost_table.to_csv(output / "cost_sensitivity.csv", index=False, encoding="utf-8-sig")
    ic_summary.to_csv(output / "factor_ic_summary.csv", index=False, encoding="utf-8-sig")
    ic_by_product.to_csv(output / "factor_ic_by_product.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{**{k: v for k, v in event.items() if k != "factors"}, "scores": json.dumps(event["scores"], ensure_ascii=False),
                   "factors": json.dumps(event["factors"], ensure_ascii=False)} for event in events]).to_csv(output / "rebalance_signals.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(12, 5.5))
    plt.plot(pd.to_datetime(daily.index), daily.equity, label=strategy, linewidth=1.4)
    plt.axhline(1, color="gray", linewidth=0.7)
    plt.title(strategy + " - Daily Mainline Research Backtest")
    plt.xlabel("Date")
    plt.ylabel("Equity (initial = 1.0)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "equity_curve.png", dpi=150)
    plt.close()

    notes = []
    if strategy == "27_elder_triple_screen":
        notes.append("The source asks for hourly and 15-minute screens. Per the agreed daily-only scope, those screens are represented with daily stochastic and daily 5/10 MA crosses; this is a daily proxy, not a faithful multi-timeframe replay.")
    if strategy in {"55_momentum_value_factor_composite", "52_multi_factor_ai_prediction"}:
        notes.append("The source implementation has an unreachable/one-shot signal condition; the as-coded backtest is expected to remain flat. See the strategy note and factor signals.")
    if strategy in {"55_momentum_value_factor_composite", "80_momentum_carry_composite", "81_term_structure_momentum_rotation"}:
        notes.append("Carry/term-structure fields are price-series proxies in the source. The cache contains main-contract daily bars, not simultaneous near/next-month spreads.")
    if strategy == "60_cross_section_ml_ranking":
        notes.append("The source code uses a weighted linear factor score; despite its title, it does not fit or train a machine-learning model.")
    if strategy == "62_macro_factor_rotation":
        notes.append("The source's liquidity factor uses current quote volume/open-interest. It is replaced here with the cached daily volume/closing open-interest ratio; no historical macroeconomic series is used.")
    if strategy == "54_multi_asset_long_short_hedge":
        notes.append("The source calculates volatility but does not use it in the rank or the position size.")
    if strategy == "50_volatility_skew":
        notes.append("The source assigns the volume-change factor weight 15.0 then multiplies it by 0.01; the implementation follows the resulting 0.15 effective weight.")
    if strategy == "46_multi_factor":
        notes.append("The source lists CZCE.cs as cotton and CZCE.mo as rapeseed oil. Exchange/product aliases map these to DCE.cs (corn starch) and CZCE.OI (rapeseed oil) respectively; the source's CZCE.cs label is inconsistent with the product ID.")

    summary = {
        "strategy": strategy,
        "source_file": SOURCE_FILES[strategy],
        "status": "completed" if base["average_gross_exposure"] > 0 else "no_trades_under_source_rules",
        "requested_period": {"start": START, "end": END},
        "actual_data_period": {"start": str(market["dates"].min()), "end": str(market["dates"].max())},
        "products": pool,
        "method": "offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL",
        "gross_exposure_normalization": "Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.",
        "cost_one_way_bps": COST_BPS,
        "cost_note": "Generic proxy for commission plus slippage. It is not an exchange-specific fee schedule or tick-by-tick fill simulation.",
        "rebalance_events": len(events),
        "metrics": base,
        "notes": notes,
        "files": ["daily_equity.csv", "equity_curve.png", "factor_contribution.csv", "factor_ic_summary.csv", "factor_ic_by_product.csv", "product_attribution.csv", "period_metrics.csv", "cost_sensitivity.csv", "rebalance_signals.csv", "target_weights_at_close.csv"],
    }
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    md_lines = [f"# {strategy} 日线研究回测", "", f"- 回测区间：{summary['actual_data_period']['start']} 至 {summary['actual_data_period']['end']}",
                f"- 品种数：{len(pool)}", f"- 调仓信号数：{len(events)}", f"- 净收益：{base['total_return']:.2%}",
                f"- 年化收益：{base['cagr']:.2%}", f"- 最大回撤：{base['max_drawdown']:.2%}", f"- Sharpe：{base['sharpe']:.2f}",
                f"- 成本假设：每边 {COST_BPS:.1f} bp", "", "## 口径说明", "", summary["method"], "", summary["gross_exposure_normalization"], ""]
    if notes:
        md_lines += ["## 特别说明", ""] + [f"- {note}" for note in notes] + [""]
    if strategy == "55_momentum_value_factor_composite":
        notes.append("The source compares the weekday number with a variable initialized to None; corrected here to weekly Monday rebalance so its stated weekly strategy can be evaluated.")
        md_lines += ["## 特别说明", "", f"- {notes[-1]}", ""]
    (output / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {"strategy": strategy, "status": summary["status"], "products": ",".join(pool),
            "rebalance_events": len(events), "total_return": base["total_return"], "cagr": base["cagr"],
            "sharpe": base["sharpe"], "max_drawdown": base["max_drawdown"], "average_gross_exposure": base["average_gross_exposure"],
            "output_dir": str(output), "ic_by_product": ic_by_product}


def main() -> None:
    market = load_market()
    if len(market["dates"]) < 1800 or market["dates"].min() > "2019-01-02" or market["dates"].max() < "2026-10-09":
        raise RuntimeError("The prepared daily mainline cache does not cover the requested period")
    if any(frame[["open", "high", "low", "close", "volume"]].isna().any().any() for frame in market["frames"].values()):
        raise RuntimeError("Daily mainline cache contains missing required OHLCV data")
    modules = load_generic_modules()

    rows = []
    compatibility = []
    for strategy in POOLS:
        pool = available_pool(strategy, market)
        if not pool:
            raise RuntimeError(f"No cached mainline products for {strategy}")
        result = write_strategy_result(strategy, market, pool, modules)
        rows.append({k: v for k, v in result.items() if k != "ic_by_product"})
        if not result["ic_by_product"].empty:
            temp = result["ic_by_product"].copy()
            temp.insert(0, "strategy", strategy)
            compatibility.append(temp)
        print(f"completed {strategy}: return={result['total_return']:.4f}, mdd={result['max_drawdown']:.4f}, events={result['rebalance_events']}")

    summary = pd.DataFrame(rows).sort_values("strategy")
    summary.to_csv(RESULTS / "multi_factor_rotation_summary.csv", index=False, encoding="utf-8-sig")
    compat = pd.concat(compatibility, ignore_index=True) if compatibility else pd.DataFrame()
    compat.to_csv(RESULTS / "product_indicator_compatibility.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(14, 7))
    for row in rows:
        data = pd.read_csv(Path(row["output_dir"]) / "daily_equity.csv")
        plt.plot(pd.to_datetime(data.trading_date), data.equity, linewidth=0.9, alpha=0.75, label=row["strategy"])
    plt.axhline(1.0, color="black", linewidth=0.8)
    plt.title("Multi-Factor and Rotation Strategies - Daily Mainline Comparison")
    plt.xlabel("Date")
    plt.ylabel("Equity (initial = 1.0)")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(RESULTS / "multi_factor_rotation_equity_curves.png", dpi=150)
    plt.close()
    print(f"summary={RESULTS / 'multi_factor_rotation_summary.csv'}")
    print(f"compatibility={RESULTS / 'product_indicator_compatibility.csv'}")


if __name__ == "__main__":
    main()
