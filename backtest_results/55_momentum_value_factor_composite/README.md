# 55_momentum_value_factor_composite 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：9
- 调仓信号数：369
- 净收益：-23.69%
- 年化收益：-3.56%
- 最大回撤：-38.02%
- Sharpe：-0.23
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source implementation has an unreachable/one-shot signal condition; the as-coded backtest is expected to remain flat. See the strategy note and factor signals.
- Carry/term-structure fields are price-series proxies in the source. The cache contains main-contract daily bars, not simultaneous near/next-month spreads.

## 特别说明

- The source compares the weekday number with a variable initialized to None; corrected here to weekly Monday rebalance so its stated weekly strategy can be evaluated.
