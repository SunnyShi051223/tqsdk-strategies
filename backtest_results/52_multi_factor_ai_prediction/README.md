# 52_multi_factor_ai_prediction 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：12
- 调仓信号数：369
- 净收益：0.00%
- 年化收益：0.00%
- 最大回撤：0.00%
- Sharpe：0.00
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source implementation has an unreachable/one-shot signal condition; the as-coded backtest is expected to remain flat. See the strategy note and factor signals.
