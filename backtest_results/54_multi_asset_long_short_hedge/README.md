# 54_multi_asset_long_short_hedge 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：12
- 调仓信号数：1862
- 净收益：-67.65%
- 年化收益：-14.02%
- 最大回撤：-76.78%
- Sharpe：-0.62
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source calculates volatility but does not use it in the rank or the position size.
