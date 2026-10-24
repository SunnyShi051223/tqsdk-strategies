# 62_macro_factor_rotation 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：20
- 调仓信号数：365
- 净收益：62.92%
- 年化收益：6.75%
- 最大回撤：-28.90%
- Sharpe：0.45
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source's liquidity factor uses current quote volume/open-interest. It is replaced here with the cached daily volume/closing open-interest ratio; no historical macroeconomic series is used.
