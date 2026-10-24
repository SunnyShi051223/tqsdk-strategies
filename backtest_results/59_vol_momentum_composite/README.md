# 59_vol_momentum_composite 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：5
- 调仓信号数：370
- 净收益：26.64%
- 年化收益：3.21%
- 最大回撤：-38.13%
- Sharpe：0.25
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.
