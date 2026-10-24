# 49_mean_variance_portfolio 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：8
- 调仓信号数：91
- 净收益：29.33%
- 年化收益：3.50%
- 最大回撤：-12.91%
- Sharpe：0.44
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.
