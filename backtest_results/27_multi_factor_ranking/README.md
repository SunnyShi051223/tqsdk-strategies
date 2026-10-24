# 27_multi_factor_ranking 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：5
- 调仓信号数：187
- 净收益：57.57%
- 年化收益：6.28%
- 最大回撤：-25.02%
- Sharpe：0.48
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.
