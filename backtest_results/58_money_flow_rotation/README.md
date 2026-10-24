# 58_money_flow_rotation 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：8
- 调仓信号数：372
- 净收益：2.85%
- 年化收益：0.38%
- 最大回撤：-35.68%
- Sharpe：0.10
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.
