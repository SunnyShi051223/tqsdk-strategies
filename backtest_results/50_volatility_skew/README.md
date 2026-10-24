# 50_volatility_skew 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：16
- 调仓信号数：93
- 净收益：-52.21%
- 年化收益：-9.41%
- 最大回撤：-52.22%
- Sharpe：-0.87
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source assigns the volume-change factor weight 15.0 then multiplies it by 0.01; the implementation follows the resulting 0.15 effective weight.
