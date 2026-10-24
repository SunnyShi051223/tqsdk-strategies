# 27_elder_triple_screen 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：1
- 调仓信号数：1843
- 净收益：-0.38%
- 年化收益：-0.05%
- 最大回撤：-18.25%
- Sharpe：0.01
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source asks for hourly and 15-minute screens. Per the agreed daily-only scope, those screens are represented with daily stochastic and daily 5/10 MA crosses; this is a daily proxy, not a faithful multi-timeframe replay.
