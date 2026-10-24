# 60_cross_section_ml_ranking 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：12
- 调仓信号数：187
- 净收益：9.85%
- 年化收益：1.27%
- 最大回撤：-47.17%
- Sharpe：0.16
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source code uses a weighted linear factor score; despite its title, it does not fit or train a machine-learning model.
