# 46_multi_factor 日线研究回测

- 回测区间：2019-01-02 至 2026-10-09
- 品种数：10
- 调仓信号数：1882
- 净收益：-8.75%
- 年化收益：-1.22%
- 最大回撤：-49.60%
- Sharpe：0.03
- 成本假设：每边 5.0 bp

## 口径说明

offline daily research simulation; daily close signal and next daily open execution; roll-adjusted OHLC, no raw cross-contract roll-gap PnL

Target lots are normalized to gross notional exposure 1.0 for cross-strategy comparison; this is not TqSim cash/lot accounting.

## 特别说明

- The source lists CZCE.cs as cotton and CZCE.mo as rapeseed oil. Exchange/product aliases map these to DCE.cs (corn starch) and CZCE.OI (rapeseed oil) respectively; the source's CZCE.cs label is inconsistent with the product ID.
