#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双均线趋势策略
====================

【关于 TqSdk】
TqSdk 是信易科技开源的 Python 量化交易开发包，面向国内期货、期权、股票等市场，提供实时行情、K 线数据、历史回测、模拟交易和实盘交易等能力。
本仓库中的策略示例通常使用 TqApi 获取行情和 K 线，用 TqSim 或模拟账户演示交易流程，并通过目标仓位或下单接口把策略信号转成交易动作。
运行这些示例前，需要先安装 tqsdk；快期账号已在 strategies/config_local.py 本地配置，合约和回测日期可在参数区修改。

【策略介绍】
双均线趋势策略用于演示如何把一个明确的交易想法落到 TqSdk 策略脚本中。核心思路是：MA5/MA20 金叉做多、死叉做空
合约与回测日期可在文件上方参数区设置；运行时先订阅行情或 K 线数据，再计算对应的指标、价差、排名或过滤条件；当信号满足要求时，策略会调整模拟账户持仓，信号消失或风险条件触发时退出。
这类策略更适合方向持续的行情，在横盘震荡中容易反复进出，需要结合风控和周期过滤使用。

【运行说明】
1. 安装依赖：pip install tqsdk -U。
2. 快期账号已在 strategies/config_local.py 本地配置，无需额外设置。
3. 先使用模拟账户运行和观察日志，不建议未经验证直接用于实盘。
4. 如果合约代码已经过期，需要替换为当前在市的主力或目标合约。

【风险提示】
本策略只用于学习和研究，不构成投资建议。技术指标和历史规律都会失效，趋势、震荡、跳空、流动性不足和手续费滑点都可能导致亏损。用于真实交易前，应先完成回测、模拟交易、参数敏感性检查和风控评估。

【原有策略说明】
双均线趋势跟踪策略 (Double Moving Average Strategy)
====================================================

策略逻辑：
    - 使用短期均线（MA5）和长期均线（MA20）的交叉信号判断趋势方向
    - 金叉（短均线从下方向上穿越长均线）：目标仓位设为 +VOLUME（做多）
    - 死叉（短均线从上方向下穿越长均线）：目标仓位设为 -VOLUME（做空）
    - 使用 TargetPosTask 管理持仓，无需手动处理追单、撤单、部分成交等细节

【为什么用 TargetPosTask 而不是 insert_order】
    insert_order 只是"发一笔委托"，策略需要自己跟踪订单状态：
      - 委托未成交怎么办？追单？改价？
      - 部分成交时剩余量如何处理？
      - 反手时先撤未成交的单？
    TargetPosTask 封装了这一切：
      - 只需告诉它"我现在想持有多少手"
      - 它自动计算需要买卖的数量，并持续追单直到达到目标仓位
      - 目标仓位改变时会自动撤掉旧委托再按新目标下单
      - 正数 = 多头 N 手，负数 = 空头 N 手，0 = 全部平仓

适用品种：
    趋势性较强的品种，如螺纹钢（SHFE.rb）、原油（INE.sc）、铜（SHFE.cu）等

风险提示：
    - 均线策略在震荡行情中容易产生频繁的假信号（亏损）
    - 建议结合成交量、波动率等过滤器使用
    - 本代码仅供学习参考，不构成任何投资建议

参数说明：
    SYMBOL      : 交易合约代码，格式为 "交易所.合约代码"
    SHORT_PERIOD: 短期均线周期（K线根数）
    LONG_PERIOD : 长期均线周期（K线根数）
    KLINE_DUR   : K线周期（秒），60=1分钟K线，3600=1小时K线
    VOLUME      : 持仓手数（目标仓位的绝对值）

依赖：
    pip install tqsdk -U

作者：tqsdk-strategies
文档：https://doc.shinnytech.com/tqsdk/latest/
      https://doc.shinnytech.com/tqsdk/latest/reference/tqsdk.api.html#tqsdk.api.TqApi.TargetPosTask
"""

from datetime import date

from tqsdk import BacktestFinished, TqApi, TqAuth, TqBacktest, TqSim, TargetPosTask
from tqsdk.tafunc import ma, crossup, crossdown

from backtest_support import create_backtest_api, run_backtest

# ===================== 策略参数 =====================
SYMBOL = "SHFE.rb2410"  # specific futures contract; adjust for the backtest interval

SHORT_PERIOD = 5            # 短期均线周期：5根K线
LONG_PERIOD = 20            # 长期均线周期：20根K线
KLINE_DUR = 60 * 60         # K线周期：3600秒 = 1小时K线
VOLUME = 1                  # 持仓手数（正数=多头，负数=空头，0=空仓）
BACKTEST_START = date(2024, 1, 1)
BACKTEST_END = date(2024, 10, 1)
init_balance = 10_000_000  # 模拟账户初始资金（元）
# ===================================================


def main():
    """
    策略主函数

    使用 TqSim 进行模拟交易，如需实盘请替换为:
        TqAccount("期货公司名称", "资金账号", "交易密码")
    运行前请先在 strategies/config_local.py 中填写快期账号
    """

    # 初始化 API：使用模拟账户进行策略测试
    api = create_backtest_api('01_double_ma', SYMBOL, BACKTEST_START, BACKTEST_END, init_balance=init_balance)

    # 订阅 K 线数据
    klines = api.get_kline_serial(SYMBOL, KLINE_DUR, data_length=LONG_PERIOD + 10)

    # ===== TargetPosTask 初始化 =====
    # 在主循环外创建，每个合约对应一个实例
    # 它会在后台持续管理委托，确保实际持仓趋向目标仓位
    target_pos = TargetPosTask(api, SYMBOL, price="ACTIVE")
    # ================================

    print(f"[策略启动] 双均线策略 | 合约: {SYMBOL} | 短周期: {SHORT_PERIOD} | 长周期: {LONG_PERIOD}")

    try:
        while True:
            # 回测模式会逐步推进历史行情，回测结束时抛出 BacktestFinished。
            api.wait_update()

            # 仅在 K 线有更新时重新计算信号
            if not api.is_changing(klines):
                continue

            print(
                f"[行情更新] 最新价: {klines.close.iloc[-1]:.2f} | "
                f"时间: {klines.iloc[-1].datetime}"
            )

            # ---- 计算技术指标 ----
            ma_short = ma(klines.close, SHORT_PERIOD)
            ma_long = ma(klines.close, LONG_PERIOD)
            is_golden_cross = crossup(ma_short, ma_long).iloc[-1]
            is_death_cross = crossdown(ma_short, ma_long).iloc[-1]

            print(
                f"MA{SHORT_PERIOD}: {ma_short.iloc[-1]:.2f} | "
                f"MA{LONG_PERIOD}: {ma_long.iloc[-1]:.2f}"
            )

            if is_golden_cross:
                print(f">>> 金叉！目标仓位: +{VOLUME}（做多）")
                target_pos.set_target_volume(VOLUME)
            elif is_death_cross:
                print(f">>> 死叉！目标仓位: -{VOLUME}（做空）")
                target_pos.set_target_volume(-VOLUME)
    except BacktestFinished:
        print(
            f"[回测完成] 主连: {SYMBOL} | "
            f"区间: {BACKTEST_START} 至 {BACKTEST_END}"
        )
    finally:
        api.close()


if __name__ == "__main__":
    run_backtest(main)
