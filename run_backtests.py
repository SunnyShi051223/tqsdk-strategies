#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run one-symbol strategies over one or more futures main contracts."""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import json
import re
import sys
import traceback
from datetime import date
from pathlib import Path
from types import ModuleType

from prepare_futures_catalog import (
    CATALOG_PATH,
    HISTORY_PATH,
    export_catalog_and_history,
)


ROOT = Path(__file__).resolve().parent
STRATEGY_DIR = ROOT / "strategies"
MAINLINE_RE = re.compile(r"^KQ\.m@([^.]+)\.([A-Za-z]+)$")
CONTRACT_RE = re.compile(r"^([^.]+)\.([A-Za-z]+?)\d{2,4}$")
MULTI_SYMBOL_NAMES = {"PAIRS", "ASSETS", "CHAINS", "PORTFOLIO"}


def _top_level_constants(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return values


def _product_from_symbol(symbol: str) -> str | None:
    mainline = MAINLINE_RE.fullmatch(symbol)
    if mainline:
        exchange, product_id = mainline.groups()
        return f"{exchange.upper()}.{product_id}"
    contract = CONTRACT_RE.fullmatch(symbol)
    if contract:
        exchange, product_id = contract.groups()
        return f"{exchange.upper()}.{product_id}"
    return None


def discover_single_product_strategies() -> list[dict[str, object]]:
    strategies = []
    for path in sorted(STRATEGY_DIR.glob("*.py")):
        if path.name.startswith("__") or path.name == "backtest_support.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        constants = _top_level_constants(tree)
        if MULTI_SYMBOL_NAMES.intersection(constants):
            continue
        symbols_value = constants.get("SYMBOLS")
        if isinstance(symbols_value, (tuple, list)):
            symbols = [value for value in symbols_value if isinstance(value, str)]
            if len(symbols) != 1:
                continue
        else:
            symbol = constants.get("SYMBOL")
            symbols = [symbol] if isinstance(symbol, str) else []
        if len(symbols) != 1:
            continue
        product = _product_from_symbol(symbols[0])
        if product is None:
            continue
        source = path.read_text(encoding="utf-8")
        has_backtest_api = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_backtest_api"
            for node in ast.walk(tree)
        )
        has_position_orders = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "TargetPosTask"
            for node in ast.walk(tree)
        )
        has_direct_orders = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "insert_order"
            for node in ast.walk(tree)
        )
        if not has_backtest_api or not (has_position_orders or has_direct_orders):
            continue
        strategies.append(
            {
                "name": path.stem,
                "path": path,
                "product": product,
                "symbol_name": "SYMBOLS" if isinstance(symbols_value, (tuple, list)) else "SYMBOL",
            }
        )
    return strategies


def _parse_selection(raw: str, available: list[dict[str, object]], label: str) -> list[dict[str, object]]:
    if raw.strip().lower() == "all":
        return available
    tokens = [token.strip() for token in re.split(r"[,;/]+", raw) if token.strip()]
    selected: list[dict[str, object]] = []
    missing = []
    for token in tokens:
        matches = [
            row
            for row in available
            if str(row.get(label, "")).lower() == token.lower()
            or str(row.get(label, "")).lower().startswith(f"{token.lower()}_")
        ]
        if not matches:
            missing.append(token)
        selected.extend(matches)
    if missing:
        raise ValueError(f"未找到 {label}: {', '.join(missing)}")
    unique = {str(row[label]): row for row in selected}
    return [unique[key] for key in sorted(unique)]


def _read_catalog() -> list[dict[str, str]]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"缺少 {CATALOG_PATH}。先运行 prepare_futures_catalog.py 生成期货品种清单。"
        )
    with CATALOG_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _read_history_dates() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if not HISTORY_PATH.exists():
        return grouped
    with HISTORY_PATH.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            product = row.get("product", "").upper()
            trading_date = row.get("trading_date", "").strip()
            if product and trading_date:
                grouped.setdefault(product, []).append(trading_date)
    return grouped


def _load_strategy(path: Path) -> ModuleType:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(STRATEGY_DIR))
    module_name = f"_backtest_{path.stem.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载策略文件: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _job_output_dir(strategy_name: str, product: str) -> Path:
    safe_product = re.sub(r'[<>:"/\\|?*]+', "_", product)
    return ROOT / "Backtest" / strategy_name / safe_product


def _already_completed(
    strategy_name: str, product: str, start: date, end: date
) -> bool:
    summary_path = _job_output_dir(strategy_name, product) / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded_start = summary.get("requested_start_date", summary.get("start_date"))
    return recorded_start == start.isoformat() and summary.get("end_date") == end.isoformat()


def run_job(
    strategy: dict[str, object],
    product: dict[str, str],
    start: date,
    end: date,
    init_balance: float,
) -> tuple[date, list[str]]:
    from strategies.backtest_support import (
        RollingTargetPosTask,
        UnavailableHistoricalContract,
        run_backtest,
    )

    requested_start = start
    effective_start = start
    skipped_contracts: list[str] = []
    mapping_path = HISTORY_PATH
    with mapping_path.open("r", newline="", encoding="utf-8-sig") as stream:
        product_rows = sorted(
            (
                row
                for row in csv.DictReader(stream)
                if row.get("product", "").upper() == product["product"].upper()
            ),
            key=lambda row: row.get("trading_date", ""),
        )

    while effective_start <= end:
        path = Path(strategy["path"])
        module = _load_strategy(path)
        mainline = product["mainline_symbol"]
        if strategy["symbol_name"] == "SYMBOLS":
            module.SYMBOLS = [mainline]
        else:
            module.SYMBOL = mainline
        module.BACKTEST_START = effective_start
        module.BACKTEST_END = end
        module.init_balance = init_balance

        # Strategies retain their signal logic. Route their target position to
        # the actual contract mapped for each historical trading day.
        module.TargetPosTask = RollingTargetPosTask
        entry = getattr(module, "main", None)
        if not callable(entry):
            raise AttributeError(f"{path.name} does not define main()")

        try:
            run_backtest(entry)
        except UnavailableHistoricalContract as exc:
            first_row = next(
                (
                    row
                    for row in product_rows
                    if row.get("trading_date", "") >= effective_start.isoformat()
                ),
                None,
            )
            if first_row is None or first_row.get("actual_contract", "").upper() != exc.instrument.upper():
                raise RuntimeError(
                    f"{product['product']} 在 {effective_start} 启动时遇到不可用合约 "
                    f"{exc.instrument}，但它与该日主力映射不一致；为避免跳过区间中段，已停止自动跳月。"
                ) from exc

            next_row = next(
                (
                    row
                    for row in product_rows
                    if row.get("trading_date", "") > first_row["trading_date"]
                    and row.get("actual_contract", "").upper() != exc.instrument.upper()
                ),
                None,
            )
            if next_row is None:
                raise RuntimeError(
                    f"{product['product']} 在回测结束前没有找到可尝试的后续主力合约；"
                    f"起始合约 {exc.instrument} 不可用。"
                ) from exc

            skipped_contracts.append(exc.instrument)
            effective_start = date.fromisoformat(next_row["trading_date"])
            print(
                f"[跳过不可用早期合约] {product['product']} {exc.instrument}；"
                f"改从 {effective_start} 的 {next_row['actual_contract']} 重试。"
            )
            continue

        output_dir = _job_output_dir(str(strategy["name"]), product["product"])
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["requested_start_date"] = requested_start.isoformat()
            summary["skipped_unavailable_contracts"] = skipped_contracts
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return effective_start, skipped_contracts

    raise RuntimeError(f"{product['product']} 没有可执行的回测日期。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按策略和期货主连品种运行单品种策略历史回测。"
    )
    parser.add_argument("--strategy", default=None, help="策略编号/文件名，支持 01/02 或 all。")
    parser.add_argument("--product", default="all", help="期货品种，例如 DCE.m，或 all。")
    parser.add_argument("--all", dest="all_products", action="store_true", help="回测全部期货品种，等同于 --product all。")
    parser.add_argument("--start", type=date.fromisoformat, default=None, help="回测开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="回测结束日期 YYYY-MM-DD")
    parser.add_argument("--init-balance", type=float, default=10_000_000.0, help="模拟初始资金")
    parser.add_argument("--resume", action="store_true", help="跳过已有相同日期区间结果的任务")
    parser.add_argument("--dry-run", action="store_true", help="仅显示计划任务，不连接行情服务")
    parser.add_argument("--list-strategies", action="store_true", help="显示可批量回测的单品种策略")
    args = parser.parse_args()
    if args.all_products:
        args.product = "all"

    strategies = discover_single_product_strategies()
    if args.list_strategies:
        for strategy in strategies:
            print(f"{strategy['name']}\t{strategy['product']}")
        return
    if not args.strategy:
        parser.error("除 --list-strategies 外，必须指定 --strategy。")

    selected_strategies = _parse_selection(args.strategy, strategies, "name")
    products = [part.strip() for part in re.split(r"[,;/]+", args.product) if part.strip()]
    start_for_mapping = args.start or date(2000, 1, 1)
    end_for_mapping = args.end or date.today()
    if args.start and args.end and args.start > args.end:
        parser.error("--start 必须早于或等于 --end")
    if not args.dry_run:
        export_catalog_and_history(
            products=products or ["all"],
            start=start_for_mapping,
            end=end_for_mapping,
        )

    catalog = _read_catalog()
    selected_products = _parse_selection(args.product, catalog, "product")
    history_dates = _read_history_dates()
    end_date = args.end or date.today()
    jobs = []
    for strategy in selected_strategies:
        for product in selected_products:
            product_key = product["product"].upper()
            dates = history_dates.get(product_key, [])
            default_start = date.fromisoformat(min(dates)) if dates else date(2000, 1, 1)
            start_date = args.start or default_start
            if start_date > end_date:
                continue
            if not dates or not (min(dates) <= end_date.isoformat() and max(dates) >= start_date.isoformat()):
                raise ValueError(
                    f"{product['product']} 的主力映射没有覆盖 {start_date} 至 {end_date}。"
                )
            jobs.append((strategy, product, start_date, end_date))

    if not jobs:
        print("没有符合条件的策略—品种任务。")
        return
    print(f"计划任务: {len(jobs)} | 品种数: {len(selected_products)} | 策略数: {len(selected_strategies)}")
    if args.dry_run:
        for index, (strategy, product, start_date, end_date) in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] 计划: {strategy['name']} | "
                f"{product['mainline_symbol']} | {start_date}..{end_date}"
            )
        return
    failures = []
    try:
        for index, (strategy, product, start_date, end_date) in enumerate(jobs, start=1):
            name = str(strategy["name"])
            symbol = product["mainline_symbol"]
            if args.resume and _already_completed(name, product["product"], start_date, end_date):
                print(f"[{index}/{len(jobs)}] 跳过已完成: {name} {product['product']} {start_date}..{end_date}")
                continue
            print(f"[{index}/{len(jobs)}] 开始: {name} | {symbol} | {start_date}..{end_date}")
            try:
                run_job(strategy, product, start_date, end_date, args.init_balance)
            except Exception as exc:
                failures.append((name, product["product"], repr(exc)))
                print(f"[失败] {name} {product['product']}: {exc}")
                traceback.print_exc()
    except KeyboardInterrupt:
        print("\n[已中断] 当前回测已停止；未完成任务不会标记为成功。")
        raise SystemExit(130) from None

    if failures:
        print(f"失败任务: {len(failures)}/{len(jobs)}")
        for name, product, error in failures:
            print(f"{name}\t{product}\t{error}")
        raise SystemExit(1)
    print(f"全部任务完成: {len(jobs)}")


if __name__ == "__main__":
    main()
