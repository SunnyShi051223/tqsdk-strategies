"""Shared historical-backtest setup and artifact export for strategy examples."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

from tqsdk import BacktestFinished, TargetPosTask as _TqTargetPosTask, TqApi, TqAuth, TqBacktest, TqSim
from tqsdk.datetime import _get_trading_day_from_timestamp, _timestamp_nano_to_datetime

try:
    from .config_local import TQ_ACCOUNT, TQ_PASSWORD
except ImportError:
    # Strategy files are usually run directly, so this module is imported
    # without its package prefix and config_local is found beside it.
    from config_local import TQ_ACCOUNT, TQ_PASSWORD


_CHINA_TZ = ZoneInfo("Asia/Shanghai")
_ACTIVE_APIS: list["BacktestApi"] = []
_MISSING_INSTRUMENT_RE = re.compile(
    r"non-existent instrument:\s*([A-Za-z0-9]+\.[A-Za-z]+\d{2,4})",
    re.IGNORECASE,
)


class UnavailableHistoricalContract(RuntimeError):
    """A backtest's first mapped contract is absent from TqSdk metadata."""

    def __init__(self, instrument: str, detail: str) -> None:
        self.instrument = instrument
        super().__init__(detail)


def _flatten_symbols(value: Any) -> list[str]:
    """Extract contract identifiers from strings and nested strategy configs."""
    found: list[str] = []
    if isinstance(value, str):
        if "." in value and not any(char.isspace() for char in value):
            found.append(value.strip())
    elif isinstance(value, dict):
        for nested in value.values():
            found.extend(_flatten_symbols(nested))
    elif isinstance(value, (tuple, list, set)):
        for nested in value:
            found.extend(_flatten_symbols(nested))
    return list(dict.fromkeys(found))


def _format_time(value: Any) -> str:
    """Convert TqSdk nanosecond timestamps to a readable China-local time."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        moment = value
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_CHINA_TZ)
        else:
            moment = moment.astimezone(_CHINA_TZ)
        return moment.isoformat(sep=" ", timespec="seconds")

    if isinstance(value, (int, float)) or hasattr(value, "__int__") and not isinstance(value, str):
        try:
            raw = int(value)
            if raw <= 0:
                return ""
            # TqSdk order and trade times use nanoseconds since the Unix epoch.
            seconds = raw / 1_000_000_000 if raw > 10**15 else raw
            return datetime.fromtimestamp(seconds, timezone.utc).astimezone(_CHINA_TZ).isoformat(
                sep=" ", timespec="seconds"
            )
        except (TypeError, ValueError, OSError, OverflowError):
            return ""

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if moment.tzinfo is not None:
        moment = moment.astimezone(_CHINA_TZ)
    return moment.isoformat(sep=" ", timespec="seconds")


def _get_field(obj: Any, name: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        value = getattr(obj, name)
    except (AttributeError, TypeError):
        return default
    return default if value is None else value


def _contract_symbol(obj: Any) -> str:
    """Return the exchange-qualified contract symbol from a TqSdk order/trade."""
    exchange_id = str(_get_field(obj, "exchange_id", "")).strip()
    instrument_id = str(_get_field(obj, "instrument_id", "")).strip()
    if exchange_id and instrument_id:
        return f"{exchange_id}.{instrument_id}"
    symbol = str(_get_field(obj, "symbol", "")).strip()
    return symbol or instrument_id


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


class BacktestApi(TqApi):
    """TqApi configured for TqBacktest and recording account/trade history."""

    def __init__(
        self,
        *,
        strategy_name: str,
        symbols: Any,
        start_dt: date,
        end_dt: date,
        account: Any = None,
        init_balance: float = 10_000_000.0,
    ) -> None:
        self._backtest_closed = False
        self._backtest_initialized = False
        self._backtest_saved = False
        self._backtest_saving = False
        self._backtest_strategy = Path(strategy_name).stem
        self._backtest_symbols = _flatten_symbols(symbols)
        self._backtest_start = start_dt
        self._backtest_end = end_dt
        self._backtest_initial_balance = float(init_balance)
        self._backtest_rows: list[dict[str, Any]] = []
        self._rolling_target_tasks: list["RollingTargetPosTask"] = []
        self._rolling_target_by_symbol: dict[str, "RollingTargetPosTask"] = {}
        self._backtest_account = None
        self._backtest_quotes: dict[str, Any] = {}

        if not self._backtest_symbols:
            raise ValueError(f"{self._backtest_strategy}: no contract symbols were configured")
        if start_dt >= end_dt:
            raise ValueError("BACKTEST_START must be earlier than BACKTEST_END")
        if not math.isfinite(self._backtest_initial_balance) or self._backtest_initial_balance <= 0:
            raise ValueError("init_balance must be a positive finite amount")

        tq_account = TQ_ACCOUNT.strip()
        tq_password = TQ_PASSWORD
        if not tq_account or not tq_password:
            raise RuntimeError(
                "请将 strategies/config_local.example.py 复制为 "
                "strategies/config_local.py，并填写快期账号和密码。"
            )

        try:
            super().__init__(
                account=account if account is not None else TqSim(init_balance=self._backtest_initial_balance),
                backtest=TqBacktest(start_dt=start_dt, end_dt=end_dt),
                auth=TqAuth(tq_account, tq_password),
            )
            self._backtest_account = self.get_account()
            self._backtest_quotes = {}
            for symbol in self._backtest_symbols:
                try:
                    self._backtest_quotes[symbol] = self.get_quote(symbol)
                except Exception as exc:
                    missing = _MISSING_INSTRUMENT_RE.search(str(exc))
                    if missing:
                        raise UnavailableHistoricalContract(
                            missing.group(1), str(exc)
                        ) from exc
                    raise
            self._backtest_initialized = True
        except BaseException:
            # TqApi starts asyncio tasks during __init__. If account setup or
            # quote resolution fails before this object can be registered in
            # _ACTIVE_APIS, close the partially initialized API here.
            self._backtest_closed = True
            if getattr(self, "_loop", None) is not None:
                try:
                    TqApi.close(self)
                except BaseException:
                    # Preserve the original startup error (including Ctrl+C).
                    pass
            raise

    def _get_rolling_target_task(
        self, symbol: str, *args: Any, **kwargs: Any
    ) -> "RollingTargetPosTask":
        task = self._rolling_target_by_symbol.get(symbol)
        if task is None:
            task = RollingTargetPosTask(self, symbol, *args, **kwargs)
        return task

    def get_position(self, symbol: str | None = None, account: Any = None) -> Any:
        if isinstance(symbol, str) and symbol.startswith("KQ.m@"):
            task = self._get_rolling_target_task(symbol)
            return RollingPositionView(self, task, account)
        return super().get_position(symbol=symbol, account=account)

    def insert_order(
        self,
        symbol: str,
        direction: str,
        offset: str = "",
        volume: int = 0,
        limit_price: Any = None,
        advanced: str | None = None,
        order_id: str | None = None,
        account: Any = None,
        offset_leg2: str = "",
    ) -> Any:
        if not symbol.startswith("KQ.m@"):
            return super().insert_order(
                symbol=symbol,
                direction=direction,
                offset=offset,
                volume=volume,
                limit_price=limit_price,
                advanced=advanced,
                order_id=order_id,
                account=account,
                offset_leg2=offset_leg2,
            )
        if direction not in {"BUY", "SELL"} or offset not in {"OPEN", "CLOSE", "CLOSETODAY"}:
            raise ValueError("主连映射下单只支持期货 BUY/SELL 与 OPEN/CLOSE/CLOSETODAY。")
        task = self._get_rolling_target_task(symbol, price="ACTIVE")
        task.adjust_for_order(direction, offset, int(volume))
        task.update()
        actual_symbol = task.actual_symbol or task.pending_symbol or ""
        return SimpleNamespace(
            order_id=order_id or "rolling-target-order",
            symbol=actual_symbol,
            direction=direction,
            offset=offset,
            volume_orign=int(volume),
            volume_left=int(volume),
            status="ALIVE",
            limit_price=limit_price,
            last_msg="目标仓位已更新，实际委托由主力合约滚动任务管理",
        )

    def wait_update(self, *args: Any, **kwargs: Any) -> bool:
        try:
            updated = super().wait_update(*args, **kwargs)
        except BacktestFinished:
            self._update_rolling_target_tasks()
            self._capture_equity()
            raise
        self._update_rolling_target_tasks()
        self._capture_equity()
        return updated

    def _update_rolling_target_tasks(self) -> None:
        for task in tuple(self._rolling_target_tasks):
            task.update()

    def _current_market_time(self) -> str:
        times = [
            formatted
            for quote in self._backtest_quotes.values()
            if (formatted := _format_time(_get_field(quote, "datetime")))
        ]
        return max(times) if times else ""

    def _capture_equity(self) -> None:
        if self._backtest_account is None:
            return
        timestamp = self._current_market_time()
        if not timestamp:
            return
        account = self._backtest_account
        row = {
            "timestamp": timestamp,
            "equity": _as_float(_get_field(account, "balance", 0.0)),
            "available": _as_float(_get_field(account, "available", 0.0)),
            "float_profit": _as_float(_get_field(account, "float_profit", 0.0)),
            "position_profit": _as_float(_get_field(account, "position_profit", 0.0)),
            "close_profit": _as_float(_get_field(account, "close_profit", 0.0)),
            "commission": _as_float(_get_field(account, "commission", 0.0)),
        }
        if self._backtest_rows and self._backtest_rows[-1]["timestamp"] == timestamp:
            self._backtest_rows[-1] = row
        else:
            self._backtest_rows.append(row)

    def _output_directory(self) -> Path:
        contracts = [
            re.sub(r'[<>:"/\\|?*]+', "_", s.removeprefix("KQ.m@"))
            for s in self._backtest_symbols
        ]
        if len(contracts) == 1:
            contract_name = contracts[0]
        else:
            contract_name = "__".join(contracts)
            if len(contract_name) > 90:
                short = hashlib.sha1("|".join(contracts).encode("utf-8")).hexdigest()[:8]
                contract_name = f"{contracts[0]}__等{len(contracts)}个合约_{short}"
        root = Path(__file__).resolve().parents[1]
        return root / "Backtest" / self._backtest_strategy / contract_name

    def _trade_rows(self) -> list[dict[str, Any]]:
        try:
            trades = self.get_trade()
        except Exception:
            return []
        values = trades.values() if hasattr(trades, "values") else ()
        rows = []
        for trade in values:
            rows.append(
                {
                    "time": _format_time(_get_field(trade, "trade_date_time")),
                    "symbol": _contract_symbol(trade),
                    "trade_id": _get_field(trade, "trade_id"),
                    "order_id": _get_field(trade, "order_id"),
                    "direction": _get_field(trade, "direction"),
                    "offset": _get_field(trade, "offset"),
                    "price": _get_field(trade, "price"),
                    "volume": _get_field(trade, "volume"),
                    "commission": _get_field(trade, "commission"),
                }
            )
        return sorted(rows, key=lambda row: row["time"])

    def _order_rows(self) -> list[dict[str, Any]]:
        try:
            orders = self.get_order()
        except Exception:
            return []
        values = orders.values() if hasattr(orders, "values") else ()
        rows = []
        for order in values:
            rows.append(
                {
                    "time": _format_time(_get_field(order, "insert_date_time")),
                    "symbol": _contract_symbol(order),
                    "order_id": _get_field(order, "order_id"),
                    "direction": _get_field(order, "direction"),
                    "offset": _get_field(order, "offset"),
                    "status": _get_field(order, "status"),
                    "volume": _get_field(order, "volume_orign"),
                    "volume_left": _get_field(order, "volume_left"),
                    "limit_price": _get_field(order, "limit_price"),
                    "trade_price": _get_field(order, "trade_price"),
                    "last_msg": _get_field(order, "last_msg"),
                }
            )
        return sorted(rows, key=lambda row: row["time"])

    @staticmethod
    def _write_equity_chart(path: Path, rows: list[dict[str, Any]]) -> None:
        width, height = 1100, 560
        left, right, top, bottom = 92, 28, 36, 72
        values = [row["equity"] for row in rows]
        if not values:
            content = '<text x="550" y="280" text-anchor="middle">回测区间内没有权益数据</text>'
        else:
            low, high = min(values), max(values)
            if math.isclose(low, high):
                margin = max(abs(low) * 0.01, 1.0)
                low -= margin
                high += margin
            points = []
            for index, value in enumerate(values):
                x = left + (width - left - right) * index / max(len(values) - 1, 1)
                y = top + (height - top - bottom) * (high - value) / (high - low)
                points.append(f"{x:.2f},{y:.2f}")
            mid_y = top + (height - top - bottom) / 2
            first_time = html.escape(rows[0]["timestamp"])
            last_time = html.escape(rows[-1]["timestamp"])
            content = (
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#64748b"/>'
                f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#64748b"/>'
                f'<line x1="{left}" y1="{mid_y:.2f}" x2="{width-right}" y2="{mid_y:.2f}" stroke="#e2e8f0"/>'
                f'<polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{" ".join(points)}"/>'
                f'<text x="{left-10}" y="{top+5}" text-anchor="end">{high:,.2f}</text>'
                f'<text x="{left-10}" y="{height-bottom+5}" text-anchor="end">{low:,.2f}</text>'
                f'<text x="{left}" y="{height-30}">{first_time}</text>'
                f'<text x="{width-right}" y="{height-30}" text-anchor="end">{last_time}</text>'
            )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<g font-family="Microsoft YaHei,Arial,sans-serif" font-size="14" fill="#334155">'
            '<text x="30" y="24" font-size="18" font-weight="600">账户权益曲线</text>'
            f'{content}</g></svg>'
        )
        path.write_text(svg, encoding="utf-8")

    def save_backtest_results(self) -> Path | None:
        if self._backtest_saved or self._backtest_saving:
            return self._output_directory()
        self._backtest_saving = True
        try:
            self._capture_equity()
            output = self._output_directory()
            output.mkdir(parents=True, exist_ok=True)

            initial_equity = self._backtest_rows[0]["equity"] if self._backtest_rows else 0.0
            peak = initial_equity
            previous = initial_equity
            return_rows = []
            for row in self._backtest_rows:
                equity = row["equity"]
                peak = max(peak, equity)
                period_return = equity / previous - 1.0 if previous else 0.0
                cumulative_return = equity / initial_equity - 1.0 if initial_equity else 0.0
                drawdown = equity / peak - 1.0 if peak else 0.0
                return_rows.append(
                    {
                        "timestamp": row["timestamp"],
                        "equity": f"{equity:.8f}",
                        "pnl": f"{equity - initial_equity:.8f}",
                        "period_return": f"{period_return:.10f}",
                        "cumulative_return": f"{cumulative_return:.10f}",
                        "drawdown": f"{drawdown:.10f}",
                    }
                )
                previous = equity

            _write_csv(
                output / "equity_curve.csv",
                ["timestamp", "equity", "available", "float_profit", "position_profit", "close_profit", "commission"],
                self._backtest_rows,
            )
            _write_csv(
                output / "returns.csv",
                ["timestamp", "equity", "pnl", "period_return", "cumulative_return", "drawdown"],
                return_rows,
            )
            trade_rows = self._trade_rows()
            _write_csv(
                output / "trades.csv",
                ["time", "symbol", "trade_id", "order_id", "direction", "offset", "price", "volume", "commission"],
                trade_rows,
            )
            order_rows = self._order_rows()
            _write_csv(
                output / "orders.csv",
                ["time", "symbol", "order_id", "direction", "offset", "status", "volume", "volume_left", "limit_price", "trade_price", "last_msg"],
                order_rows,
            )
            self._write_equity_chart(output / "equity_curve.svg", self._backtest_rows)

            final_equity = self._backtest_rows[-1]["equity"] if self._backtest_rows else 0.0
            max_drawdown = min((float(row["drawdown"]) for row in return_rows), default=0.0)
            summary = {
                "strategy": self._backtest_strategy,
                "symbols": self._backtest_symbols,
                "start_date": self._backtest_start.isoformat(),
                "end_date": self._backtest_end.isoformat(),
                "configured_initial_balance": self._backtest_initial_balance,
                "equity_points": len(self._backtest_rows),
                "initial_equity": initial_equity,
                "final_equity": final_equity,
                "net_profit": final_equity - initial_equity,
                "total_return": final_equity / initial_equity - 1.0 if initial_equity else 0.0,
                "max_drawdown": max_drawdown,
                "trade_count": len(trade_rows),
                "order_count": len(order_rows),
                "generated_at": datetime.now(_CHINA_TZ).isoformat(sep=" ", timespec="seconds"),
            }
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._backtest_saved = True
            print(f"[回测结果] 已保存到: {output}")
            return output
        finally:
            self._backtest_saving = False

    def close(self) -> None:
        if getattr(self, "_backtest_closed", False):
            return
        self._backtest_closed = True
        try:
            # Strategy exceptions and Ctrl+C can run close() from a finally
            # block. Keep partial files out of the completed-results folder;
            # otherwise --resume could mistake them for successful runs.
            if (
                getattr(self, "_backtest_initialized", False)
                and sys.exc_info()[0] is None
            ):
                self.save_backtest_results()
        finally:
            super().close()


def create_backtest_api(
    strategy_name: str,
    symbols: Any,
    start_dt: date,
    end_dt: date,
    *,
    account: Any = None,
    init_balance: float = 10_000_000.0,
) -> BacktestApi:
    """Create a local TqSim account running over the configured historical range."""
    api = BacktestApi(
        strategy_name=strategy_name,
        symbols=symbols,
        start_dt=start_dt,
        end_dt=end_dt,
        account=account,
        init_balance=init_balance,
    )
    _ACTIVE_APIS.append(api)
    return api


class RollingTargetPosTask:
    """Route mainline signals to each day's tradable contract and roll positions."""

    def __new__(cls, api: BacktestApi, symbol: str, *args: Any, **kwargs: Any):
        if symbol.startswith("KQ.m@"):
            existing = getattr(api, "_rolling_target_by_symbol", {}).get(symbol)
            if existing is not None:
                return existing
        return super().__new__(cls)

    def __init__(self, api: BacktestApi, symbol: str, *args: Any, **kwargs: Any) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._api = api
        self._symbol = symbol
        self._args = args
        self._kwargs = kwargs
        self._desired_volume = 0
        self._actual_symbol: str | None = None
        self._pending_symbol: str | None = None
        self._target_task: Any = None
        self._contract_symbols: set[str] = set()
        self._cancelled = False

        if not symbol.startswith("KQ.m@"):
            self._delegate = _TqTargetPosTask(api, symbol, *args, **kwargs)
            return

        self._delegate = None
        self._product = symbol.removeprefix("KQ.m@")
        self._contracts_by_date = self._load_contract_mapping(self._product)
        api._rolling_target_tasks.append(self)
        api._rolling_target_by_symbol[symbol] = self

    @staticmethod
    def _load_contract_mapping(product: str) -> dict[str, str]:
        root = Path(__file__).resolve().parents[1]
        path = root / "Backtest" / "_data" / "main_contract_history.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"缺少主力合约映射文件 {path}；请先运行 prepare_futures_catalog.py。"
            )
        result: dict[str, str] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("product", "").upper() == product.upper():
                    trading_date = row.get("trading_date", "").strip()
                    actual_contract = row.get("actual_contract", "").strip()
                    if trading_date and actual_contract:
                        result[trading_date] = actual_contract
        if not result:
            raise ValueError(
                f"映射文件中没有 {product} 的记录；请先为该品种生成主力合约映射。"
            )
        return result

    def _current_trading_date(self) -> str | None:
        state = _get_field(getattr(self._api, "_data", {}), "_tqsdk_backtest", {})
        current_ns = _get_field(state, "current_dt", 0)
        try:
            current_ns = int(current_ns)
        except (TypeError, ValueError):
            return None
        if current_ns <= 0:
            return None
        trading_day_ns = _get_trading_day_from_timestamp(current_ns)
        return _timestamp_nano_to_datetime(trading_day_ns).date().isoformat()

    def _start_contract_task(self, actual_symbol: str) -> None:
        self._actual_symbol = actual_symbol
        self._contract_symbols.add(actual_symbol)
        self._pending_symbol = None
        self._target_task = _TqTargetPosTask(
            self._api, actual_symbol, *self._args, **self._kwargs
        )
        if self._desired_volume:
            self._target_task.set_target_volume(self._desired_volume)

    def update(self) -> None:
        if self._delegate is not None or self._cancelled:
            return
        trading_date = self._current_trading_date()
        if trading_date is None:
            return
        actual_symbol = self._contracts_by_date.get(trading_date)
        if actual_symbol is None:
            start = self._api._backtest_start.isoformat()
            end = self._api._backtest_end.isoformat()
            if start <= trading_date <= end:
                raise ValueError(
                    f"{self._product} 在回测交易日 {trading_date} 没有主力合约映射；"
                    "请重新生成覆盖完整回测区间的映射。"
                )
            return

        if self._actual_symbol is None:
            self._start_contract_task(actual_symbol)
            return
        if self._pending_symbol is not None:
            self._pending_symbol = actual_symbol
            position = self._api.get_position(self._actual_symbol)
            try:
                current_volume = float(_get_field(position, "pos", float("nan")))
            except (TypeError, ValueError):
                return
            if not math.isfinite(current_volume) or not math.isclose(current_volume, 0.0):
                return
            old_task = self._target_task
            self._target_task = None
            if old_task is not None:
                old_task.cancel()
            self._actual_symbol = None
            self._start_contract_task(self._pending_symbol)
            return
        if actual_symbol != self._actual_symbol:
            self._pending_symbol = actual_symbol
            if self._target_task is not None:
                self._target_task.set_target_volume(0)
            else:
                self._start_contract_task(actual_symbol)

    def set_target_volume(self, volume: int) -> None:
        if self._delegate is not None:
            self._delegate.set_target_volume(volume)
            return
        self._desired_volume = int(volume)
        if self._target_task is not None and self._pending_symbol is None:
            self._target_task.set_target_volume(self._desired_volume)

    @property
    def desired_volume(self) -> int:
        if self._delegate is not None:
            return 0
        return self._desired_volume

    @property
    def actual_symbol(self) -> str | None:
        return self._actual_symbol

    @property
    def pending_symbol(self) -> str | None:
        return self._pending_symbol

    def adjust_for_order(self, direction: str, offset: str, volume: int) -> None:
        if volume <= 0:
            raise ValueError("volume must be a positive integer")
        current = self.desired_volume
        if offset == "OPEN":
            target = current + volume if direction == "BUY" else current - volume
        elif direction == "SELL":
            target = max(0, current - volume) if current > 0 else current
        else:
            target = min(0, current + volume) if current < 0 else current
        self.set_target_volume(target)

    def cancel(self) -> None:
        self._cancelled = True
        if self._delegate is not None:
            self._delegate.cancel()
        elif self._target_task is not None:
            self._target_task.cancel()


class RollingPositionView:
    """Read-only aggregate position view across contracts visited by a mainline."""

    def __init__(self, api: BacktestApi, task: RollingTargetPosTask, account: Any = None) -> None:
        self._api = api
        self._task = task
        self._account = account

    def __getattr__(self, name: str) -> Any:
        if name == "symbol":
            return self._task._symbol
        values = []
        for symbol in sorted(self._task._contract_symbols):
            position = self._api.get_position(symbol, account=self._account)
            value = _get_field(position, name, None)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                values.append(numeric)
        if values:
            return sum(values)
        if name == "pos" or name.startswith("pos_"):
            return 0.0
        return float("nan")


def run_backtest(entry: Callable[[], Any]) -> Any:
    """Run a strategy, treat BacktestFinished as normal, then save and close."""
    _ACTIVE_APIS.clear()
    try:
        return entry()
    except BacktestFinished:
        print("[回测完成] 已到达设定的回测结束日期。")
        return None
    finally:
        for api in reversed(_ACTIVE_APIS):
            api.close()
        _ACTIVE_APIS.clear()

