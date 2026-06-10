"""VICENT CLI — typer-based entry point.

Usage:
  vicent run                  # run agent (paper by default)
  vicent run --mode live      # run in live mode
  vicent status               # print portfolio snapshot
  vicent trades               # print recent trades
  vicent serve                # start status HTTP server only
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import structlog
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from vicent.config import Mode, get_settings
from vicent.state.ledger import get_all_trades, get_latest_snapshot, init_db

app = typer.Typer(
    name="vicent",
    help="VICENT — Autonomous AI Spot Trading Agent on BNB Smart Chain using TWAK",
    no_args_is_help=True,
)

console = Console()
log = structlog.get_logger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    import logging
    import structlog

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


@app.command()
def run(
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="paper | live"),
    capital: float = typer.Option(20.0, "--capital", help="Initial capital in USD (paper mode only)"),
    interval: Optional[int] = typer.Option(None, "--interval", help="Loop interval in seconds"),
) -> None:
    """Run the VICENT trading agent."""
    cfg = get_settings()
    _setup_logging(cfg.log_level)

    if mode:
        import os
        os.environ["VICENT_MODE"] = mode
        get_settings.cache_clear()
        cfg = get_settings()

    if interval:
        import os
        os.environ["VICENT_LOOP_INTERVAL_SEC"] = str(interval)
        get_settings.cache_clear()
        cfg = get_settings()

    console.print(f"[bold green]VICENT Agent starting[/bold green]")
    console.print(f"  Mode:      [bold]{cfg.vicent_mode.value}[/bold]")
    console.print(f"  Strategy:  [bold]{cfg.vicent_strategy.value}[/bold]")
    console.print(f"  Interval:  [bold]{cfg.vicent_loop_interval_sec}s[/bold]")
    console.print(f"  Capital:   [bold]${capital:.2f}[/bold] (paper)")
    console.print(f"  Chain:     [bold]{cfg.twak_chain}[/bold] — {'TWAK enabled' if cfg.twak_enabled else 'TWAK disabled'}")

    if cfg.is_live and not cfg.cmc_keys:
        console.print("[red]ERROR: CMC_API_KEYS not set in .env[/red]")
        raise typer.Exit(1)

    from vicent.agent import VICENTAgent

    agent = VICENTAgent(initial_capital_usd=capital)

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent stopped by user.[/yellow]")


@app.command()
def compete(
    action: str = typer.Argument(..., help="register | status")
) -> None:
    """Register or check status for the BNB Hack competition."""
    from vicent.execution.twak import TWAKExecutor
    executor = TWAKExecutor()
    
    if action == "register":
        console.print("[yellow]Registering agent wallet on-chain for the competition...[/yellow]")
        success = asyncio.run(executor.register_competition())
        if success:
            console.print("[bold green]Successfully registered for BNB HACK: AI Trading Agent Edition![/bold green]")
        else:
            console.print("[bold red]Registration failed. Please check twak configuration and try again.[/bold red]")
    elif action == "status":
        console.print("[yellow]Checking competition registration status...[/yellow]")
        status_info = asyncio.run(executor.check_competition_status())
        if status_info.get("registered"):
            console.print("[bold green]Status: REGISTERED[/bold green]")
        else:
            console.print(f"[bold red]Status: NOT REGISTERED[/bold red] (Deadline: {status_info.get('deadline', 'N/A')})")
    else:
        console.print(f"[red]Unknown competition action: {action}. Use 'register' or 'status'.[/red]")


@app.command()
def status() -> None:
    """Print current portfolio snapshot."""
    _setup_logging()
    init_db()
    snapshot = get_latest_snapshot()
    if not snapshot:
        console.print("[yellow]No snapshot yet — agent hasn't run.[/yellow]")
        return
    console.print_json(json.dumps(snapshot, indent=2))


@app.command()
def trades(limit: int = typer.Option(20, "--limit", "-n")) -> None:
    """Print recent trades."""
    _setup_logging()
    init_db()
    rows = get_all_trades(limit=limit)
    if not rows:
        console.print("[yellow]No trades yet.[/yellow]")
        return

    table = Table(title=f"Last {limit} Trades")
    table.add_column("ID", style="dim")
    table.add_column("Time", style="dim")
    table.add_column("Symbol")
    table.add_column("Dir")
    table.add_column("USD")
    table.add_column("Price")
    table.add_column("Conf")
    table.add_column("Mode")
    table.add_column("TxHash", style="dim")

    for r in rows:
        table.add_row(
            str(r.get("id", "")),
            str(r.get("ts", ""))[:16],
            r.get("symbol", ""),
            r.get("direction", ""),
            f"${r.get('amount_usd', 0):.2f}",
            f"${r.get('price_usd', 0):.4f}",
            f"{r.get('confidence', 0):.2f}" if r.get("confidence") else "-",
            r.get("mode", ""),
            (str(r.get("tx_hash", "")) or "")[:12],
        )

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
) -> None:
    """Start the monitoring dashboard server at http://localhost:8080"""
    _setup_logging()
    console.print(f"[bold green]VICENT Dashboard[/bold green] → http://{host}:{port}")
    console.print(f"  Auto-refresh every 10 seconds")
    console.print(f"  Ctrl+C to stop")
    uvicorn.run("vicent.server:app", host=host, port=port, reload=False, log_level="warning")


if __name__ == "__main__":
    app()
