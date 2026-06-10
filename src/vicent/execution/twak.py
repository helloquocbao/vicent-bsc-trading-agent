"""Trust Wallet Agent Kit (TWAK) executor.

Executes spot swaps, checks balances, and handles competition status
on BNB Smart Chain (BSC) by calling the `twak` CLI tool under the hood.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from typing import Any

import structlog

from vicent.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class TWAKResult:
    success: bool
    tx_hash: str | None = None
    error: str | None = None
    simulated: bool = False
    fill_price: float | None = None


class TWAKExecutor:
    """Execute spot trades and query wallet state via twak CLI."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._is_live = cfg.is_live
        self._chain = cfg.twak_chain
        self._password = cfg.twak_password
        self._twak_path = shutil.which("twak") or "twak"
        self._wallet_address: str | None = None

    async def _run_cli(self, args: list[str], input_str: str | None = None) -> dict[str, Any] | str:
        """Helper to execute twak command asynchronously and capture output."""
        cmd = [self._twak_path] + args
        try:
            log.debug("running_twak_cli", cmd=" ".join(cmd))
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(
                input=input_str.encode() if input_str else None
            )

            stdout_str = stdout.decode().strip()
            stderr_str = stderr.decode().strip()

            if proc.returncode != 0:
                log.warning("twak_cli_error", code=proc.returncode, stdout=stdout_str, stderr=stderr_str)
                # Try to parse stdout as JSON error if possible
                try:
                    return json.loads(stdout_str)
                except json.JSONDecodeError:
                    return {"error": stderr_str or stdout_str or f"Exit code {proc.returncode}"}

            try:
                return json.loads(stdout_str)
            except json.JSONDecodeError:
                return stdout_str

        except Exception as e:
            log.error("twak_cli_exception", cmd=" ".join(cmd), error=str(e))
            return {"error": str(e)}

    async def get_address(self) -> str:
        """Get wallet address for the configured chain."""
        if self._wallet_address:
            return self._wallet_address

        if not self._is_live:
            # Paper mode placeholder
            self._wallet_address = "0x5Bd4a9B01Ab4a05Fa8DC0F5094dC3B53aDB3b929"
            return self._wallet_address

        res = await self._run_cli(["wallet", "addresses", "--json"])
        if isinstance(res, dict) and "addresses" in res:
            for addr in res["addresses"]:
                if addr.get("chainId") == self._chain:
                    self._wallet_address = addr.get("address")
                    return self._wallet_address
        return "0x0000000000000000000000000000000000000000"

    async def get_equity(self) -> float:
        """Return total wallet balance in USD across native + token holdings."""
        if not self._is_live:
            return 1000.0  # Default paper trading initial balance

        res = await self._run_cli(["wallet", "balance", "--chain", self._chain, "--json"])
        if isinstance(res, dict):
            # Parse totalUsd
            total_usd = float(res.get("totalUsd") or 0.0)
            return total_usd
        return 0.0

    async def get_balance(self, token_symbol: str) -> float:
        """Get token balance for a specific token symbol on BSC."""
        if not self._is_live:
            return 0.0

        res = await self._run_cli(["wallet", "balance", "--chain", self._chain, "--json"])
        if isinstance(res, dict):
            # Check native
            if res.get("symbol", "").upper() == token_symbol.upper():
                return float(res.get("available") or 0.0)

            # Check ERC-20 tokens
            for tok in res.get("tokens", []):
                if tok.get("symbol", "").upper() == token_symbol.upper():
                    return float(tok.get("balance") or 0.0)
        return 0.0

    async def get_price(self, token_symbol: str) -> float:
        """Get current price in USD for a token via twak CLI."""
        res = await self._run_cli(["price", token_symbol, "--json"])
        if isinstance(res, dict) and "priceUsd" in res:
            return float(res["priceUsd"])
        return 0.0

    async def swap(
        self,
        from_token: str,
        to_token: str,
        amount_usd: float | None = None,
        amount: float | None = None,
        slippage: float = 1.0,
    ) -> TWAKResult:
        """Execute a swap transaction via twak CLI."""
        if not self._is_live:
            # Simulate a successful swap in paper mode
            price = await self.get_price(to_token) or 1.0
            return TWAKResult(success=True, simulated=True, fill_price=price, tx_hash="0xsimulatedtxhash")

        args = ["swap", "--chain", self._chain, "--slippage", str(slippage)]
        
        if amount_usd is not None:
            args += ["--usd", str(amount_usd)]
            args += [from_token, to_token]
        else:
            args += [str(amount), from_token, to_token]

        if self._password:
            args += ["--password", self._password]

        args += ["--json"]

        res = await self._run_cli(args)
        if isinstance(res, dict) and "error" not in res:
            # Expected success format contains txHash and execution details
            tx_hash = res.get("txHash") or res.get("hash")
            fill_price = None
            if "price" in res:
                fill_price = float(res["price"])
            elif "executionPrice" in res:
                fill_price = float(res["executionPrice"])
            
            log.info("twak_swap_success", from_token=from_token, to_token=to_token, tx_hash=tx_hash, fill_price=fill_price)
            return TWAKResult(success=True, tx_hash=tx_hash, fill_price=fill_price)
        else:
            err = res.get("error") if isinstance(res, dict) else str(res)
            log.error("twak_swap_failed", from_token=from_token, to_token=to_token, error=err)
            return TWAKResult(success=False, error=err)

    async def register_competition(self) -> bool:
        """Register the agent wallet for the BNB Hack competition."""
        args = ["compete", "register"]
        if self._password:
            args += ["--password", self._password]
        args += ["--json"]

        res = await self._run_cli(args)
        if isinstance(res, dict) and "error" not in res:
            log.info("competition_registration_success")
            return True
        err = res.get("error") if isinstance(res, dict) else str(res)
        log.error("competition_registration_failed", error=err)
        return False

    async def check_competition_status(self) -> dict[str, Any]:
        """Check whether the agent wallet is registered for the competition."""
        res = await self._run_cli(["compete", "status", "--json"])
        if isinstance(res, dict):
            return res
        return {"registered": False, "error": str(res)}
