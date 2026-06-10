"""Configuration — loaded from .env via pydantic-settings."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Mode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class Strategy(str, Enum):
    TREND = "trend"
    MEANREV = "meanrev"
    ENSEMBLE = "ensemble"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Wallet ---
    private_key: str = Field(default="")
    wallet_address: str = Field(default="")

    # --- CMC ---
    cmc_api_keys: str = Field(default="")    # comma-separated list (at least 1 key)
    cmc_api_base: str = Field(default="https://pro-api.coinmarketcap.com")
    cmc_mcp_url: str = Field(default="https://mcp.coinmarketcap.com/mcp")
    cmc_x402_enabled: bool = Field(default=False)

    # --- Strategy ---
    vicent_mode: Mode = Field(default=Mode.PAPER)
    vicent_strategy: Strategy = Field(default=Strategy.ENSEMBLE)
    vicent_loop_interval_sec: int = Field(default=900)
    vicent_min_trades_per_day: int = Field(default=2)

    # --- Risk (hard limits) ---
    risk_max_drawdown_pct: float = Field(default=0.20)
    risk_daily_loss_pct: float = Field(default=0.05)
    risk_per_trade_nav_pct: float = Field(default=0.20)
    risk_max_slippage_bps: int = Field(default=80)
    risk_min_liquidity_usd: float = Field(default=250_000)

    # --- TWAK (Trust Wallet Agent Kit) ---
    twak_chain: str = Field(default="bsc")
    twak_password: str = Field(default="")
    twak_enabled: bool = Field(default=True)

    # --- Database ---
    vicent_db_path: str = Field(default="vicent_spot_trades.db")

    # --- Hyperliquid (Deprecated) ---
    hl_testnet: bool = Field(default=True)
    hl_enabled: bool = Field(default=False)

    # --- Server ---
    vicent_server_host: str = Field(default="127.0.0.1")
    vicent_server_port: int = Field(default=8080)
    log_level: str = Field(default="INFO")

    @classmethod
    def drawdown_cap(cls, v: float) -> float:
        # Hard-cap at 25% — we give ourselves a 5% buffer vs competition's 30%
        if v > 0.25:
            raise ValueError("risk_max_drawdown_pct must be <= 0.25 (competition cap is 0.30)")
        return v

    @field_validator("risk_per_trade_nav_pct")
    @classmethod
    def trade_size_cap(cls, v: float) -> float:
        if v > 0.30:
            raise ValueError("risk_per_trade_nav_pct must be <= 0.30")
        return v

    @property
    def is_live(self) -> bool:
        return self.vicent_mode == Mode.LIVE

    @property
    def slippage_pct(self) -> float:
        return self.risk_max_slippage_bps / 10_000

    @property
    def cmc_keys(self) -> list[str]:
        """Parse comma-separated CMC_API_KEYS into list of keys."""
        return [k.strip() for k in self.cmc_api_keys.split(",") if k.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
