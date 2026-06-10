"""Token watchlist — không giới hạn bởi competition whitelist.

Phân tier theo liquidity, signal quality và perps support trên BSC.

Tier 1 — Blue chip, volume cao nhất, perps support đầy đủ
Tier 2 — Mid cap, liquidity tốt, theo dõi định kỳ
Tier 3 — Emerging/trending, chỉ trade khi signal rất mạnh
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Token:
    symbol: str
    tier: int
    min_liquidity_usd: float = 250_000
    has_bsc_perps: bool = False   # có perps native trên BSC không


# ── TIER 1: Blue chip, volume > $50M/day, strong signal ─────────────────────
TIER1 = [
    # Majors
    Token("BNB",    1, 100_000_000, True),   # native BSC, deepest liquidity
    Token("ETH",    1,  50_000_000, True),   # bridged ETH on BSC
    Token("BTC",    1,  50_000_000, True),   # BTCB on BSC

    # DeFi blue chips
    Token("CAKE",   1,  10_000_000, True),   # PancakeSwap — BSC native DEX
    Token("AAVE",   1,   5_000_000, True),
    Token("UNI",    1,   5_000_000, True),
    Token("LINK",   1,  10_000_000, True),

    # L1 ecosystem
    Token("SOL",    1,  20_000_000, True),   # high vol, clear signals
    Token("AVAX",   1,  10_000_000, True),
    Token("DOT",    1,   5_000_000, True),
    Token("ATOM",   1,   5_000_000, True),
    Token("INJ",    1,   3_000_000, True),

    # High volume altcoins
    Token("DOGE",   1,  20_000_000, True),
    Token("XRP",    1,  20_000_000, True),
    Token("ADA",    1,  10_000_000, True),
    Token("TRX",    1,  10_000_000, True),
    Token("LTC",    1,   5_000_000, True),
    Token("TON",    1,  10_000_000, True),

    # AI narrative
    Token("FET",    1,   5_000_000, True),
    Token("NEAR",   1,   5_000_000, True),
    Token("ARB",    1,  10_000_000, False),  # bridged token trên BSC có liquidity thấp
    Token("OP",     1,   5_000_000, False),  # tương tự ARB
]

# ── TIER 2: Mid cap, volume $5M–$50M/day ────────────────────────────────────
TIER2 = [
    # DeFi
    Token("LDO",    2,   2_000_000),
    Token("PENDLE", 2,   2_000_000),
    Token("SNX",    2,   1_000_000),
    Token("COMP",   2,   1_000_000),
    Token("1INCH",  2,   1_000_000),
    Token("SUSHI",  2,   1_000_000),
    Token("CRV",    2,   2_000_000),
    Token("BAL",    2,     500_000),
    Token("RUNE",   1,   2_000_000),

    # Gaming/NFT
    Token("AXS",    2,   2_000_000),
    Token("SAND",   2,   2_000_000),
    Token("MANA",   2,   1_000_000),
    Token("APE",    2,   2_000_000),
    Token("GMT",    2,   1_000_000),

    # Layer 2 / Infra
    Token("MATIC",  2,   5_000_000),
    Token("FTM",    2,   2_000_000),
    Token("ZIL",    2,     500_000),
    Token("FIL",    2,   2_000_000),
    Token("EGLD",   2,   1_000_000),

    # Memes avec volume
    Token("FLOKI",  2,   2_000_000),
    Token("BONK",   2,   2_000_000),
    Token("SHIB",   2,   5_000_000),
]

# ── TIER 3: Emerging/trending, chỉ khi signal rất cao ───────────────────────
TIER3 = [
    Token("PENGU",  3,     500_000),
    Token("RAY",    3,     500_000),
    Token("STG",    3,     500_000),
    Token("ZRO",    3,     500_000),
    Token("PEAQ",   3,     250_000),
    Token("KAVA",   3,     250_000),
]

# Stablecoins — base token, không trade
STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "TUSD",
    "FRAX", "FRXUSD", "USDD", "USD1", "USDe",
    "BUSD", "PAXG", "LUSD",
}

ALL_TOKENS: list[Token] = TIER1 + TIER2 + TIER3
TOKEN_MAP: dict[str, Token] = {t.symbol: t for t in ALL_TOKENS}


def get_watchlist(max_tier: int = 1) -> list[Token]:
    """Return tokens up to and including max_tier."""
    return [t for t in ALL_TOKENS if t.tier <= max_tier]


def get_perps_eligible(max_tier: int = 2) -> list[Token]:
    """Return tokens that have BSC perps support."""
    return [t for t in ALL_TOKENS if t.tier <= max_tier and t.has_bsc_perps]


def is_stable(symbol: str) -> bool:
    return symbol.upper() in STABLES
