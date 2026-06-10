# VICENT — Volatility-Informed Crypto Execution & Negotiation Trader

> Autonomous AI trading agent on BNB Smart Chain — BNB Hack: AI Trading Agent Edition

[![Track 1](https://img.shields.io/badge/Track-1%20Autonomous%20Trading-orange)](https://dorahacks.io/hackathon/bnbhack-twt-cmc)
[![BSC](https://img.shields.io/badge/Chain-BNB%20Smart%20Chain-yellow)](https://bscscan.com)
[![CMC](https://img.shields.io/badge/Data-CoinMarketCap%20API-blue)](https://coinmarketcap.com/api/agent)
[![TWAK](https://img.shields.io/badge/Execution-Trust%20Wallet%20AgentKit-teal)](https://portal.trustwallet.com)

VICENT is an advanced autonomous trading agent optimized for **Spot Swaps on the BNB Smart Chain (BSC)**. It combines real-time signals from the **CoinMarketCap (CMC) API**, self-custody execution via the **Trust Wallet Agent Kit (TWAK) CLI**, a self-correcting **Reflexion Engine (machine learning loop)**, and a **Market Defense Layer** to dynamically manage risks and protect capital.

---

## 🤝 Partner Tools & Integrations (API & CLI Mapping)

VICENT utilizes the complete partner ecosystem (CoinMarketCap + Trust Wallet Agent Kit + BNB Smart Chain) to form a unified autonomous agent:

### 1. CoinMarketCap (CMC) API
Used as the agent's sensory perception layer for macro sentiments and micro technical analysis:
- **Global Sentiment Metrics**: Consumes Fear & Greed index from the `/v1/global-metrics/quotes/latest` endpoint to evaluate macro risk level.
- **Micro Trends & Momentum**: Fetches RSI, MACD, and EMA indicators from the technical analysis data suite to generate signal inputs.
- **Sensory Volatility (ATR)**: Fetches rolling 1h, 24h, and 7-day price deviations from the `/v2/cryptocurrency/quotes/latest` endpoint to dynamically calculate trailing stops and stop losses.

### 2. Trust Wallet Agent Kit (TWAK) CLI
Used as the agent's secure custody and transaction execution layer:
- **On-chain Wallet Resolution**: Executes `twak wallet addresses` to fetch EVM addresses.
- **On-chain Balances**: Executes `twak wallet balance --chain bsc` to fetch real-world BNB and BEP-20 balances.
- **Self-Custody Swap Execution**: Executes `twak swap --chain bsc --usd <amount_usd> <from_token> <to_token> --password <pass>` for secure on-chain token swaps.
- **Competition Smart Contract Interface**: Executes `twak compete register` and `twak compete status` to register the agent's wallet address directly on the hackathon contract.

### 3. BNB Smart Chain (BSC)
The network layer of the agent, providing ultra-low transaction costs (gas fees) and deep liquidity routing on PancakeSwap V3 for BEP-20 assets (BNB, CAKE, BUSD, USDT, etc.).

---

## 🧠 AI Cognitive Reasoning Flow (How It Works)

Unlike standard trading bots that rely on static parameters, VICENT is designed as a **cognitive AI agent** following an iterative loop:

```
    ┌─────────────────────────────────────────────────────────────┐
    │ 1. PERCEIVE: Fetch CMC Global & Technical data series       │
    └──────────────────────────────┬──────────────────────────────┘
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 2. GUARD: Assess Market Health (DefensePosture)             │
    │    - Halt trading if BTC crash guard triggered              │
    │    - Scale down trade sizes if breadth is weak              │
    └──────────────────────────────┬──────────────────────────────┘
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 3. ANALYZE: Generate predictions & trade scores             │
    │    - Linear regression trend projections (R² filter)         │
    │    - Candle pattern recognition & breakout scores           │
    └──────────────────────────────┬──────────────────────────────┘
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 4. EXECUTE: Call TWAK CLI for on-chain BSC spot swaps       │
    └──────────────────────────────┬──────────────────────────────┘
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ 5. REFLEXION: Learn from closed trades                      │
    │    - Perform Trade Autopsies to find loss causes            │
    │    - Adjust weights (Signal Bias) & trigger cool-downs      │
    └─────────────────────────────────────────────────────────────┘
```

1.  **Perceive**: Collects multi-dimensional price and sentiment data via CoinMarketCap API.
2.  **Guard**: Checks global market risks (BTC flash crash, dumping cascades, extreme Fear & Greed) to establish the current `DefensePosture` (Normal, Caution, Defensive, Halt).
3.  **Analyze**: Uses **Ensemble Voting** combined with a **Linear Regression Trend Projection** filter. If the regression $R^2$ is low, the trend is considered noise, and the agent avoids entry.
4.  **Execute**: Trades are sized dynamically based on signal confidence and remaining account drawdown. The agent calls the **TWAK CLI** to swap USDT for selected BSC tokens.
5.  **Reflexion**: When a position is closed, the **Reflexion Engine** runs an autopsy. It identifies failing indicators and reduces their weights (Signal Bias), and places underperforming tokens on a temporary trade cooldown.

---

## 🚀 Quick Start

### 1. Install Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -e ".[onchain]"
```

### 2. Install TWAK CLI
Install the Trust Wallet Agent Kit CLI globally:
```bash
npm install -g @trustwallet/cli
```

### 3. Configure Environment Variables
Copy the template and fill in your details:
```bash
cp .env.example .env
```
Key configurations inside `.env`:
- `PRIVATE_KEY`: Your EVM wallet private key.
- `CMC_API_KEYS`: Comma-separated list of CoinMarketCap API keys (auto-rotated).
- `VICENT_MODE`: Set to `paper` for simulated trading or `live` for on-chain swaps.
- `VICENT_SERVER_PORT`: Port for the dashboard (defaults to `9090`).
- `VICENT_DB_PATH`: Custom path for SQLite databases (defaults to `vicent_spot_trades.db`).

### 4. Running the Agent & Dashboard
We provide isolated wrapper scripts to control the background processes without conflicting with other local projects (using local PID trackers):

*   **Start in Demo / Paper Trading Mode**:
    ```bash
    ./start.sh --mode paper --port 9090
    ```
*   **Start in Real Mainnet Trading Mode**:
    ```bash
    ./start.sh --mode live --port 9090
    ```
*   **Stop all background processes**:
    ```bash
    ./stop.sh
    ```

---

## 🏆 Competition Integration

VICENT integrates directly with the BNB Hack smart contract. You can execute registration commands and check status from the command line:

*   **Check on-chain registration status**:
    ```bash
    PYTHONPATH=src python3 src/vicent/cli.py compete status
    ```
*   **Register your agent's wallet address**:
    ```bash
    PYTHONPATH=src python3 src/vicent/cli.py compete register
    ```

---

## 📁 File Structure

```
src/vicent/
├── cli.py               # Typer CLI entry point
├── config.py            # Pydantic env configuration settings
├── agent.py             # Main execution state-machine loop
├── server.py            # FastAPI dashboard server & API
├── execution/
│   └── twak.py          # TWAK CLI subprocess bridge (swap, balance)
├── signals/
│   ├── cmc_client.py    # CMC Client with key rotation
│   ├── indicators.py    # Indicators calculation
│   ├── prediction.py    # Pattern matching & linear regression forecasting
│   └── call_scheduler.py# Scheduling CMC updates
├── strategy/
│   ├── defense.py       # Market defense posture engine
│   ├── ensemble.py      # Multi-indicator voting
│   ├── risk.py          # Sizing controls & drawdown stops
│   └── tokens.py        # Token watchlist & filters
└── state/
    ├── ledger.py        # SQLite trades & equity tracker
    └── price_history.py # SQLite price feeds history
```
