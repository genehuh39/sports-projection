# sports-projection

[![tests](https://github.com/genehuh39/sports-projection/actions/workflows/test.yml/badge.svg)](https://github.com/genehuh39/sports-projection/actions/workflows/test.yml)
[![release](https://img.shields.io/github/v/release/genehuh39/sports-projection)](https://github.com/genehuh39/sports-projection/releases)
[![python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

NBA game projections and betting-value dashboard. Pulls live schedules and odds, runs an XGBoost projection model with schedule-strength and injury-aware adjustments, and surfaces edge / EV against bookmaker prices.

![dashboard](assets/dashboard.png)

## What's interesting about it

This is a small project, but it's been built honestly. Two things worth pointing at:

- **Walk-forward CV before tuning.** `sports-eval` runs 6 chronological folds and reports mean ± std MAE, so the noise floor (~±0.6 margin points) is visible before claiming any improvement. Several "obviously good" features turned out to be statistically indistinguishable from the baseline; the README and commit history reflect those negative results rather than hiding them.
- **A calibration knob that was wrong, found and fixed.** The injury-adjustment damping factor shipped at 0.35 (from public research). Walk-forward calibration showed it was actively degrading predictions by ~2.7 margin points. Lowered to 0.05; even at the optimum, the adjustment is roughly a no-op. See `sports-calibrate`.

## Quick start

On macOS, install the OpenMP runtime that XGBoost requires:

```bash
brew install libomp
```

Install dependencies:

```bash
uv sync
```

## Commands

| Command | What it does |
|---|---|
| `uv run sports-projection` | Terminal pipeline — fetches games, prints projections + EV |
| `uv run sports-dashboard`  | NiceGUI dashboard at `localhost:8080` (auto-picks next port if busy) |
| `uv run sports-train`      | Trains the XGBoost model and saves to `artifacts/nba_projection_model.joblib` |
| `uv run sports-eval`       | Walk-forward CV over 6 folds, reports per-fold and aggregate MAE |
| `uv run sports-calibrate`  | Sweeps injury damping factors against historical box scores |
| `uv run sports-backtest`   | Calibration metrics (Brier, log loss, accuracy) + synthetic-market ROI by edge |
| `uv run sports-kalshi`     | Compare model probabilities against live Kalshi NBA game prices |
| `uv run sports-polymarket` | Compare model probabilities against live Polymarket NBA game prices |
| `uv run sports-paper-trade [edge]` | Log "would have placed" entries above edge threshold (default 0.05) |
| `uv run sports-settle`     | Resolve past-dated open paper trades against actual game outcomes |
| `uv run sports-pnl`        | Cumulative ROI, win rate, edge-bucket and venue breakdown over the journal |
| `uv run sports-test`       | Test suite |

`sports-eval` and `sports-calibrate` accept positional args, e.g. `uv run sports-eval 8 100` for 8 folds × 100 games each.

## How the model works

For each upcoming game, the projection engine builds per-team pregame features and feeds them to two XGBoost regressors (one for margin, one for total).

**Features (37 total):**
- Recent form (10-game rolling): offense, defense, margin, pace, win rate
- Season form (expanding mean): offense, defense, margin, pace, win rate
- Rest-day count and back-to-back flags
- Games played to-date
- Simple Rating System (SRS) rating, computed leakage-free per game date
- Differentials between home and away on each of the above

**Post-prediction adjustments:**
- ESPN injury feed → per-team "points absent," weighted by status (Out / Doubtful / Questionable / Day-to-day)
- Damping factor `0.05` calibrated empirically — see `sports-calibrate`
- Players inactive in the last 10 games are filtered out (their absence is already baked into the rolling features)

**Current out-of-sample performance** (6-fold walk-forward CV on 2024-25 + 2025-26):
- Margin MAE: **12.13 ± 0.59** points
- Total MAE: **16.54 ± 0.89** points

The ±0.59 std is the noise floor — any future change smaller than ~1.2 margin points (2σ) is statistically indistinguishable.

## Prediction-market integration

The project ships read-only clients for both Kalshi (`KXNBAGAME` series) and Polymarket (gamma API) NBA game markets. For each upcoming game the model has an opinion on, it pulls the venue's price and computes per-side edge.

Empirical liquidity finding (April 2026):

| | Kalshi | Polymarket |
|---|---|---|
| Events listed | All upcoming NBA games | All upcoming NBA games |
| Markets with quotes | **0** | **All**, six-figure 24h volumes |

Kalshi NBA game-line markets are essentially empty right now. Polymarket has real prices, real volume, and is the venue worth measuring against today.

### Paper-trading workflow

Three commands form a daily loop for measuring the model against live prices without placing real money:

```bash
uv run sports-paper-trade 0.05    # nightly: log positive-edge bets above 5%
uv run sports-settle              # any time: resolve past games
uv run sports-pnl                 # report cumulative results
```

Trades are stored in a local SQLite journal (`data/paper_trades.db`, gitignored). PnL accounting follows prediction-market conventions: a YES contract bought at price `p` returns `(1-p)/p` if it wins, `-1.0` if it loses.

The `(venue, game_id, side)` uniqueness constraint makes `sports-paper-trade` idempotent — re-running it the same day appends only new entries. `sports-settle` looks up actual outcomes via `nba_api.leaguegamefinder` (regular season + playoffs) and writes realized PnL.

This is where the project genuinely answers "does this model make money?" — a question that's unanswerable by MAE alone and unsupported by historical odds APIs. Run it nightly, settle daily, and after a few hundred bets the answer becomes statistical.

## Dashboard

Filters: positive-EV only, minimum edge %, odds source, game date.

If port `8080` is busy the app picks the next open port. Override with `PORT`:

```bash
PORT=8081 uv run sports-dashboard
```

## Optional: The Odds API

For higher-quality odds, set `THE_ODDS_API_KEY` and the app will prefer it over ESPN:

```bash
export THE_ODDS_API_KEY=your_key_here
uv run sports-dashboard
```

## Repo structure

```
src/
├── data/
│   ├── nba_fetcher.py        live schedule + odds + team form
│   ├── odds_providers.py     ESPN / DraftKings / The Odds API
│   ├── injury_provider.py    ESPN injury feed → per-team points absent
│   ├── kalshi_provider.py    Kalshi NBA-game-market REST client
│   ├── polymarket_provider.py Polymarket gamma API client
│   ├── outcome_lookup.py     leaguegamefinder lookup for paper-trade settlement
│   └── journal.py            SQLite-backed paper-trade journal
├── models/
│   ├── advanced_engine.py    projection engine, applies adjustments
│   ├── trained_nba_model.py  feature engineering, training, CV, calibration
│   └── value_engine.py       implied probability, edge, EV math
├── ui/
│   └── main_ui.py            NiceGUI dashboard
├── pipeline.py               terminal pipeline entry
└── cli.py                    CLI command dispatch
tests/
└── test_engines.py
```

## Caveats

- Third-party data sources (`nba_api`, ESPN, optional The Odds API) — check their ToS before heavy use.
- EV calculation is currently based on the home-side moneyline.
- Damping factor for injuries is calibrated against a historical proxy (top-N rotation by season MPG); the production path uses today's ESPN injury list, which may differ from what `sports-calibrate` measures against.
- Bookmaker coverage depends on what each source publishes for a given game.

## License

MIT
