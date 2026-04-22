# sports-projection

NBA projections and betting value dashboard built with `uv`, `polars`, `nba_api`, and NiceGUI.

## What it does

- pulls upcoming NBA games from `nba_api`
- pulls recent team form from real NBA results
- merges live odds from multiple sources
  - ESPN scoreboard / DraftKings when available
  - optional The Odds API via `THE_ODDS_API_KEY`
- calculates:
  - model win probability
  - implied probability
  - edge
  - expected value (EV)
- supports a trained XGBoost projection model using:
  - recent form
  - season-long form
  - rest / back-to-back features
  - pace and margin features
- shows everything in a NiceGUI dashboard with filters
- falls back to mock data when live data is unavailable

## Quick start

On macOS, install the OpenMP runtime that XGBoost requires:

```bash
brew install libomp
```

Install dependencies:

```bash
uv sync
```

Run the terminal pipeline:

```bash
uv run sports-projection
```

Run the dashboard:

```bash
uv run sports-dashboard
```

Train the model artifact:

```bash
uv run sports-train
```

Run tests:

```bash
uv run sports-test
```

## Model training

The trained model uses historical NBA games to build pregame features such as:

- recent offense / defense
- recent margin
- recent pace
- season offense / defense
- season margin
- season pace
- rest days
- back-to-back flags
- team form differentials

Training persists a model artifact to `artifacts/nba_projection_model.joblib`.
The live pipeline and dashboard can use this trained model when auto-training is enabled.

## Dashboard features

The dashboard currently shows:

- matchup
- game date
- bookmaker
- projected home / away score
- model win probability
- implied probability
- edge
- home moneyline
- spread
- total
- EV

Available filters:

- positive EV only
- minimum edge %
- odds source
- game date

If port `8080` is busy, the app automatically picks the next open port.
You can still prefer a port manually:

```bash
PORT=8081 uv run sports-dashboard
```

## Optional odds provider

To enable The Odds API as a higher-priority odds source:

```bash
export THE_ODDS_API_KEY=your_key_here
```

Then run the app normally.
When available, The Odds API is preferred over ESPN during odds merging.

## Project scripts

Defined in `pyproject.toml`:

```bash
uv run sports-projection
uv run sports-dashboard
uv run sports-train
uv run sports-test
```

## Repo structure

- `src/data/nba_fetcher.py` – live NBA schedule + merged odds + team form
- `src/data/odds_providers.py` – odds providers and normalization
- `src/models/advanced_engine.py` – projection engine / trained-model fallback logic
- `src/models/trained_nba_model.py` – historical feature engineering and XGBoost training
- `src/models/value_engine.py` – implied probability, edge, EV math
- `src/ui/main_ui.py` – NiceGUI dashboard
- `src/pipeline.py` – terminal pipeline entry logic
- `tests/test_engines.py` – test suite

## Notes

- current EV is based on the home-side moneyline
- the model is still lightweight and based mainly on recent scoring context
- bookmaker coverage depends on source availability for a given game

## License

MIT
