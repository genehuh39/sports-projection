import runpy
import sys
import unittest
from pathlib import Path

from src.models.trained_nba_model import NBAModelManager
from src.pipeline import run_full_pipeline


def main() -> None:
    """Run the terminal pipeline."""
    run_full_pipeline()


def dashboard() -> None:
    """Run the NiceGUI dashboard."""
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "run_final.py"), run_name="__main__")


def train() -> None:
    """Train and persist the NBA projection model."""
    artifacts = NBAModelManager(auto_train=True).train_and_save(force=True)
    if artifacts is None:
        raise SystemExit("Training did not produce a model artifact.")
    print("Model trained.")
    print(f"Seasons: {', '.join(artifacts.seasons)}")
    print(f"Margin MAE: {artifacts.metrics.get('margin_mae', 0):.2f}")
    print(f"Total MAE: {artifacts.metrics.get('total_mae', 0):.2f}")


def evaluate() -> None:
    """Walk-forward cross-validation of the NBA projection model."""
    n_folds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    test_size = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    results = NBAModelManager().walk_forward_evaluate(n_folds=n_folds, test_size=test_size)

    if "error" in results and not results.get("folds"):
        raise SystemExit(f"Evaluation failed: {results['error']}")

    print(f"Seasons: {', '.join(results['seasons'])}")
    print(f"Features: {results['feature_count']}")
    print(f"Folds: {results['n_folds']}")
    print()
    print(f"{'fold':>4}  {'dates':<25}  {'train':>6}  {'test':>5}  {'m_mae':>6}  {'t_mae':>6}")
    for f in results["folds"]:
        dates = f"{f['test_start_date']}..{f['test_end_date']}"
        print(
            f"{f['fold']:>4}  {dates:<25}  {f['train_rows']:>6}  "
            f"{f['test_rows']:>5}  {f['margin_mae']:>6.2f}  {f['total_mae']:>6.2f}"
        )
    print()
    print(
        f"Margin MAE: {results['margin_mae_mean']:.2f} ± {results['margin_mae_std']:.2f}"
    )
    print(
        f"Total  MAE: {results['total_mae_mean']:.2f} ± {results['total_mae_std']:.2f}"
    )


def calibrate() -> None:
    """Sweep injury damping factors against walk-forward CV."""
    results = NBAModelManager().calibrate_injury_damping()
    if "error" in results and not results.get("grid"):
        raise SystemExit(f"Calibration failed: {results['error']}")

    print(f"Seasons: {', '.join(results['seasons'])}")
    print(f"Folds: {results['n_folds']}")
    print(f"Top-N rotation: {results['top_n_players']} (min fraction {results['min_fraction']})")
    print()
    print(f"{'damping':>8}  {'margin mae':>16}  {'total mae':>16}")
    for row in results["grid"]:
        print(
            f"{row['damping']:>8.2f}  "
            f"{row['margin_mae_mean']:>6.3f} ± {row['margin_mae_std']:>4.2f}   "
            f"{row['total_mae_mean']:>6.3f} ± {row['total_mae_std']:>4.2f}"
        )
    print()
    print(
        f"Best damping for margin: {results['best_margin_damping']:.2f} "
        f"(baseline 0.0 = {results['baseline_margin_mae']:.3f})"
    )
    print(
        f"Best damping for total:  {results['best_total_damping']:.2f} "
        f"(baseline 0.0 = {results['baseline_total_mae']:.3f})"
    )


def backtest() -> None:
    """Walk-forward backtest against a synthetic uniform -110 market.

    Runs both raw and calibrated configurations side-by-side so the
    calibration impact is visible.
    """
    manager = NBAModelManager()
    raw = manager.backtest_evaluate(apply_calibration=False)
    if "error" in raw and not raw.get("folds"):
        raise SystemExit(f"Backtest failed: {raw['error']}")
    calibrated = manager.backtest_evaluate(apply_calibration=True)

    print(f"Seasons: {', '.join(raw['seasons'])}")
    print(
        f"Synthetic market: vig={raw['vig']:.4f}  "
        f"breakeven_prob={raw['breakeven_prob']:.4f}  "
        f"win_payout={raw['win_payout']:+.4f}"
    )
    print(f"Folds: {raw['n_folds']}")
    print()
    print(f"{'metric':<10}  {'raw':>10}  {'calibrated':>12}")
    print(
        f"{'Accuracy':<10}  {raw['accuracy_mean']:>10.3f}  "
        f"{calibrated['accuracy_mean']:>12.3f}"
    )
    print(
        f"{'Brier':<10}  {raw['brier_mean']:>10.4f}  "
        f"{calibrated['brier_mean']:>12.4f}"
    )
    print(
        f"{'Log loss':<10}  {raw['log_loss_mean']:>10.4f}  "
        f"{calibrated['log_loss_mean']:>12.4f}"
    )
    print()
    print("Calibration table (raw vs calibrated)")
    print(
        f"  {'bucket':<14}  {'n_raw':>5}  {'pred_r':>7}  {'act_r':>7}  "
        f"{'n_cal':>5}  {'pred_c':>7}  {'act_c':>7}"
    )
    raw_by_bucket = {(r["bucket_lo"], r["bucket_hi"]): r for r in raw["calibration"]}
    cal_by_bucket = {(r["bucket_lo"], r["bucket_hi"]): r for r in calibrated["calibration"]}
    for key in sorted(set(raw_by_bucket) | set(cal_by_bucket)):
        r = raw_by_bucket.get(key)
        c = cal_by_bucket.get(key)
        bucket = f"{key[0]:.1f}-{key[1]:.1f}"
        r_n = r["n"] if r else 0
        r_pred = f"{r['mean_predicted']:.3f}" if r else "    -  "
        r_act = f"{r['actual_rate']:.3f}" if r else "    -  "
        c_n = c["n"] if c else 0
        c_pred = f"{c['mean_predicted']:.3f}" if c else "    -  "
        c_act = f"{c['actual_rate']:.3f}" if c else "    -  "
        print(
            f"  {bucket:<14}  {r_n:>5}  {r_pred:>7}  {r_act:>7}  "
            f"{c_n:>5}  {c_pred:>7}  {c_act:>7}"
        )
    print()
    print("Synthetic-market ROI by edge threshold (raw vs calibrated)")
    print(
        f"  {'edge>':>6}  {'n_raw':>6}  {'roi_r':>9}  {'wr_r':>6}  "
        f"{'n_cal':>6}  {'roi_c':>9}  {'wr_c':>6}"
    )
    raw_by_t = {row["edge_threshold"]: row for row in raw["thresholds"]}
    cal_by_t = {row["edge_threshold"]: row for row in calibrated["thresholds"]}
    for t in sorted(set(raw_by_t) | set(cal_by_t)):
        r = raw_by_t.get(t)
        c = cal_by_t.get(t)
        r_n = r["n_bets"] if r else 0
        r_roi = f"{r['roi_mean'] * 100.0:+.2f}%" if r and r_n else "    -  "
        r_wr = f"{r['win_rate']:.3f}" if r and r_n else "   -  "
        c_n = c["n_bets"] if c else 0
        c_roi = f"{c['roi_mean'] * 100.0:+.2f}%" if c and c_n else "    -  "
        c_wr = f"{c['win_rate']:.3f}" if c and c_n else "   -  "
        print(
            f"  {t:>6.2f}  {r_n:>6}  {r_roi:>9}  {r_wr:>6}  "
            f"{c_n:>6}  {c_roi:>9}  {c_wr:>6}"
        )
    print()
    print(
        "Note: ROI is against a uniform -110 market — a benchmark, not a "
        "real-world PnL number. Real bookmaker prices will differ."
    )


def kalshi() -> None:
    """Compare model probabilities against live Kalshi NBA game prices.

    Drives predictions directly off Kalshi's event list so we get a row
    for every market Kalshi has open, independent of nba_api scheduling
    quirks.
    """
    from src.data.kalshi_provider import KalshiProvider, compute_kalshi_edges
    from src.data.upcoming import build_upcoming_from_market_games
    from src.models.advanced_engine import AdvancedModelingEngine

    snapshot = KalshiProvider().fetch()
    if not snapshot.games:
        print(
            f"Kalshi returned no open NBA game markets (source={snapshot.source})."
        )
        return

    upcoming = build_upcoming_from_market_games(
        snapshot.games, id_attr="event_ticker"
    )
    if upcoming.is_empty():
        print("Kalshi events returned but no NBA team-abbreviation matches.")
        return

    engine = AdvancedModelingEngine(
        auto_train=True, use_trained_model=True, apply_injury_adjustment=False
    )
    projections = engine.generate_projections(upcoming)
    annotated = compute_kalshi_edges(snapshot, projections)

    rows = annotated.to_dicts()
    print(
        f"{'matchup':<14}  {'date':<10}  {'p_home':>6}  {'kal_h':>6}  "
        f"{'edge_h':>7}  {'kal_a':>6}  {'edge_a':>7}  {'event_ticker'}"
    )
    matched = 0
    quoted = 0
    for r in rows:
        matchup = (
            f"{(r.get('away_team_code') or '')[:3]:<3} @ "
            f"{(r.get('home_team_code') or '')[:3]:<3}"
        )
        gd = str(r.get("game_date", ""))[:10]
        ph = r.get("home_win_prob")
        kh = r.get("kalshi_home_yes_ask")
        ka = r.get("kalshi_away_yes_ask")
        eh = r.get("edge_home")
        ea = r.get("edge_away")
        ev = r.get("kalshi_event_ticker") or ""
        if ev:
            matched += 1
        if kh is not None or ka is not None:
            quoted += 1
        print(
            f"{matchup:<14}  {gd:<10}  "
            f"{(f'{ph:.3f}' if ph is not None else '   -  '):>6}  "
            f"{(f'{kh:.2f}' if kh is not None else '  -  '):>6}  "
            f"{(f'{eh:+.3f}' if eh is not None else '   -  '):>7}  "
            f"{(f'{ka:.2f}' if ka is not None else '  -  '):>6}  "
            f"{(f'{ea:+.3f}' if ea is not None else '   -  '):>7}  "
            f"{ev}"
        )
    print()
    print(
        f"Matched {matched} / {len(rows)} games to Kalshi events; "
        f"{quoted} have at least one resting quote."
    )
    print(
        "edge_h = model_home_prob - kalshi_home_yes_ask  "
        "(positive => buy YES home is +EV at this ask)"
    )


def polymarket() -> None:
    """Compare model probabilities against live Polymarket NBA prices."""
    from src.data.polymarket_provider import (
        PolymarketProvider,
        compute_polymarket_edges,
    )
    from src.data.upcoming import build_upcoming_from_market_games
    from src.models.advanced_engine import AdvancedModelingEngine

    snapshot = PolymarketProvider().fetch()
    if not snapshot.games:
        print(
            f"Polymarket returned no NBA game markets (source={snapshot.source})."
        )
        return

    upcoming = build_upcoming_from_market_games(snapshot.games, id_attr="slug")
    if upcoming.is_empty():
        print("Polymarket markets returned but no NBA team-abbreviation matches.")
        return

    engine = AdvancedModelingEngine(
        auto_train=True, use_trained_model=True, apply_injury_adjustment=False
    )
    projections = engine.generate_projections(upcoming)
    annotated = compute_polymarket_edges(snapshot, projections)

    rows = annotated.to_dicts()
    print(
        f"{'matchup':<14}  {'date':<10}  {'p_home':>6}  {'pm_h':>6}  "
        f"{'edge_h':>7}  {'pm_a':>6}  {'edge_a':>7}  {'slug'}"
    )
    matched = 0
    for r in rows:
        matchup = (
            f"{(r.get('away_team_code') or '')[:3]:<3} @ "
            f"{(r.get('home_team_code') or '')[:3]:<3}"
        )
        gd = str(r.get("game_date", ""))[:10]
        ph = r.get("home_win_prob")
        ph_pm = r.get("polymarket_home_price")
        pa_pm = r.get("polymarket_away_price")
        eh = r.get("edge_home_pm")
        ea = r.get("edge_away_pm")
        slug = r.get("polymarket_slug") or ""
        if slug:
            matched += 1
        print(
            f"{matchup:<14}  {gd:<10}  "
            f"{(f'{ph:.3f}' if ph is not None else '   -  '):>6}  "
            f"{(f'{ph_pm:.3f}' if ph_pm is not None else '  -  '):>6}  "
            f"{(f'{eh:+.3f}' if eh is not None else '   -  '):>7}  "
            f"{(f'{pa_pm:.3f}' if pa_pm is not None else '  -  '):>6}  "
            f"{(f'{ea:+.3f}' if ea is not None else '   -  '):>7}  "
            f"{slug}"
        )
    print()
    print(f"Matched {matched} / {len(rows)} games to Polymarket markets.")
    print(
        "edge_h = model_home_prob - polymarket_home_price  "
        "(positive => model thinks home more likely than market)"
    )
    print(
        "Note: prices are last-trade from Polymarket gamma API, not bid/ask. "
        "Real fills happen at CLOB book."
    )


def paper_trade() -> None:
    """Log paper trades against current Polymarket prices.

    For each market with edge > threshold (default 0.05) on either side,
    inserts a journal entry. Idempotent per (venue, game_id, side).
    """
    from src.data.journal import PaperTrade, PaperTradeJournal
    from src.data.polymarket_provider import (
        PolymarketProvider,
        compute_polymarket_edges,
    )
    from src.data.upcoming import build_upcoming_from_market_games
    from src.models.advanced_engine import AdvancedModelingEngine

    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    snapshot = PolymarketProvider().fetch()
    if not snapshot.games:
        print("Polymarket returned no NBA markets.")
        return

    upcoming = build_upcoming_from_market_games(
        snapshot.games, id_attr="slug", future_only=True
    )
    if upcoming.is_empty():
        print("No future-dated Polymarket games to evaluate.")
        return

    # Reuse the saved artifact if available; only retrain when missing.
    # Nightly cron should not pay the full retrain cost on every run.
    engine = AdvancedModelingEngine(
        auto_train=False, use_trained_model=True, apply_injury_adjustment=False
    )
    if engine.model_manager.ensure_artifacts() is None:
        engine.model_manager.auto_train = True
    projections = engine.generate_projections(upcoming)
    annotated = compute_polymarket_edges(snapshot, projections)

    journal = PaperTradeJournal()
    inserted = 0
    skipped = 0
    for r in annotated.to_dicts():
        slug = r.get("polymarket_slug")
        gd_str = str(r.get("game_date", ""))[:10]
        home = r.get("home_team_code", "")
        away = r.get("away_team_code", "")
        model_p = r.get("home_win_prob")
        ph = r.get("polymarket_home_price")
        pa = r.get("polymarket_away_price")
        if slug is None or model_p is None or ph is None or pa is None:
            continue

        candidates = [
            ("home", float(model_p), float(ph), float(model_p) - float(ph)),
            ("away", 1.0 - float(model_p), float(pa), (1.0 - float(model_p)) - float(pa)),
        ]
        for side, prob, price, edge in candidates:
            if edge < threshold:
                continue
            trade = PaperTrade(
                venue="polymarket",
                game_id=slug,
                game_date=gd_str,
                home_team_abbr=home,
                away_team_abbr=away,
                side=side,
                model_prob=prob,
                market_price=price,
                edge=edge,
            )
            if journal.append(trade):
                inserted += 1
                print(
                    f"  [{slug}] bet {side} @ {price:.3f} (model {prob:.3f}, edge +{edge:.3f})"
                )
            else:
                skipped += 1

    print()
    print(f"Inserted {inserted} new trades; skipped {skipped} duplicates.")


def settle() -> None:
    """Look up outcomes for past-dated open trades and write realized PnL."""
    from datetime import date as date_cls

    from src.data.journal import PaperTradeJournal, yes_pnl
    from src.data.outcome_lookup import fetch_outcomes_for_dates

    journal = PaperTradeJournal()
    today_iso = date_cls.today().isoformat()
    open_trades = journal.list_open_past_games(today_iso)
    if not open_trades:
        print("No open trades on past-dated games.")
        return

    target_dates = sorted({date_cls.fromisoformat(t.game_date) for t in open_trades})
    print(f"Looking up {len(target_dates)} game date(s) for {len(open_trades)} open trade(s)...")
    outcomes = fetch_outcomes_for_dates(target_dates)
    print(f"Fetched {len(outcomes)} game outcomes.")

    settled = 0
    for t in open_trades:
        key = (t.home_team_abbr, t.away_team_abbr, t.game_date)
        outcome = outcomes.get(key)
        if outcome is None:
            print(f"  unresolved: {t.game_id} ({t.away_team_abbr} @ {t.home_team_abbr} on {t.game_date})")
            continue
        side_won = outcome.home_won if t.side == "home" else not outcome.home_won
        pnl = yes_pnl(t.market_price, side_won)
        journal.mark_settled(
            trade_id=t.id, home_won=outcome.home_won, side_won=side_won, realized_pnl=pnl
        )
        settled += 1
        result = "WIN" if side_won else "loss"
        print(
            f"  {result}: {t.game_id} bet {t.side} @ {t.market_price:.3f} -> pnl {pnl:+.3f}"
        )
    print()
    print(f"Settled {settled} trades.")


def pnl() -> None:
    """Cumulative ROI and edge-bucket breakdown over the settled journal."""
    from src.data.journal import PaperTradeJournal

    settled = PaperTradeJournal().list_settled()
    if not settled:
        print("No settled trades yet. Run sports-paper-trade then sports-settle.")
        return

    total_stake = sum(t.stake for t in settled)
    total_pnl = sum((t.realized_pnl or 0) * t.stake for t in settled)
    wins = sum(1 for t in settled if (t.realized_pnl or 0) > 0)
    roi = total_pnl / total_stake if total_stake else 0.0

    print(f"Settled trades: {len(settled)}")
    print(f"Total stake:    ${total_stake:.2f}")
    print(f"Total PnL:      ${total_pnl:+.2f}")
    print(f"ROI:            {roi*100:+.2f}%")
    print(f"Win rate:       {wins/len(settled):.3f}")
    print()

    print("By edge bucket:")
    print(f"  {'edge>=':>6}  {'edge<':>6}  {'n':>4}  {'roi':>8}  {'win_rate':>9}")
    buckets = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
    for lo, hi in buckets:
        sub = [t for t in settled if lo <= t.edge < hi]
        if not sub:
            continue
        s_stake = sum(t.stake for t in sub)
        s_pnl = sum((t.realized_pnl or 0) * t.stake for t in sub)
        s_wr = sum(1 for t in sub if (t.realized_pnl or 0) > 0) / len(sub)
        print(
            f"  {lo:>6.2f}  {hi:>6.2f}  {len(sub):>4}  "
            f"{(s_pnl/s_stake)*100:>+7.2f}%  {s_wr:>9.3f}"
        )
    print()
    print("By venue:")
    venues = sorted({t.venue for t in settled})
    for v in venues:
        sub = [t for t in settled if t.venue == v]
        s_stake = sum(t.stake for t in sub)
        s_pnl = sum((t.realized_pnl or 0) * t.stake for t in sub)
        print(
            f"  {v:<12}  {len(sub):>4} bets, ${s_pnl:+.2f} pnl on ${s_stake:.2f} stake "
            f"({(s_pnl/s_stake)*100:+.2f}% ROI)"
        )


def test() -> None:
    """Run the unittest suite."""
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "main"
    if command == "main":
        main()
    elif command == "dashboard":
        dashboard()
    elif command == "train":
        train()
    elif command == "evaluate":
        evaluate()
    elif command == "calibrate":
        calibrate()
    elif command == "backtest":
        backtest()
    elif command == "kalshi":
        kalshi()
    elif command == "polymarket":
        polymarket()
    elif command == "paper-trade":
        paper_trade()
    elif command == "settle":
        settle()
    elif command == "pnl":
        pnl()
    elif command == "test":
        test()
    else:
        raise SystemExit(f"Unknown command: {command}")
