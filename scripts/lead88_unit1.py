from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

SCHEDULE = Path('data/raw/20260908T162105Z/schedules.parquet')
BASE = Path('artifacts/pick_probability/20260929T192747Z/per_game.parquet')
MARGINS = Path('artifacts/margins/20260929T192312Z/predictions.parquet')
OUTPUT = Path('tests/scratch/codex/lead88_unit1')
ZONE = 'America/New_York'
KEYS = ['game_id', 'bookmaker_key']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_frame(path, columns):
    return pq.read_table(path, columns=columns, use_threads=False).to_pandas(use_threads=False)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def load_games():
    games = read_frame(SCHEDULE, ['game_id', 'season', 'week', 'game_type', 'gameday', 'gametime', 'home_team', 'away_team'])
    games = games.loc[games.season.between(2020, 2025) & games.game_type.eq('REG')].copy()
    if games.game_id.duplicated().any():
        raise ValueError('Duplicate schedule game identifiers')
    local = pd.to_datetime(games.gameday.astype(str) + ' ' + games.gametime.astype(str), errors='raise')
    games['kickoff'] = local.dt.tz_localize(ZONE, ambiguous='raise', nonexistent='raise').dt.tz_convert('UTC')
    first = local.dt.normalize().groupby([games.season, games.week]).transform('min')
    sunday = first + pd.to_timedelta((6 - first.dt.dayofweek) % 7, unit='D')
    for name, offset in {'tuesday_start': pd.Timedelta(days=-5), 'freeze': pd.Timedelta(days=-5, hours=12), 'sunday_start': pd.Timedelta(0), 'deadline': pd.Timedelta(hours=12, minutes=30)}.items():
        games[name] = (sunday + offset).dt.tz_localize(ZONE).dt.tz_convert('UTC')
    last_ids = games.sort_values(['kickoff', 'game_id'], kind='stable').groupby(['season', 'week']).tail(1).game_id
    games['is_last_game'] = games.game_id.isin(last_ids)
    return games.reset_index(drop=True)


def load_pairs(games):
    frames, sources = [], []
    columns = ['nflverse_game_id', 'bookmaker_key', 'market', 'outcome_side', 'line', 'observed_at_utc', 'bookmaker_last_update_utc', 'market_last_update_utc', 'commence_time_utc', 'home_team', 'away_team', 'provider', 'sport_key', 'capture_kind']
    for path in sorted(Path('data/market/raw').glob('*/manifest.json')):
        manifest = json.loads(path.read_text(encoding='utf-8-sig'))
        request = manifest.get('request', {})
        label = request.get('decision_label')
        if not (manifest.get('provider') == 'the-odds-api' and manifest.get('capture_kind') == 'historical_backfill' and request.get('season') in range(2020, 2026) and request.get('sport') == 'americanfootball_nfl' and label in {'tue_open', 'sun_early_close'} and 'totals' in str(request.get('markets', '')).split(',')):
            continue
        quote_path = path.with_name('quotes.parquet')
        expected = manifest.get('files', {}).get('quotes.parquet', {}).get('sha256')
        if not quote_path.is_file() or not expected or digest(quote_path) != expected:
            raise ValueError(f'Missing or corrupt declared source: {quote_path}')
        frame = read_frame(quote_path, columns)
        sources.append({'path': quote_path.as_posix(), 'rows': len(frame), 'sha256': expected, 'manifest_sha256': digest(path), 'season': request['season'], 'week': request['week'], 'label': label})
        frame = frame.loc[frame.market.eq('totals')].copy()
        frame['game_id'] = frame.pop('nflverse_game_id')
        frame['manifest_season'], frame['manifest_week'], frame['label'] = request['season'], request['week'], label
        frame['snapshot'] = pd.to_datetime(manifest.get('snapshot_timestamp_utc'), utc=True, errors='coerce')
        frame['requested'] = pd.to_datetime(manifest.get('requested_at_utc'), utc=True, errors='coerce')
        frame['source'] = quote_path.as_posix()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), {'sources': sources, 'rejected': {}, 'source_absent': True}
    quotes = pd.concat(frames, ignore_index=True).merge(games, on='game_id', how='left', suffixes=('', '_schedule'), validate='many_to_one')
    for name in ['observed_at_utc', 'bookmaker_last_update_utc', 'market_last_update_utc', 'commence_time_utc']:
        quotes[name] = pd.to_datetime(quotes[name], utc=True, errors='coerce')
    quotes['line'] = pd.to_numeric(quotes.line, errors='coerce')
    is_tuesday = quotes.label.eq('tue_open')
    upper = quotes.freeze.where(is_tuesday, quotes.deadline)
    lower = quotes.tuesday_start.where(is_tuesday, quotes.sunday_start)
    checks = {
        'target_season_week': quotes.manifest_season.eq(quotes.season) & quotes.manifest_week.eq(quotes.week),
        'team_identity': quotes.home_team.eq(quotes.home_team_schedule) & quotes.away_team.eq(quotes.away_team_schedule),
        'source_identity': quotes.provider.eq('the-odds-api') & quotes.sport_key.eq('americanfootball_nfl') & quotes.capture_kind.eq('historical_backfill'),
        'snapshot_window': quotes.snapshot.ge(lower) & quotes.snapshot.le(upper),
        'requested_window': quotes.requested.ge(lower) & quotes.requested.le(upper) & quotes.snapshot.le(quotes.requested),
        'sunday_request': is_tuesday | quotes.requested.eq(quotes.deadline),
        'observed_clock': quotes.observed_at_utc.eq(quotes.snapshot),
        'book_clock': quotes.bookmaker_last_update_utc.notna() & quotes.bookmaker_last_update_utc.le(quotes.observed_at_utc),
        'market_clock': quotes.market_last_update_utc.notna() & quotes.market_last_update_utc.le(quotes.observed_at_utc),
        'pregame': quotes.requested.lt(quotes.kickoff) & quotes.requested.lt(quotes.commence_time_utc),
        'positive_total': np.isfinite(quotes.line) & quotes.line.gt(0),
        'book_and_outcome': quotes.bookmaker_key.notna() & quotes.outcome_side.isin(['OVER', 'UNDER']),
    }
    valid = pd.Series(True, index=quotes.index)
    for check in checks.values():
        valid &= check
    evidence = quotes.loc[valid]
    panels, invalid_panels = [], 0
    for (game_id, book, label, source), group in evidence.groupby([*KEYS, 'label', 'source'], sort=True):
        if len(group) != 2 or set(group.outcome_side) != {'OVER', 'UNDER'} or group.line.nunique() != 1:
            invalid_panels += 1
            continue
        row = group.iloc[0]
        panels.append({'game_id': game_id, 'bookmaker_key': book, 'label': label, 'source': source, 'total': float(row.line), 'observed': row.observed_at_utc, 'snapshot': row.snapshot, 'requested': row.requested, 'book_updated': group.bookmaker_last_update_utc.max(), 'market_updated': group.market_last_update_utc.max(), 'commence': group.commence_time_utc.min()})
    panel = pd.DataFrame(panels)
    inventory = {'sources': sources, 'rejected': {key: int((~value).sum()) for key, value in checks.items()}, 'source_absent': False, 'input_total_rows': len(quotes), 'admissible_total_rows': len(evidence), 'invalid_over_under_panels': invalid_panels}
    if panel.empty:
        return panel, inventory
    if panel.groupby([*KEYS, 'label', 'snapshot']).total.nunique().gt(1).any():
        raise ValueError('Conflicting totals at the same book and snapshot')
    panel = panel.sort_values(['snapshot', 'source']).drop_duplicates([*KEYS, 'label'], keep='last')
    parts = []
    for label, prefix in [('tue_open', 'tuesday'), ('sun_early_close', 'deadline')]:
        part = panel.loc[panel.label.eq(label)].drop(columns='label')
        parts.append(part.rename(columns={name: prefix + '_' + name for name in part.columns if name not in KEYS}))
    pairs = parts[0].merge(parts[1], on=KEYS, validate='one_to_one')
    pairs['total_move'] = pairs.deadline_total - pairs.tuesday_total
    inventory.update(valid_panels=len(panel), tuesday_book_games=len(parts[0]), deadline_book_games=len(parts[1]))
    return pairs, inventory


def audit_base(joined):
    audit = {}
    if BASE.is_file():
        base = read_frame(BASE, ['game_id', 'chronological_home_probability'])
        if base.game_id.duplicated().any():
            raise ValueError('Duplicate frozen probability rows')
        base['has_four_term_row'] = True
        base['has_chronological_probability'] = np.isfinite(base.pop('chronological_home_probability'))
        joined = joined.merge(base, on='game_id', how='left', validate='one_to_one')
        audit['four_term_sha256'] = digest(BASE)
    for name in ['has_four_term_row', 'has_chronological_probability']:
        if name not in joined:
            joined[name] = False
        joined[name] = joined[name].eq(True)
    joined['upstream_before_freeze'] = False
    if MARGINS.is_file():
        archive = read_frame(MARGINS, ['game_id', 'method', 'model_name', 'train_max_gameday'])
        archive = archive.loc[archive.method.eq('market_residual') & archive.model_name.eq('ridge')]
        if archive.game_id.duplicated().any():
            raise ValueError('Duplicate upstream training cutoff rows')
        joined = joined.merge(archive[['game_id', 'train_max_gameday']], on='game_id', how='left', validate='one_to_one')
        trained = pd.to_datetime(joined.train_max_gameday, utc=True, errors='coerce')
        joined['upstream_before_freeze'] = trained.notna() & (trained + pd.Timedelta(days=1)).le(joined.freeze)
        audit['margin_archive_sha256'] = digest(MARGINS)
    return joined, audit


def run():
    declaration = Path('docs/lanes/lead88.md').read_text(encoding='utf-8-sig')
    if '713 looks' not in declaration or 'frozen before outcomes' not in declaration:
        raise ValueError('The required pre-outcome declaration is missing')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    games = load_games()
    pairs, inventory = load_pairs(games)
    joined, upstream = games.iloc[:0].copy(), {}
    if not pairs.empty:
        panel = pairs.groupby('game_id', as_index=False).agg(tuesday_total=('tuesday_total', 'median'), deadline_total=('deadline_total', 'median'), same_book_count=('bookmaker_key', 'nunique'), tuesday_first_observed=('tuesday_observed', 'min'), tuesday_last_observed=('tuesday_observed', 'max'), deadline_first_observed=('deadline_observed', 'min'), deadline_last_observed=('deadline_observed', 'max'))
        panel['total_move'] = panel.deadline_total - panel.tuesday_total
        joined, upstream = audit_base(games.merge(panel, on='game_id', validate='one_to_one'))
        if not (joined.tuesday_last_observed.le(joined.freeze).all() and joined.deadline_last_observed.le(joined.deadline).all() and joined.deadline.lt(joined.kickoff).all()):
            raise ValueError('Paired total feature violates the pregame clock')
        pairs.to_parquet(OUTPUT / 'book_pairs.parquet', index=False)
        joined.to_parquet(OUTPUT / 'paired_games.parquet', index=False)
    season_rows, folds = [], []
    for year in range(2020, 2026):
        selected = joined.loc[joined.season.eq(year)]
        season_rows.append([year, int(games.season.eq(year).sum()), len(selected), int(selected.is_last_game.sum()), *[int(selected.get(name, pd.Series(dtype=bool)).sum()) for name in ['has_four_term_row', 'has_chronological_probability', 'upstream_before_freeze']]])
    for year in (2023, 2024, 2025):
        test = joined.loc[joined.season.eq(year)]
        folds.append([year, int(joined.season.le(year - 3).sum()), int(joined.season.eq(year - 2).sum()), int(joined.season.eq(year - 1).sum()), len(test), int(test.is_last_game.sum())])
    totals = {'schedule_games': len(games), 'schedule_last_games': int(games.is_last_game.sum()), 'source_files': len(inventory['sources']), 'source_rows': sum(source['rows'] for source in inventory['sources']), 'same_book_pairs': len(pairs), 'paired_games': len(joined), 'paired_last_games': int(joined.is_last_game.sum()), 'planned_statistical_looks': 713, 'executed_statistical_looks': 0}
    payload = {'unit': 'paired_total_join', 'outcomes_read': False, 'totals': totals, 'inventory': inventory, 'upstream': upstream, 'seasons': season_rows, 'folds': folds, 'schedule_sha256': digest(SCHEDULE), 'script_sha256': digest(Path(__file__)), 'protocol_sha256': digest(Path('docs/lead88_protocol.md'))}
    (OUTPUT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# LEAD-88 unit 1: paired-total join', '',
        '**Measured:** outcome-free source/clock inventory and paired-total join; no fit, score, registry write or serving change.',
        'The roadmap separates this paired-total unit from its one-parameter replay. Declaration was saved first in',
        '`docs/lanes/lead88.md`; verbatim row and shared Protocol C are in `docs/lead88_protocol.md`.', '',
        '## Source coverage', '',
        f"**Measured:** {totals['source_files']} verified source files / {totals['source_rows']:,} quote rows;",
        f"{totals['same_book_pairs']:,} same-book total pairs give {totals['paired_games']:,} games and",
        f"{totals['paired_last_games']} actual weekly last games of {totals['schedule_last_games']} scheduled last games.",
        'The historical sources are present. Missing games are not replaced with zero movement or closing lines.', '',
        table(['Season', 'Scheduled REG', 'Paired games', 'Last games', 'Four-term rows', 'Chronological p present', 'Upstream day before freeze'], season_rows), '',
        '**Measured:** rejected-row clock/identity counts overlap; they are not additive:', '',
        table(['Check', 'Rows failing'], [[key, value] for key, value in inventory.get('rejected', {}).items()]), '',
        '## Join rules', '',
        '**Read/inferred implementation:** select historical target-season/week Tuesday and Sunday-12:30 manifests,',
        'verify quote hashes, require exactly one matching OVER/UNDER pair at the same total per book/capture,',
        'then retain the latest admissible capture per game/book/endpoint. Require source and schedule team identity,',
        'observation = snapshot, bookmaker/market updates no later than observation, and requests before both',
        'source and schedule kickoff. Tuesday anchors precede noon; they are not evidence of a noon quote.',
        'Sunday requests must equal the declared 12:30 boundary. Only books present at both endpoints enter the',
        'game-level medians. Preserve each original bookmaker pair and clock in the scratch artifact.',
        'The last-game flag is computed from the complete REG schedule before quote filtering. No score, margin,',
        'cover outcome, or closing total column is loaded. Probability availability is checked without scoring.', '',
        '## Replay fold inventory', '',
        table(['Outer year', 'Fit through Y-3', 'Tune Y-2', 'Calibrate Y-1', 'Outer games', 'Outer last games'], folds), '',
        '**Measured:** no in-sample/OOS result, gap, coefficient, decisive record, interval or probability_positive',
        'has been computed. These are not zero or 0.5. Planned family: 713 looks; executed statistical looks: 0.',
        'The source counts are census counts, not sampled outcome estimates; an outcome interval is inapplicable.', '',
        '## Open replay requirements', '',
        '**Read:** `scripts/tiebreaker_total_study.py:33` rounds half-up, whereas the current served path in',
        '`src/nfl_ats/tiebreaker.py:427` uses `pick_consistent_top_score` with a fixed side and margin centre.',
        'Its no-model fallback at line 466 rounds a neighborhood median. The declaration says existing rounding;',
        'the replay must resolve the intended entry point before fitting, rather than silently substitute a rule.',
        '**Read:** the frozen four-term source excludes opener pushes and lacks chronological probabilities in',
        'its earliest seasons. The coverage table identifies missing rows; keep pushes in the total population.',
        '**Inferred:** a training calendar day ending before freeze is a date-granularity check, not proof of',
        'completion-time availability or Protocol C fit/tune/calibration separation. Reconstruct matched four-term features,',
        'verify fold lineage, retain a fixed margin probability/side/Best Pick, calibrate the discrete total law on',
        'the separate year, and report all declared comparisons. This unit makes no signal or pool-rank verdict.', '',
        '## Saved artifacts', '',
        '`tests/scratch/codex/lead88_unit1/{book_pairs,paired_games}.parquet` contains quote features and clocks only.',
        '`summary.json` contains source hashes, coverage, rejected checks and fold counts. No outcome rows enter docs.',
        'Record commands: none, because no effect has been estimated or adjudicated.', '',
    ]
    Path('docs/lead88_unit1.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(totals, sort_keys=True))
    print('report=docs/lead88_unit1.md artifacts=' + OUTPUT.as_posix() + ' outcomes_read=false')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        run()
