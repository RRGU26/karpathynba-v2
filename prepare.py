"""
V2: NBA spread prediction data preparation.
Adds player game logs, streak features, and more seasons vs v1.

Usage:
    python prepare.py               # download and prepare all data (20 seasons)
    python prepare.py --seasons 20  # explicit season count

Data is stored in ~/.cache/autoresearch-nba-v2/.

THIS FILE IS READ-ONLY. The agent modifies only train.py.
"""

import argparse
import math
import os
import pickle
import time

import mlx.core as mx
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playergamelog
from nba_api.stats.static import teams as nba_teams

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300  # 5 minutes wall-clock training time
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-nba-v2")
DATA_PATH = os.path.join(CACHE_DIR, "features.pkl")
RAW_PATH = os.path.join(CACHE_DIR, "raw_games.pkl")
PLAYER_RAW_PATH = os.path.join(CACHE_DIR, "player_games.pkl")
VAL_FRACTION = 0.15  # last 15% of data is validation
TEST_FRACTION = 0.05  # last 5% of data is test (held out)

# Rolling windows for feature computation
ROLLING_WINDOWS = [5, 10, 20]

# Minimum games before a team's games are included (cold start)
MIN_GAMES_WARMUP = 20

# Elo parameters
ELO_K = 20
ELO_HOME_ADV = 100
ELO_INIT = 1500
ELO_SEASON_REGRESS = 0.75  # regress 25% toward mean between seasons

# Season range
CURRENT_SEASON_YEAR = 2025  # 2024-25 season
START_SEASON_YEAR = 2006    # 2005-06 season (20 seasons)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------


def _season_string(year):
    """Convert year to NBA season string, e.g. 2024 -> '2024-25'."""
    return f"{year-1}-{str(year)[-2:]}"


def download_raw_games(n_seasons=10):
    """Download team game logs from nba_api for the last N seasons."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("n_seasons") == n_seasons:
            days_old = (pd.Timestamp.now() - pd.Timestamp(cached["downloaded_at"])).days
            if days_old < 2:
                print(f"Raw data: using cache ({len(cached['games'])} team-game rows, {days_old}d old)")
                return cached["games"]
            print(f"Raw data: cache is {days_old} days old, refreshing...")

    end_year = CURRENT_SEASON_YEAR
    start_year = end_year - n_seasons + 1

    all_games = []
    for year in range(start_year, end_year + 1):
        season = _season_string(year)
        print(f"  Downloading {season}...", end=" ", flush=True)
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="T",
                season_type_all_star="Regular Season",
            )
            df = log.get_data_frames()[0]
            df["SEASON_YEAR"] = year
            all_games.append(df)
            print(f"{len(df)} rows")
        except Exception as e:
            print(f"FAILED ({e})")
        time.sleep(0.7)  # rate limit

    games = pd.concat(all_games, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    games = games.sort_values("GAME_DATE").reset_index(drop=True)

    # Cache
    with open(RAW_PATH, "wb") as f:
        pickle.dump({
            "games": games,
            "n_seasons": n_seasons,
            "downloaded_at": pd.Timestamp.now().isoformat(),
        }, f)

    print(f"Raw data: {len(games)} total team-game rows across {n_seasons} seasons")
    return games


def download_player_games(n_seasons=20):
    """Download player-level game logs (league-wide, one call per season)."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(PLAYER_RAW_PATH):
        with open(PLAYER_RAW_PATH, "rb") as f:
            cached = pickle.load(f)
        if cached.get("n_seasons") == n_seasons:
            days_old = (pd.Timestamp.now() - pd.Timestamp(cached["downloaded_at"])).days
            if days_old < 2:
                print(f"Player data: using cache ({len(cached['games'])} rows, {days_old}d old)")
                return cached["games"]
            print(f"Player data: cache is {days_old} days old, refreshing...")

    end_year = CURRENT_SEASON_YEAR
    start_year = end_year - n_seasons + 1

    all_player_games = []
    for year in range(start_year, end_year + 1):
        season = _season_string(year)
        print(f"  Player logs {season}...", end=" ", flush=True)
        try:
            log = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
            )
            df = log.get_data_frames()[0]
            df["SEASON_YEAR"] = year
            all_player_games.append(df)
            print(f"{len(df)} rows")
        except Exception as e:
            print(f"FAILED ({e})")
        time.sleep(0.7)

    player_games = pd.concat(all_player_games, ignore_index=True)
    player_games["GAME_DATE"] = pd.to_datetime(player_games["GAME_DATE"])
    player_games = player_games.sort_values("GAME_DATE").reset_index(drop=True)

    with open(PLAYER_RAW_PATH, "wb") as f:
        pickle.dump({
            "games": player_games,
            "n_seasons": n_seasons,
            "downloaded_at": pd.Timestamp.now().isoformat(),
        }, f)

    print(f"Player data: {len(player_games)} total player-game rows")
    return player_games


# ---------------------------------------------------------------------------
# Game pairing: join home + away into single rows
# ---------------------------------------------------------------------------


def build_game_pairs(games):
    """
    Each GAME_ID has 2 rows (one per team). Parse MATCHUP to identify
    home ("vs.") and away ("@") teams. Join into single game rows.
    """
    # Identify home/away from MATCHUP column
    games = games.copy()
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.", regex=False)

    home = games[games["IS_HOME"]].copy()
    away = games[~games["IS_HOME"]].copy()

    # Rename columns with prefixes
    home_cols = {c: f"HOME_{c}" for c in home.columns if c not in ("GAME_ID", "GAME_DATE", "SEASON_YEAR")}
    away_cols = {c: f"AWAY_{c}" for c in away.columns if c not in ("GAME_ID", "GAME_DATE", "SEASON_YEAR")}

    home = home.rename(columns=home_cols)
    away = away.rename(columns=away_cols)

    # Merge on GAME_ID
    paired = home.merge(away, on=["GAME_ID", "GAME_DATE", "SEASON_YEAR"], how="inner")
    paired = paired.sort_values("GAME_DATE").reset_index(drop=True)

    # Targets
    paired["target_point_diff"] = paired["HOME_PTS"] - paired["AWAY_PTS"]
    paired["target_home_win"] = (paired["target_point_diff"] > 0).astype(float)

    print(f"Game pairs: {len(paired)} games")
    return paired


# ---------------------------------------------------------------------------
# Rolling feature computation
# ---------------------------------------------------------------------------


def _compute_team_rolling_stats(team_games, windows):
    """
    Compute rolling stats for a single team's game history.
    team_games should be sorted by date with columns from the raw game log.
    Returns a DataFrame with rolling features, indexed same as input.
    """
    feats = pd.DataFrame(index=team_games.index)

    pts = team_games["PTS"].astype(float)
    fgm = team_games["FGM"].astype(float)
    fga = team_games["FGA"].astype(float)
    fg3m = team_games["FG3M"].astype(float)
    fg3a = team_games["FG3A"].astype(float)
    ftm = team_games["FTM"].astype(float)
    fta = team_games["FTA"].astype(float)
    oreb = team_games["OREB"].astype(float)
    dreb = team_games["DREB"].astype(float)
    ast = team_games["AST"].astype(float)
    tov = team_games["TOV"].astype(float)
    stl = team_games["STL"].astype(float)
    blk = team_games["BLK"].astype(float)
    pf = team_games["PF"].astype(float)
    plus_minus = team_games["PLUS_MINUS"].astype(float)
    wl = (team_games["WL"] == "W").astype(float)

    # Possessions estimate
    poss = fga + 0.44 * fta - oreb + tov

    # Shift everything by 1 to avoid leakage (use only past games)
    def shifted_roll(series, w):
        return series.shift(1).rolling(w, min_periods=max(w // 2, 3)).mean()

    for w in windows:
        suffix = f"_{w}g"

        # Scoring
        feats[f"pts{suffix}"] = shifted_roll(pts, w)
        feats[f"plus_minus{suffix}"] = shifted_roll(plus_minus, w)

        # Efficiency
        _fga = shifted_roll(fga, w)
        _fgm = shifted_roll(fgm, w)
        _fg3m = shifted_roll(fg3m, w)
        _fta = shifted_roll(fta, w)
        _ftm = shifted_roll(ftm, w)

        # eFG% = (FGM + 0.5 * FG3M) / FGA
        feats[f"efg_pct{suffix}"] = (shifted_roll(fgm + 0.5 * fg3m, w)) / _fga.replace(0, np.nan)

        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        feats[f"ts_pct{suffix}"] = shifted_roll(pts, w) / (2 * (_fga + 0.44 * _fta)).replace(0, np.nan)

        # 3-point rate
        feats[f"fg3_rate{suffix}"] = shifted_roll(fg3a, w) / _fga.replace(0, np.nan)

        # Free throw rate
        feats[f"ft_rate{suffix}"] = _fta / _fga.replace(0, np.nan)

        # Turnover rate
        _poss = shifted_roll(poss, w)
        feats[f"tov_rate{suffix}"] = shifted_roll(tov, w) / _poss.replace(0, np.nan)

        # Rebound rates
        feats[f"oreb_avg{suffix}"] = shifted_roll(oreb, w)
        feats[f"ast_avg{suffix}"] = shifted_roll(ast, w)
        feats[f"stl_avg{suffix}"] = shifted_roll(stl, w)
        feats[f"blk_avg{suffix}"] = shifted_roll(blk, w)

        # Pace proxy
        feats[f"pace{suffix}"] = _poss

        # Win rate
        feats[f"win_pct{suffix}"] = shifted_roll(wl, w)

    return feats


def compute_features(paired_games, raw_games):
    """
    Compute all features for paired games.
    Returns a DataFrame with features + targets, no NaN rows.
    """
    print("Computing rolling features...")

    # Build per-team game histories from raw data
    raw = raw_games.copy()
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.sort_values("GAME_DATE").reset_index(drop=True)

    # Compute rolling stats per team
    team_rolling = {}
    team_ids = raw["TEAM_ID"].unique()
    for tid in team_ids:
        mask = raw["TEAM_ID"] == tid
        team_df = raw[mask].copy()
        team_rolling[tid] = _compute_team_rolling_stats(team_df, ROLLING_WINDOWS)

    # Now map rolling features to paired games
    features = pd.DataFrame(index=paired_games.index)

    # For each paired game, look up the home team's and away team's rolling stats
    for i, row in paired_games.iterrows():
        home_tid = row["HOME_TEAM_ID"]
        away_tid = row["AWAY_TEAM_ID"]
        game_date = row["GAME_DATE"]

        # Find the rolling stats row for this team on this date
        # Match by TEAM_ID and GAME_DATE in raw_games to get the index
        home_mask = (raw["TEAM_ID"] == home_tid) & (raw["GAME_DATE"] == game_date)
        away_mask = (raw["TEAM_ID"] == away_tid) & (raw["GAME_DATE"] == game_date)

        home_idx = raw[home_mask].index
        away_idx = raw[away_mask].index

        if len(home_idx) == 0 or len(away_idx) == 0:
            continue

        home_idx = home_idx[0]
        away_idx = away_idx[0]

        if home_tid in team_rolling and home_idx in team_rolling[home_tid].index:
            for col in team_rolling[home_tid].columns:
                features.loc[i, f"home_{col}"] = team_rolling[home_tid].loc[home_idx, col]

        if away_tid in team_rolling and away_idx in team_rolling[away_tid].index:
            for col in team_rolling[away_tid].columns:
                features.loc[i, f"away_{col}"] = team_rolling[away_tid].loc[away_idx, col]

    print(f"  Rolling features mapped: {features.shape}")

    # --- Elo ratings ---
    print("Computing Elo ratings...")
    elo_ratings = {}
    elo_for_game = []

    for i, row in paired_games.iterrows():
        home_tid = row["HOME_TEAM_ID"]
        away_tid = row["AWAY_TEAM_ID"]
        season = row["SEASON_YEAR"]

        # Initialize or season-regress
        for tid in (home_tid, away_tid):
            if tid not in elo_ratings:
                elo_ratings[tid] = {"rating": ELO_INIT, "season": season}
            elif elo_ratings[tid]["season"] != season:
                # Regress toward mean at season boundary
                elo_ratings[tid]["rating"] = (
                    ELO_SEASON_REGRESS * elo_ratings[tid]["rating"]
                    + (1 - ELO_SEASON_REGRESS) * ELO_INIT
                )
                elo_ratings[tid]["season"] = season

        home_elo = elo_ratings[home_tid]["rating"]
        away_elo = elo_ratings[away_tid]["rating"]

        elo_for_game.append({
            "idx": i,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo + ELO_HOME_ADV,
        })

        # Update Elo after game
        point_diff = row["target_point_diff"]
        expected_home = 1.0 / (1.0 + 10 ** (-(home_elo - away_elo + ELO_HOME_ADV) / 400))
        actual_home = 1.0 if point_diff > 0 else (0.5 if point_diff == 0 else 0.0)

        # Margin-of-victory multiplier
        mov_mult = math.log(max(abs(point_diff), 1) + 1) * (2.2 / (
            (home_elo - away_elo if point_diff > 0 else away_elo - home_elo) * 0.001 + 2.2
        ))

        elo_change = ELO_K * mov_mult * (actual_home - expected_home)
        elo_ratings[home_tid]["rating"] += elo_change
        elo_ratings[away_tid]["rating"] -= elo_change

    elo_df = pd.DataFrame(elo_for_game).set_index("idx")
    features["home_elo"] = elo_df["home_elo"]
    features["away_elo"] = elo_df["away_elo"]
    features["elo_diff"] = elo_df["elo_diff"]

    # --- Rest days ---
    print("Computing rest days...")
    last_game_date = {}
    rest_home = []
    rest_away = []

    for i, row in paired_games.iterrows():
        home_tid = row["HOME_TEAM_ID"]
        away_tid = row["AWAY_TEAM_ID"]
        gdate = row["GAME_DATE"]

        h_rest = (gdate - last_game_date.get(home_tid, gdate - pd.Timedelta(days=3))).days
        a_rest = (gdate - last_game_date.get(away_tid, gdate - pd.Timedelta(days=3))).days

        rest_home.append(min(h_rest, 7))  # cap at 7
        rest_away.append(min(a_rest, 7))

        last_game_date[home_tid] = gdate
        last_game_date[away_tid] = gdate

    features["home_rest_days"] = rest_home
    features["away_rest_days"] = rest_away
    features["home_b2b"] = (features["home_rest_days"] == 1).astype(float)
    features["away_b2b"] = (features["away_rest_days"] == 1).astype(float)
    features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

    # --- Calendar features ---
    features["day_of_week"] = paired_games["GAME_DATE"].dt.dayofweek / 6.0
    features["month"] = (paired_games["GAME_DATE"].dt.month - 1) / 11.0
    features["season_progress"] = paired_games.groupby("SEASON_YEAR").cumcount() / 1230.0  # ~1230 games/season

    # --- Differential features (home - away) ---
    for w in ROLLING_WINDOWS:
        suffix = f"_{w}g"
        for stat in ["pts", "efg_pct", "ts_pct", "tov_rate", "pace", "win_pct", "plus_minus"]:
            h_col = f"home_{stat}{suffix}"
            a_col = f"away_{stat}{suffix}"
            if h_col in features.columns and a_col in features.columns:
                features[f"diff_{stat}{suffix}"] = features[h_col] - features[a_col]

    # --- Targets ---
    features["target_point_diff"] = paired_games["target_point_diff"].astype(float)
    features["target_home_win"] = paired_games["target_home_win"].astype(float)

    # Drop NaN rows (from rolling warmup)
    n_before = len(features)
    features = features.dropna()
    n_after = len(features)
    print(f"  Dropped {n_before - n_after} rows with NaN ({n_after} remaining)")

    return features


# ---------------------------------------------------------------------------
# Optimized feature computation (vectorized)
# ---------------------------------------------------------------------------


def compute_features_fast(paired_games, raw_games):
    """
    Vectorized feature computation — much faster than row-by-row iteration.
    Returns a DataFrame with features + targets, no NaN rows.
    """
    print("Computing rolling features (vectorized)...")

    raw = raw_games.copy()
    raw["GAME_DATE"] = pd.to_datetime(raw["GAME_DATE"])
    raw = raw.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Compute rolling stats for all teams at once using groupby
    raw["WL_NUM"] = (raw["WL"] == "W").astype(float)
    raw["POSS"] = raw["FGA"] + 0.44 * raw["FTA"] - raw["OREB"] + raw["TOV"]
    raw["EFG_NUM"] = raw["FGM"] + 0.5 * raw["FG3M"]  # numerator for eFG%

    # Stats to compute rolling averages for
    stat_cols = ["PTS", "PLUS_MINUS", "FGA", "FGM", "FG3M", "FG3A", "FTA", "FTM",
                 "OREB", "AST", "STL", "BLK", "TOV", "POSS", "WL_NUM", "EFG_NUM"]

    # Compute shifted rolling means per team (shift first to avoid leakage)
    rolling_dfs = {}
    for w in ROLLING_WINDOWS:
        grouped = raw.groupby("TEAM_ID")[stat_cols].shift(1).groupby(raw["TEAM_ID"])
        rolled = grouped.rolling(w, min_periods=max(w // 2, 3)).mean()
        # Flatten the multi-index
        rolled = rolled.droplevel(0)
        rolled.columns = [f"{c}_roll{w}" for c in stat_cols]
        rolling_dfs[w] = rolled

    # Merge rolling stats back to raw
    for w, rdf in rolling_dfs.items():
        raw = pd.concat([raw, rdf], axis=1)

    # Create lookup: (TEAM_ID, GAME_DATE) -> row index in raw
    raw["_raw_idx"] = raw.index
    lookup = raw.set_index(["TEAM_ID", "GAME_DATE"])["_raw_idx"]
    # Handle duplicates (shouldn't happen but just in case)
    lookup = lookup[~lookup.index.duplicated(keep="first")]

    # Map rolling features to paired games
    print("  Mapping features to game pairs...")

    # Look up raw indices for home and away teams
    home_keys = list(zip(paired_games["HOME_TEAM_ID"], paired_games["GAME_DATE"]))
    away_keys = list(zip(paired_games["AWAY_TEAM_ID"], paired_games["GAME_DATE"]))

    home_raw_idx = pd.Series([lookup.get(k, -1) for k in home_keys], index=paired_games.index)
    away_raw_idx = pd.Series([lookup.get(k, -1) for k in away_keys], index=paired_games.index)

    # Filter out games where we can't find both teams
    valid = (home_raw_idx >= 0) & (away_raw_idx >= 0)
    paired_valid = paired_games[valid].copy()
    home_idx_valid = home_raw_idx[valid].values.astype(int)
    away_idx_valid = away_raw_idx[valid].values.astype(int)

    features = pd.DataFrame(index=paired_valid.index)

    # Extract rolling features
    for w in ROLLING_WINDOWS:
        suffix = f"_{w}g"
        for stat in stat_cols:
            rc = f"{stat}_roll{w}"
            if rc in raw.columns:
                features[f"home_{stat.lower()}{suffix}"] = raw[rc].values[home_idx_valid]
                features[f"away_{stat.lower()}{suffix}"] = raw[rc].values[away_idx_valid]

    # Compute derived efficiency metrics
    for w in ROLLING_WINDOWS:
        suffix = f"_{w}g"
        for side in ["home", "away"]:
            fga = features[f"{side}_fga{suffix}"].replace(0, np.nan)
            fta = features[f"{side}_fta{suffix}"].replace(0, np.nan)
            poss = features[f"{side}_poss{suffix}"].replace(0, np.nan)

            # eFG%
            features[f"{side}_efg_pct{suffix}"] = features[f"{side}_efg_num{suffix}"] / fga
            # TS%
            features[f"{side}_ts_pct{suffix}"] = features[f"{side}_pts{suffix}"] / (2 * (fga + 0.44 * fta))
            # 3pt rate
            features[f"{side}_fg3_rate{suffix}"] = features[f"{side}_fg3a{suffix}"] / fga
            # FT rate
            features[f"{side}_ft_rate{suffix}"] = fta / fga
            # TOV rate
            features[f"{side}_tov_rate{suffix}"] = features[f"{side}_tov{suffix}"] / poss

    # Drop raw stat columns we don't need as features (keep derived ones)
    drop_cols = []
    for w in ROLLING_WINDOWS:
        suffix = f"_{w}g"
        for side in ["home", "away"]:
            for raw_stat in ["fga", "fgm", "fg3m", "fg3a", "fta", "ftm", "efg_num"]:
                col = f"{side}_{raw_stat}{suffix}"
                if col in features.columns:
                    drop_cols.append(col)
    features = features.drop(columns=[c for c in drop_cols if c in features.columns])

    # --- Elo ratings ---
    print("  Computing Elo ratings...")
    elo_ratings = {}
    home_elos = np.zeros(len(paired_valid))
    away_elos = np.zeros(len(paired_valid))
    elo_diffs = np.zeros(len(paired_valid))

    for pos, (i, row) in enumerate(paired_valid.iterrows()):
        home_tid = row["HOME_TEAM_ID"]
        away_tid = row["AWAY_TEAM_ID"]
        season = row["SEASON_YEAR"]

        for tid in (home_tid, away_tid):
            if tid not in elo_ratings:
                elo_ratings[tid] = {"rating": ELO_INIT, "season": season}
            elif elo_ratings[tid]["season"] != season:
                elo_ratings[tid]["rating"] = (
                    ELO_SEASON_REGRESS * elo_ratings[tid]["rating"]
                    + (1 - ELO_SEASON_REGRESS) * ELO_INIT
                )
                elo_ratings[tid]["season"] = season

        h_elo = elo_ratings[home_tid]["rating"]
        a_elo = elo_ratings[away_tid]["rating"]

        home_elos[pos] = h_elo
        away_elos[pos] = a_elo
        elo_diffs[pos] = h_elo - a_elo + ELO_HOME_ADV

        # Update after game
        point_diff = row["target_point_diff"] if "target_point_diff" not in features.columns else (
            row["HOME_PTS"] - row["AWAY_PTS"]
        )
        expected = 1.0 / (1.0 + 10 ** (-(h_elo - a_elo + ELO_HOME_ADV) / 400))
        actual = 1.0 if point_diff > 0 else (0.5 if point_diff == 0 else 0.0)
        mov_mult = math.log(max(abs(point_diff), 1) + 1) * (
            2.2 / ((h_elo - a_elo if point_diff > 0 else a_elo - h_elo) * 0.001 + 2.2)
        )
        change = ELO_K * mov_mult * (actual - expected)
        elo_ratings[home_tid]["rating"] += change
        elo_ratings[away_tid]["rating"] -= change

    features["home_elo"] = home_elos
    features["away_elo"] = away_elos
    features["elo_diff"] = elo_diffs

    # --- Rest days ---
    print("  Computing rest days...")
    last_game = {}
    h_rest = np.full(len(paired_valid), 3.0)
    a_rest = np.full(len(paired_valid), 3.0)

    for pos, (i, row) in enumerate(paired_valid.iterrows()):
        htid = row["HOME_TEAM_ID"]
        atid = row["AWAY_TEAM_ID"]
        gd = row["GAME_DATE"]

        if htid in last_game:
            h_rest[pos] = min((gd - last_game[htid]).days, 7)
        if atid in last_game:
            a_rest[pos] = min((gd - last_game[atid]).days, 7)

        last_game[htid] = gd
        last_game[atid] = gd

    features["home_rest_days"] = h_rest
    features["away_rest_days"] = a_rest
    features["home_b2b"] = (features["home_rest_days"] == 1).astype(float)
    features["away_b2b"] = (features["away_rest_days"] == 1).astype(float)
    features["rest_advantage"] = features["home_rest_days"] - features["away_rest_days"]

    # --- Calendar features ---
    features["day_of_week"] = paired_valid["GAME_DATE"].dt.dayofweek / 6.0
    features["month"] = (paired_valid["GAME_DATE"].dt.month - 1) / 11.0
    season_game_num = paired_valid.groupby("SEASON_YEAR").cumcount()
    features["season_progress"] = season_game_num / season_game_num.groupby(paired_valid["SEASON_YEAR"]).transform("max").replace(0, 1)

    # --- Scoring variance (consistency) ---
    print("  Computing scoring variance...")
    raw_sorted = raw.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    for w in ROLLING_WINDOWS:
        std_col = f"PTS_std{w}"
        raw_sorted[std_col] = (
            raw_sorted.groupby("TEAM_ID")["PTS"]
            .shift(1)
            .groupby(raw_sorted["TEAM_ID"])
            .rolling(w, min_periods=max(w // 2, 3))
            .std()
            .droplevel(0)
        )
        # Map to features via lookup
        lookup_std = raw_sorted.set_index(["TEAM_ID", "GAME_DATE"])[std_col]
        lookup_std = lookup_std[~lookup_std.index.duplicated(keep="first")]
        h_std = pd.Series([lookup_std.get(k, np.nan) for k in home_keys], index=paired_valid.index)
        a_std = pd.Series([lookup_std.get(k, np.nan) for k in away_keys], index=paired_valid.index)
        features[f"home_pts_std_{w}g"] = h_std.values
        features[f"away_pts_std_{w}g"] = a_std.values

    # --- Win/loss streaks ---
    print("  Computing streaks...")
    raw_str = raw.sort_values(["TEAM_ID", "GAME_DATE"]).copy()
    raw_str["WIN"] = (raw_str["WL"] == "W").astype(int)
    # Compute current streak: positive = win streak, negative = loss streak
    def compute_streak(group):
        streaks = []
        current = 0
        for w in group["WIN"].values:
            if w == 1:
                current = max(current, 0) + 1
            else:
                current = min(current, 0) - 1
            streaks.append(current)
        # Shift by 1 to avoid leakage
        return pd.Series([0] + streaks[:-1], index=group.index)
    raw_str["STREAK"] = raw_str.groupby("TEAM_ID", group_keys=False).apply(compute_streak)
    lookup_streak = raw_str.set_index(["TEAM_ID", "GAME_DATE"])["STREAK"]
    lookup_streak = lookup_streak[~lookup_streak.index.duplicated(keep="first")]
    features["home_streak"] = [lookup_streak.get(k, 0) for k in home_keys]
    features["away_streak"] = [lookup_streak.get(k, 0) for k in away_keys]
    features["streak_diff"] = features["home_streak"] - features["away_streak"]

    # --- Player concentration features ---
    print("  Computing player features...")
    if os.path.exists(PLAYER_RAW_PATH):
        with open(PLAYER_RAW_PATH, "rb") as f:
            player_cache = pickle.load(f)
        player_games = player_cache["games"]
        player_games["GAME_DATE"] = pd.to_datetime(player_games["GAME_DATE"])
        player_games = player_games.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

        # Convert MIN to float (format is "MM:SS")
        def parse_min(x):
            if pd.isna(x) or x == "" or x is None:
                return 0.0
            try:
                parts = str(x).split(":")
                return float(parts[0]) + float(parts[1]) / 60 if len(parts) == 2 else float(parts[0])
            except (ValueError, IndexError):
                return 0.0

        player_games["MIN_FLOAT"] = player_games["MIN"].apply(parse_min)

        # For each team-game, compute star concentration (top-2 scorer share)
        # and roster depth (players with >15 min)
        def team_game_stats(group):
            pts = group["PTS"].astype(float)
            mins = group["MIN_FLOAT"]
            total_pts = pts.sum()
            if total_pts == 0:
                return pd.Series({"star_share": 0.5, "depth": 8})
            top2 = pts.nlargest(2).sum()
            star_share = top2 / total_pts
            depth = (mins >= 15).sum()
            return pd.Series({"star_share": star_share, "depth": float(depth)})

        pg_stats = player_games.groupby(["TEAM_ID", "GAME_DATE"]).apply(
            team_game_stats, include_groups=False
        )
        pg_stats = pg_stats.reset_index()

        # Compute rolling averages (shifted to avoid leakage)
        pg_stats = pg_stats.sort_values(["TEAM_ID", "GAME_DATE"])
        for stat_col in ["star_share", "depth"]:
            for w in [10]:  # single window for simplicity
                roll_col = f"{stat_col}_{w}g"
                pg_stats[roll_col] = (
                    pg_stats.groupby("TEAM_ID")[stat_col]
                    .shift(1)
                    .groupby(pg_stats["TEAM_ID"])
                    .rolling(w, min_periods=5)
                    .mean()
                    .droplevel(0)
                )

        # Map to features
        pg_lookup = pg_stats.set_index(["TEAM_ID", "GAME_DATE"])
        pg_lookup = pg_lookup[~pg_lookup.index.duplicated(keep="first")]

        for stat_col in ["star_share_10g", "depth_10g"]:
            h_vals = [pg_lookup[stat_col].get(k, np.nan) if k in pg_lookup.index else np.nan for k in home_keys]
            a_vals = [pg_lookup[stat_col].get(k, np.nan) if k in pg_lookup.index else np.nan for k in away_keys]
            features[f"home_{stat_col}"] = h_vals
            features[f"away_{stat_col}"] = a_vals

        # Fill NaN with league averages (don't drop games)
        for col in ["home_star_share_10g", "away_star_share_10g"]:
            if col in features.columns:
                features[col] = features[col].fillna(0.45)
        for col in ["home_depth_10g", "away_depth_10g"]:
            if col in features.columns:
                features[col] = features[col].fillna(8.0)
        print(f"    Added player features: star_share_10g, depth_10g (NaN filled with defaults)")
    else:
        print("    No player data found, skipping player features")

    # --- Differential features ---
    for w in ROLLING_WINDOWS:
        suffix = f"_{w}g"
        for stat in ["pts", "plus_minus", "efg_pct", "ts_pct", "tov_rate",
                      "poss", "wl_num", "oreb", "ast", "stl", "blk",
                      "pts_std"]:
            h_col = f"home_{stat}{suffix}"
            a_col = f"away_{stat}{suffix}"
            if h_col in features.columns and a_col in features.columns:
                features[f"diff_{stat}{suffix}"] = features[h_col] - features[a_col]

    # Player feature diffs
    for stat_col in ["star_share_10g", "depth_10g"]:
        h_col = f"home_{stat_col}"
        a_col = f"away_{stat_col}"
        if h_col in features.columns and a_col in features.columns:
            features[f"diff_{stat_col}"] = features[h_col] - features[a_col]

    # --- Targets ---
    features["target_point_diff"] = (paired_valid["HOME_PTS"] - paired_valid["AWAY_PTS"]).astype(float)
    features["target_home_win"] = (features["target_point_diff"] > 0).astype(float)

    # Drop NaN rows
    n_before = len(features)
    features = features.dropna()
    n_after = len(features)
    print(f"  Dropped {n_before - n_after} rows with NaN ({n_after} remaining)")

    return features


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def download_data(n_seasons=20):
    """Download and prepare all NBA data."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check processed cache
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "rb") as f:
            cached = pickle.load(f)
        last_date = cached.get("last_game_date")
        if last_date:
            days_old = (pd.Timestamp.now() - pd.Timestamp(last_date)).days
            if days_old < 2:
                print(f"Data: using cache ({cached['n_games']} games, last: {last_date})")
                return cached

    print(f"Data: preparing {n_seasons} seasons of NBA data...")
    t0 = time.time()

    # Download raw games
    raw_games = download_raw_games(n_seasons)

    # Build game pairs
    paired = build_game_pairs(raw_games)

    # Compute features
    features = compute_features_fast(paired, raw_games)

    # Split indices (time-based)
    n = len(features)
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VAL_FRACTION)
    n_train = n - n_val - n_test

    feature_cols = [c for c in features.columns if not c.startswith("target_")]

    result = {
        "features": features,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_games": n,
        "feature_cols": feature_cols,
        "target_cols": ["target_point_diff", "target_home_win"],
        "last_game_date": features.index[-1] if isinstance(features.index, pd.DatetimeIndex) else str(paired["GAME_DATE"].iloc[-1].date()),
    }

    with open(DATA_PATH, "wb") as f:
        pickle.dump(result, f)

    t1 = time.time()
    print(f"Data: {n} games ({n_train} train / {n_val} val / {n_test} test)")
    print(f"Data: {len(feature_cols)} features, saved to {DATA_PATH}")
    print(f"Data: completed in {t1 - t0:.1f}s")
    return result


def load_data():
    """Load prepared data. Returns dict with arrays and metadata."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"No prepared data at {DATA_PATH}. Run prepare.py first.")
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


class DataLoader:
    """
    Provides train/val batches as MLX arrays.
    Features are z-score normalized using training set statistics.
    """

    def __init__(self, batch_size, split="train"):
        data = load_data()
        features = data["features"]
        feature_cols = data["feature_cols"]
        n_train = data["n_train"]
        n_val = data["n_val"]

        X_all = features[feature_cols].values.astype(np.float32)
        y_diff = features["target_point_diff"].values.astype(np.float32)
        y_win = features["target_home_win"].values.astype(np.float32)

        # Normalize using training set stats
        self.train_mean = X_all[:n_train].mean(axis=0)
        self.train_std = X_all[:n_train].std(axis=0)
        self.train_std[self.train_std < 1e-8] = 1.0

        X_all = (X_all - self.train_mean) / self.train_std
        X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

        if split == "train":
            self.X = X_all[:n_train]
            self.y_diff = y_diff[:n_train]
            self.y_win = y_win[:n_train]
        elif split == "val":
            self.X = X_all[n_train:n_train + n_val]
            self.y_diff = y_diff[n_train:n_train + n_val]
            self.y_win = y_win[n_train:n_train + n_val]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.batch_size = batch_size
        self.n_samples = len(self.X)
        self.n_features = self.X.shape[1]
        self.n_batches = max(1, self.n_samples // batch_size)
        self._indices = np.arange(self.n_samples)

    def shuffle(self):
        np.random.shuffle(self._indices)

    def __iter__(self):
        for i in range(self.n_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.n_samples)
            idx = self._indices[start:end]
            X_batch = mx.array(self.X[idx])
            y_d = mx.array(self.y_diff[idx])
            y_w = mx.array(self.y_win[idx])
            yield X_batch, y_d, y_w

    def __len__(self):
        return self.n_batches


def get_n_features():
    """Return the number of input features."""
    data = load_data()
    return len(data["feature_cols"])


def get_feature_names():
    """Return list of feature column names."""
    data = load_data()
    return data["feature_cols"]


# ---------------------------------------------------------------------------
# Evaluation (ground truth metric — do not modify)
# ---------------------------------------------------------------------------


def evaluate(model, batch_size=256):
    """
    Evaluate model on validation set. Returns a dict with:
      - val_score: combined metric (lower is better)
      - win_accuracy: fraction of correct home/away win predictions
      - mae_point_diff: mean absolute error of point differential predictions
      - mean_loss: average combined loss on val set

    val_score = (1 - win_accuracy) + 0.05 * mae_point_diff

    The agent optimizes for val_score (lower is better).
    Home teams win ~58% of games, so always-home baseline gives val_score ~ 0.42 + penalty.
    A good model should get win_accuracy > 65% and MAE < 9.
    """
    val_loader = DataLoader(batch_size, split="val")

    all_pred_win = []
    all_true_win = []
    all_pred_diff = []
    all_true_diff = []
    total_loss = 0.0
    n_batches = 0

    for X, y_diff, y_win in val_loader:
        pred_diff, win_logit = model(X)

        pred_win_prob = mx.sigmoid(win_logit)
        pred_win_class = (pred_win_prob > 0.5).astype(mx.float32)

        all_pred_win.append(np.array(pred_win_class.reshape(-1)))
        all_true_win.append(np.array(y_win.reshape(-1)))
        all_pred_diff.append(np.array(pred_diff.reshape(-1)))
        all_true_diff.append(np.array(y_diff.reshape(-1)))

        # Loss for logging
        mse = mx.mean((pred_diff.reshape(-1) - y_diff) ** 2)
        bce = mx.mean(
            mx.maximum(win_logit.reshape(-1), mx.array(0.0))
            - win_logit.reshape(-1) * y_win
            + mx.log1p(mx.exp(-mx.abs(win_logit.reshape(-1))))
        )
        total_loss += float((mse + bce).item())
        n_batches += 1

    all_pred_win = np.concatenate(all_pred_win)
    all_true_win = np.concatenate(all_true_win)
    all_pred_diff = np.concatenate(all_pred_diff)
    all_true_diff = np.concatenate(all_true_diff)

    win_accuracy = float(np.mean(all_pred_win == all_true_win))
    mae_point_diff = float(np.mean(np.abs(all_pred_diff - all_true_diff)))
    mean_loss = total_loss / max(n_batches, 1)

    # Combined score: lower is better
    # Win accuracy is most important (negated so lower = better)
    # MAE secondary, scaled to similar range (~9 MAE * 0.05 = 0.45)
    val_score = (1.0 - win_accuracy) + 0.05 * mae_point_diff

    return {
        "val_score": val_score,
        "win_accuracy": win_accuracy,
        "mae_point_diff": mae_point_diff,
        "mean_loss": mean_loss,
    }


# ---------------------------------------------------------------------------
# Main: download and prepare
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NBA prediction data")
    parser.add_argument("--seasons", type=int, default=20, help="Number of seasons to download")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()
    result = download_data(n_seasons=args.seasons)
    print()
    print(f"Features ({len(result['feature_cols'])}):")
    for i, col in enumerate(result['feature_cols']):
        print(f"  {i+1:3d}. {col}")
    print()
    print("Done! Ready to train.")
