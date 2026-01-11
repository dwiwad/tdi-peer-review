import pandas as pd
import numpy as np
import s3fs
from app.core.config import S3_BUCKET

# Read the roster_with_sog_assist_corsi_xg.csv from S3
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"
OBJECT = "roster_with_sog_assist_corsi_xg.csv"

fs = s3fs.S3FileSystem(anon=False)
path = f"s3://{S3_BUCKET}/{PREFIX}/{OBJECT}"

print(f"Loading {path}")
rosters = pd.read_csv(path, storage_options={"anon": False}, low_memory=False)

print(f"Loaded {len(rosters)} rows")

##############################################################################
# BUILD GAME LEVEL METRICS AND SAVE
##############################################################################

# Shrink the data to not include goalies
shooters = rosters[rosters['positionCode'] != 'G']

# Bring in Olivia Guest's gini func, modified slightly
# https://github.com/oliviaguest/gini

def gini(x, eps=1e-9):
    """Gini coefficient for a 1D array-like of nonnegative values."""
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return np.nan
    # Shift up if any negatives (shouldn't happen for SOG, but safe)
    amin = a.min()
    if amin < 0:
        a = a - amin
    s = a.sum()
    if s <= 0:
        return 0.0  # all zeros -> perfectly equal
    a = np.sort(a)
    n = a.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return ((2 * idx - n - 1) @ a) / (n * s + eps)


# Shot depth; This will be the root final data we build on
game_data = (
    shooters.groupby(['game_id', 'teamAbbrev'])['sog_count']
      .apply(gini)
      .rename('sog_gini')
      .reset_index()
)

# Get the total sogs by team game
sogs_by_team_game = (
    shooters.groupby(['game_id', 'teamAbbrev'])['sog_count']
      .sum()
      .rename('total_sogs')
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             sogs_by_team_game[['game_id', 'teamAbbrev', 'total_sogs']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# Bring in the game outcome
games = pd.read_csv('~/dev/hockey_site/data/total-depth-index/all_seasons/all_games_meta_20102025.csv')
games['winAbbrev'] = np.where(games['awayTeam.score'] > games['homeTeam.score'], games['awayTeam.abbrev'], games['homeTeam.abbrev'])

# Merge it in
games = games.rename(columns = {'id': 'game_id'})

game_data = game_data.merge(
    games[["game_id", "winAbbrev"]],
    on="game_id",
    how="left"
)

game_data['outcome'] = np.where(game_data['teamAbbrev'] == game_data['winAbbrev'], 1, 0)

# Get the assist gini
assist_gini = (
    shooters.groupby(['game_id', 'teamAbbrev'])['assist_count']
      .apply(gini)
      .rename('assist_gini')
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             assist_gini[['game_id', 'teamAbbrev', 'assist_gini']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# Get the xG gini
xgoal = (
    shooters.groupby(['game_id', 'teamAbbrev'])['sum_xg']
      .apply(gini)
      .rename('xgoal_gini')
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             xgoal[['game_id', 'teamAbbrev', 'xgoal_gini']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# Also just xG so I can calc differentials
xgoal_by_team_game = (
    shooters.groupby(['game_id', 'teamAbbrev'])['sum_xg']
      .sum()
      .rename('xgoal')
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             xgoal_by_team_game[['game_id', 'teamAbbrev', 'xgoal']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# Get the cf gini
cf_gini = (
    shooters.groupby(['game_id', 'teamAbbrev'])['corsi_for']
      .apply(gini)
      .rename('cf_gini')
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             cf_gini[['game_id', 'teamAbbrev', 'cf_gini']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# CF = shots-on-goal + goal + blocks + misses
# Get the total by team game
cf_by_team_game = (
    shooters.groupby(['game_id', 'teamAbbrev'])['corsi_for']
      .sum()
      .reset_index()
)

game_data = (
    pd.merge(game_data,
             cf_by_team_game[['game_id', 'teamAbbrev', 'corsi_for']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)


# Get the TOI gini
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"

fs = s3fs.S3FileSystem(anon=False)

dfs = []
for year in range(2010, 2025):
    path = f"s3://{S3_BUCKET}/{PREFIX}/toi_gini_{year}.csv"
    print(f"Loading {path}")
    dfs.append(pd.read_csv(path, storage_options={"anon": False}))

toi_gini_all = pd.concat(dfs, ignore_index=True)
print(f"Stacked {len(toi_gini_all)} rows total")


game_data = (
    pd.merge(game_data,
             toi_gini_all[['game_id', 'teamAbbrev', 'toi_gini']], 
             on=['game_id', 'teamAbbrev'], 
             how='left')
)

# This is just myself wanting it in the right order lol
new_order = ['game_id', 'teamAbbrev', 'outcome', 'total_sogs', 'xgoal', 'sog_gini', 
             'assist_gini', 'toi_gini', 'xgoal_gini', 'cf_gini', 'corsi_for']

# Reassign the DataFrame with the new column order
game_data = game_data[new_order]

PREFIX = "static-ds-analyses/total-depth-index/all-seasons"
OBJECT = "final_game_data_20102025.csv"

fs = s3fs.S3FileSystem(anon=False)
out_path = f"s3://{S3_BUCKET}/{PREFIX}/{OBJECT}"

game_data.to_csv(out_path, index=False, storage_options={"anon": False})
print(f"Wrote {len(game_data)} rows to {out_path}")







  