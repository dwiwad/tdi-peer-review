"""
Created on Tue Sep 16 07:45:52 2025

@author: dwiwad
"""
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
from wakepy import keep
import statsmodels.formula.api as smf
from scipy.stats import zscore
import json
from pathlib import Path
import s3fs
from app.core.config import S3_BUCKET


##############################################################################
#
# GET THE GAME DATA THAT I NEED
#
##############################################################################

##############################################################################
# GAME IDS
##############################################################################

# Getting every game_id from 2010 to 2025


# Helper, get mondays in a range inclusive
def get_mondays(start_date, end_date):
    # empty list and date to use
    mondays = []
    current_date = start_date
    # If current date is not a monday, add a day
    while current_date.weekday() != 0:
        current_date += timedelta(days=1)
    # If the current day is less than the end day, keep going to the next week
    # And also, if current date is a monday, add to list
    while current_date <= end_date:
        mondays.append(current_date)
        current_date += timedelta(weeks=1)
    return mondays

# Pull every monday for 15 years
start_date = date(2010, 9, 1)
end_date = date(2025, 6, 23)

mondays = get_mondays(start_date, end_date)

# Okay now a function to get all the games
# Just takes a monday, pulls the games for that week, appends to list if there are games, returns it
def get_all_games(list_of_mondays):
    
    # Initialize a list
    games = []
    n = 1
    
    for monday in list_of_mondays:
        print(f"Working on week { n } of { len(list_of_mondays) }")
        # Make the url, pull the data from that day
        url = f"https://api-web.nhle.com/v1/schedule/{ monday }"
        response = requests.get(url)
        data = response.json()
        
        # Pull the game data for every day that week
        for i in range(7):
            game_list = data['gameWeek'][i]['games']
            if len(game_list) > 0:
                games.extend(game_list)
        n += 1
        
        # Add a quick pause to be polite
        time.sleep(2)
    
    return games
   
# Fetch and flatten
with keep.running():
    games= get_all_games(mondays)

games = pd.json_normalize(games)

# Save the file for later
games.to_csv('~/dev/hockey_site/data/total-depth-index/all_games_meta_20102025.csv', index=False)

##############################################################################
# GAME PLAY-BY-PLAY
##############################################################################   

# Function to get the play by play data for every game
def fetch_game_pbp(list_of_games):
    
    # Initialize a list
    plays = []
    n = 1
    
    for game in list_of_games:
        print(f"working on game { n } of { len(list_of_games) }")
        # Build the actual url and make the call
        url = f"https://api-web.nhle.com/v1/gamecenter/{ game }/play-by-play"
        response = requests.get(url)
        data = response.json()
        
        # I need to add a game_id to each play
        for d in data['plays']:
            d['game_id'] = game
        
        plays.extend(data['plays'])
        
        n += 1
        
        # Add a quick pause to be polite
        time.sleep(2)
    
    return plays

# Get the list of games:
list_of_games = games['id']

# Fetch and flatten
with keep.running():
    pbp = fetch_game_pbp(list_of_games)

pbp = pd.json_normalize(pbp)

# Save the data
pbp.to_csv('~/dev/hockey_site/data/total-depth-index/all_pbp_20102025.csv', index=False)

##############################################################################
# GET THE ROSTER INFORMATION
############################################################################## 

# This was overall a huge inefficiency. This function should have been combined with getting the
# Game data--I effectively pulled 20k game pbps twice. Such is life.
# For the season make a df that is game_id, away_abbrev, home_abbrev
teams = games[['id', 'awayTeam.abbrev', 'homeTeam.abbrev']].rename(columns = {'id': 'game_id'})

# Merge in the abbrevs
pbp = pd.merge(pbp, teams, on = 'game_id', how = 'inner')

# I need player lookups too... I'm aware this isn't the most efficient way to
# do this. Maybe before posting I'll create a single fetch pbp and roster func.
# New version that batches
def fetch_game_roster(list_of_games, batch_size=500, out_path="/Users/dwiwad/dev/hockey_site/data/total-depth-index/all_rosters_20102025.ndjson"):
    out_file = Path(out_path)
    n_games = len(list_of_games)
    batch = []
    
    for idx, game in enumerate(list_of_games, 1):
        print(f"working on game {idx} of {n_games}")
        url = f"https://api-web.nhle.com/v1/gamecenter/{game}/play-by-play"
        response = requests.get(url, timeout=(5, 20))
        response.raise_for_status()
        data = response.json()
        
        # Build teamId → abbrev lookup
        team_lookup = {}
        for side in ("awayTeam", "homeTeam"):
            t = data.get(side, {}) or {}
            if t:
                team_lookup[t.get("id")] = t.get("abbrev")
        
        for d in data.get("rosterSpots", []):
            d["game_id"] = game
            d["teamAbbrev"] = team_lookup.get(d.get("teamId"))
            batch.append(d)
        
        # polite delay
        time.sleep(1.5)
        
        # ---- batch commit ----
        if idx % batch_size == 0 or idx == n_games:
            # Only write if batch filled without errors
            with out_file.open("a") as f:
                for row in batch:
                    f.write(json.dumps(row))
                    f.write("\n")
            print(f"  Saved batch ending at game {idx}, {len(batch)} rows")
            batch = []  # reset for next batch

with keep.running():
    fetch_game_roster(
        list_of_games=games["id"].tolist(),
        batch_size=500
    )
    
    
# Load NDJSON into pandas
df = pd.read_json("~/dev/hockey_site/data/total-depth-index/all_rosters_20102025.ndjson", lines=True)

# Flatten periodDescriptor
fn_flat = pd.json_normalize(df["firstName"]).rename(columns = {'default': 'firstName'})

# Flatten details
ln_flat = pd.json_normalize(df["lastName"]).rename(columns = {'default': 'lastName'})

# Combine everything
df_flat = pd.concat([df.drop(columns=["firstName","lastName"]), fn_flat, ln_flat], axis=1)

df_flat.to_csv("~/dev/hockey_site/data/total-depth-index/all_rosters_20102025.csv", index=False)    
    
##############################################################################
# GET SHIFT DATA FOR TOI CALCULATIONS
############################################################################## 

# The shift data is here:
# https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId=2024010001

# game_id, playerId, duration
# Function to get the shift data for every game
def fetch_game_shifts(
    list_of_games,
    batch_size=500,
    out_path="/Users/dwiwad/dev/hockey_site/data/total-depth-index/all_shifts_20102025.ndjson",
    per_request_sleep=1.5,
    retries=3,
    backoff=2.0,
):
    """
    Pulls NHL shift charts for each game in list_of_games and streams to NDJSON in batches.
    - Batches every `batch_size` games
    - Adds 'game_id' to each row
    - Retries transient failures per game up to `retries` times with exponential backoff
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_games = len(list_of_games)
    batch = []

    for idx, game in enumerate(list_of_games, 1):
        print(f"working on game {idx} of {n_games}", end = "\r", flush = True)

        url = f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game}"

        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(url, timeout=(5, 20))
                response.raise_for_status()
                data = response.json() or {}
                rows = data.get("data", []) or []

                # Attach game_id and collect
                for d in rows:
                    d["game_id"] = game
                batch.extend(rows)

                # polite delay
                time.sleep(per_request_sleep)
                break  # success; exit retry loop

            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                attempt += 1
                if attempt >= retries:
                    # Log and move on; don't kill the whole job
                    print(f"  FAILED game {game} after {retries} attempts: {e}")
                else:
                    delay = backoff ** attempt
                    print(f"  retry {attempt}/{retries} for game {game} in {delay:.1f}s due to: {e}")
                    time.sleep(delay)

        # ---- batch commit ----
        if (idx % batch_size == 0 or idx == n_games) and batch:
            with out_file.open("a") as f:
                for row in batch:
                    f.write(json.dumps(row))
                    f.write("\n")
            print(f"  Saved batch ending at game {idx}, {len(batch)} rows")
            batch = []  # reset for next batch

# Example usage (mirrors your roster call)
with keep.running():
    fetch_game_shifts(
        list_of_games=games["id"].tolist(),
        batch_size=500
    )

##############################################################################
#
# DEFINING, CALCULATING, AND MERGING ALL THE RELEVANT DATA
#
##############################################################################      
    
##############################################################################
# SHOTS-ON-GOAL AND GOAL WITH PLAYER AND TEAM ID
##############################################################################  

# Had to clear the environment. Reload data.
games = pd.read_csv('~/dev/hockey_site/data/total-depth-index/all_seasons/all_games_meta_20102025.csv')
pbp = pd.read_csv('~/dev/hockey_site/data/total-depth-index/all_seasons/all_pbp_20102025.csv')
# Shifts I did separately in all_season_shift_depth.py
#shifts = pd.read_csv('~/dev/hockey_site/data/total-depth-index/all_shifts_20242025.csv')
rosters = pd.read_csv('~/dev/hockey_site/data/total-depth-index/all_seasons/all_rosters_20102025.csv')
#mpxg = pd.read_csv('~/dev/hockey_site/data/total-depth-index/moneypuck_xg_20242025.csv')

# Let's start with team by team. Perhaps just with Edmonton.
# For every game, create a df that is:
# game_id, player_name, shots_on_goal, goals, shot_attempts
# I am just gonna wanna merge counts into the roster data I think, preserving 0s

# What I need to do now is loop through shots and count by player_id and add to rosters.
pbp['typeDescKey'].value_counts()

# For shots, remove shootouts and shrink to shots-on-goal
shots = pbp[pbp['period_periodType'] != 'SO']
shots = shots[shots['typeDescKey'].isin(['shot-on-goal', 'goal'])]    
# Quick check
shots['typeDescKey'].value_counts()    

# Create a playerId col that uses scoring or shooting playerId
shots['playerId'] = np.where(
    shots['typeDescKey'].eq('goal'),
    shots['detail_scoringPlayerId'],
    shots['detail_shootingPlayerId']
)

# Count shots-on-goal per player per game
game_sogs = (
    shots
    .groupby('game_id')['playerId']
    .value_counts()
    .reset_index(name='sog_count')
)


# Merge into the roster
rosters = pd.merge(rosters, game_sogs, on = ['game_id', 'playerId'], how = 'left').fillna(0)

##############################################################################
# ASSISTS BY PLAYER_ID
##############################################################################

# We can apply the same player counting logic for assists. I'm realizing we don't
# even need to filter the data we can just count
# What I need to do now is loop through shots and count by player_id and add to rosters.
pbp['typeDescKey'].value_counts()

# For shots, remove shootouts and shrink to goals; the only thing w/ assists
goals = pbp[pbp['period_periodType'] != 'SO']
goals = goals[goals['typeDescKey'].isin(['goal'])]    

# Stack both assist columns into one
assists = goals.melt(
    id_vars=['game_id'], 
    value_vars=['detail_assist1PlayerId', 'detail_assist2PlayerId'],
    value_name='playerId'
)

# Count assists per player per game
game_assists = (
    assists
    .groupby(['game_id', 'playerId'])
    .size()
    .reset_index(name='assist_count')
)

# Merge into the roster
rosters = pd.merge(rosters, game_assists, on = ['game_id', 'playerId'], how = 'left').fillna(0)

##############################################################################
# CORSI FOR - SHOT ATTEMPTS AT EVEN STRENGTH
##############################################################################

pbp['typeDescKey'].value_counts()

# For shots, remove shootouts and shrink to goals; the only thing w/ assists
corsi = pbp[pbp['period_periodType'] != 'SO']
corsi = corsi[corsi['typeDescKey'].isin(['shot-on-goal', 'blocked-shot', 'missed-shot', 'goal'])]  

# It seeeems like situationCode describes players on ice so 1551 is even strength
# https://gitlab.com/dword4/nhlapi/-/issues/112
corsi = corsi[corsi['situationCode'] == 1551]

# Quick check
corsi['typeDescKey'].value_counts()    

# Create a playerId col that uses scoring or shooting playerId
corsi['playerId'] = np.where(
    corsi['typeDescKey'].eq('goal'),
    corsi['detail_scoringPlayerId'],
    corsi['detail_shootingPlayerId']
)

# Count shots-on-goal per player per game
game_cf = (
    corsi
    .groupby('game_id')['playerId']
    .value_counts()
    .reset_index(name='corsi_for')
)


# Merge into the roster
rosters = pd.merge(rosters, game_cf, on = ['game_id', 'playerId'], how = 'left').fillna(0)

# Proper split: bucket + prefix + object
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"
OBJECT = "roster_with_sog_assist_corsi.csv"

fs = s3fs.S3FileSystem(anon=False)

# Build full S3 path
out_path = f"s3://{S3_BUCKET}/{PREFIX}/{OBJECT}"

# Write DataFrame
rosters.to_csv(out_path, index=False, storage_options={"anon": False})

print(f"Wrote {len(rosters)} rows to {out_path}")

##############################################################################
# EXPECTED GOALS
##############################################################################

# Pull all the expected goals for each season
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"

fs = s3fs.S3FileSystem(anon=False)

# The game id is going to need to be changed. In my data 2024020003 becomes 20003
# or 2024030416 becomes 30416
# Moneypuck drops 20240, so we can just append those back on.
xg = {}
for year in range(2010, 2025):
    path = f"s3://{S3_BUCKET}/{PREFIX}/shots_{year}.csv"
    print(f"Loading {path}")
    df = pd.read_csv(path, storage_options={"anon": False})
    # prepend the season year and left-pad the short id to 6 digits, e.g., 20001 -> 020001 -> 2024020001
    df["game_id"] = df["game_id"].apply(lambda v: f"{year}{str(int(v)).zfill(6)}")
    xg[year] = df

# combine all years
xg_all = pd.concat(xg.values(), ignore_index=True)
xg_all["game_id"] = xg_all["game_id"].astype("int64")

# This is a bit different. I'm using Moneypuck's data as a simple test. I would
# eventually want to recreate his model I think for prediction, but as far as 
# explanation and the SEM goes, simply using his data should be sufficient.
cols = ['game_id', 'shooterPlayerId', 'xGoal']

xg_all = xg_all[cols].rename(columns = {'shooterPlayerId': 'playerId'})


# Count mean xG per player per game
game_xg = (
    xg_all
    .groupby(['game_id', 'playerId'])['xGoal']
    .sum()
    .reset_index(name='sum_xg')
)

# Merge into the roster
rosters = pd.merge(rosters, game_xg, on = ['game_id', 'playerId'], how = 'left').fillna(0)

# Proper split: bucket + prefix + object
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"
OBJECT = "roster_with_sog_assist_corsi_xg.csv"

fs = s3fs.S3FileSystem(anon=False)

# Build full S3 path
out_path = f"s3://{S3_BUCKET}/{PREFIX}/{OBJECT}"

# Write DataFrame
rosters.to_csv(out_path, index=False, storage_options={"anon": False})

print(f"Wrote {len(rosters)} rows to {out_path}")




















