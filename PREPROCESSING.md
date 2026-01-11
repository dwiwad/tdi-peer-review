 # Data Preprocessing Pipeline

  This document describes the complete data pipeline from raw NHL API data to the final analysis dataset.

  ---

  ## Pipeline Overview

  NHL API + Moneypuck
          ↓
  Raw Data Collection (12+ hours)
          ↓
  Player-Level Statistics
          ↓
  Gini Coefficient Calculation
          ↓
  Depth Metrics (1 - Gini)
          ↓
  Standardization & SEM
          ↓
  final_data_20102025.csv

  ---

  ## Step 1: Data Collection

  **Script:** `scripts/data_wrangling_all_seasons.py`

  ### NHL API Endpoints

  1. **Schedule Endpoint**
     - URL: `https://api-web.nhle.com/v1/schedule/{date}`
     - Fetches all games for each day from 2010-10-01 to 2025-01-10
     - Iterates weekly to avoid overloading API
     - Extracts: game_id, teams, scores, dates

  2. **Play-by-Play Endpoint**
     - URL: `https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play`
     - Fetches all events (shots, goals, penalties, etc.)
     - Extracts shooter IDs for shot distribution analysis
     - **Rate limit:** 2-second delay between requests

  3. **Shift Charts Endpoint**
     - URL: `https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}`
     - Fetches time-on-ice data for each player
     - Format: Start time, end time, duration per shift
     - Output: NDJSON format (5.8 GB total)

  ### Moneypuck Data

  - **Source:** https://moneypuck.com/data.htm
  - **Files:** `shots_2010.csv` through `shots_2024.csv`
  - **Contains:** Expected goals (xG) for every shot
  - **Join key:** `game_id` + shooter player ID

  ### Output Files

  - `all_games_meta_20102025.csv` (18 MB) - Game metadata
  - `all_pbp_20102025.csv` (817 MB) - Play-by-play events
  - `all_rosters_20102025.csv` (94 MB) - Player stats per game
  - `all_shifts_20102025.ndjson` (5.8 GB) - Shift-level TOI data

  **Runtime:** ~12 hours (due to rate limiting)

  ---

  ## Step 2: Time-on-Ice Depth

  **Script:** `scripts/all_season_shift_depth.py`

  ### Process

  1. Stream `all_shifts_20102025.ndjson` (too large to load into memory)
  2. Convert shift durations (MM:SS format) to seconds
  3. Sum total TOI per player per game
  4. Calculate Gini coefficient across 18 rostered skaters
  5. Output `toi_gini` for each team-game

  ### Gini Calculation

  ```python
  def gini(x):
      """
      Calculate Gini coefficient (0 = perfect equality, 1 = total inequality)
      Based on: https://github.com/oliviaguest/gini
      """
      sorted_x = np.sort(x)
      n = len(x)
      cumsum = np.cumsum(sorted_x)
      return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

  Output: toi_gini_YYYY.csv files (one per season)

  ---
  Step 3: Player-Level Statistics

  Script: scripts/creating_final_all_season_df.py

  Aggregation

  For each player in each game, calculate:

  1. Shots on Goal (SOG)
    - Count from play-by-play where eventType = "shot-on-goal"
    - Exclude shootout shots (period > 4)
  2. Assists
    - Count from play-by-play where player listed in assist1Id or assist2Id
  3. Corsi-For (CF)
    - SOG + missed shots + blocked shots (at even strength only)
    - eventType in ["shot-on-goal", "missed-shot", "blocked-shot"]
    - Strength = "even"
  4. Expected Goals (xG)
    - Join with Moneypuck data on game_id + player_id
    - Sum xG for all shots by this player
  5. Time-on-Ice (TOI)
    - From shift data (Step 2)
    - Total seconds on ice

  Output

  all_rosters_20102025.csv with columns:
  - game_id, player_id, teamAbbrev
  - sog, assists, corsi_for, toi_seconds, xgoal

  ---
  Step 4: Team-Game Gini Coefficients

  Script: scripts/creating_final_all_season_df.py (continued)

  Process

  For each team in each game:

  1. Extract all rostered players (typically 18 skaters)
  2. Get distribution of: SOG, assists, Corsi-for, xG, TOI
  3. Calculate Gini coefficient for each metric
  4. Store as {metric}_gini

  Example

  Game 2010020009, Anaheim Ducks:
  - 18 rostered skaters
  - 21 total shots on goal
  - Distribution: [4, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
  - Gini = 0.664 (highly concentrated)
  - Shot Depth = 1 - 0.664 = 0.336 (low depth)

  ---
  Step 5: Depth Metrics

  Script: scripts/creating_final_all_season_df.py (continued)

  Inversion

  Depth = 1 - Gini

  This inverts the metric so:
  - Higher depth = more balanced roster (good)
  - Lower depth = concentrated in few players (bad)

  Standardization

  All depth metrics standardized to z-scores:
  df['sog_depth_z'] = (df['sog_depth_raw'] - mean) / sd

  Mean = 0, SD = 1 across all 19,067 observations.

  ---
  Step 6: Latent Variable Modeling

  Script: analysis/tdi_complete_analysis.R

  SEM Model

  model <- '
    depth =~ cf_depth_z + sog_depth_z + toi_depth_z + xgoal_depth_z
  '

  fit <- sem(model, data = data, std.lv = TRUE)

  Factor Scores

  Extract latent "depth" score for each team-game:
  data$tdi_factor <- lavPredict(fit)[, "depth"]

  Output: tdi_factor variable (Total Depth Index)

  ---
  Step 7: Rolling Averages

  Script: analysis/tdi_complete_analysis.R (continued)

  10-Game Rolling Mean

  Within each team-season:
  data <- data %>%
    group_by(teamAbbrev, season) %>%
    mutate(
      depth_rolling10 = slide_dbl(
        tdi_factor,
        mean,
        .before = 9,
        .complete = TRUE
      )
    )

  Result: Smoothed depth measure for predictive modeling

  ---
  Final Dataset

  File: data/final_data_20102025.csv

  Rows: 19,067 team-games
  Columns: 22 variables

  Key variables:
  - Identifiers (game_id, teamAbbrev, season)
  - Outcome (win/loss)
  - Volume metrics (total_sogs, xgoal, corsi_for)
  - Gini coefficients (5 metrics)
  - Raw depth (5 metrics)
  - Standardized depth (4 metrics)
  - Latent depth (tdi_factor, depth_rolling10)

  ---
  Data Quality Checks

  1. Missing data: < 0.1% (only first 10 games per team-season for rolling avg)
  2. Duplicate games: None (verified via game_id uniqueness)
  3. Shootout exclusion: Verified (no period > 4 shots counted)
  4. API failures: Retry logic with exponential backoff (max 3 retries)

  ---
  Computational Requirements

  - Storage: ~7 GB for raw data, 12 MB for final dataset
  - RAM: 8 GB minimum (for shift data streaming)
  - Runtime:
    - Data collection: ~12 hours
    - Processing: ~2 hours
    - Analysis: ~15 minutes

  ---
  Reproducibility Notes

  1. API rate limiting: 2-second delays ensure reproducibility without getting blocked
  2. Random seed: Not applicable (deterministic calculations)
  3. Package versions: See requirements.txt and R script header
  4. Data snapshots: Final dataset frozen as of January 11, 2026

  ---
  Contact

  Questions about preprocessing: dwiwad@gmail.com
