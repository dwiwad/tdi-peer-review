# Codebook: Total Depth Index Dataset

  **Dataset:** `final_data_20102025.csv`
  **Observations:** 19,067 team-games
  **Period:** 2010-11 through 2024-25 NHL seasons

  ---

  ## Identifiers

  | Variable | Type | Description |
  |----------|------|-------------|
  | `game_id` | integer | NHL game ID (format: YYYY0DNNNN where YYYY=season end year, D=game type: 2=regular season, 3=playoffs) |
  | `teamAbbrev` | string | Three-letter team abbreviation (e.g., "TOR", "BOS", "EDM") |
  | `season` | integer | Season identifier (YYYYYYYY format, e.g., 20102011 for 2010-11 season) |

  ---

  ## Outcome Variable

  | Variable | Type | Description | Values |
  |----------|------|-------------|--------|
  | `outcome` | integer | Game outcome for this team | 0 = loss, 1 = win |

  ---

  ## Volume Metrics

  | Variable | Type | Description |
  |----------|------|-------------|
  | `total_sogs` | float | Total shots on goal by this team in this game |
  | `xgoal` | float | Total expected goals (from Moneypuck) |
  | `corsi_for` | float | Total Corsi-for (shots + missed shots + blocked shots at even strength) |

  ---

  ## Inequality Metrics (Gini Coefficients)

  Gini coefficient measures concentration/inequality (0 = perfect equality, 1 = total concentration in one player).

  | Variable | Type | Description |
  |----------|------|-------------|
  | `sog_gini` | float | Gini coefficient for shots on goal distribution |
  | `xgoal_gini` | float | Gini coefficient for expected goals distribution |
  | `cf_gini` | float | Gini coefficient for Corsi-for distribution |
  | `assist_gini` | float | Gini coefficient for assists distribution |
  | `toi_gini` | float | Gini coefficient for time-on-ice distribution |

  ---

  ## Depth Metrics (Raw)

  Depth = 1 - Gini (inverted inequality; higher = more balanced roster).

  | Variable | Type | Description | Range |
  |----------|------|-------------|-------|
  | `sog_depth_raw` | float | Shot depth (1 - sog_gini) | 0-1 |
  | `xgoal_depth_raw` | float | Expected goals depth (1 - xgoal_gini) | 0-1 |
  | `cf_depth_raw` | float | Corsi-for depth (1 - cf_gini) | 0-1 |
  | `assist_depth_raw` | float | Assists depth (1 - assist_gini) | 0-1 |
  | `toi_depth_raw` | float | Time-on-ice depth (1 - toi_gini) | 0-1 |

  ---

  ## Standardized Variables (Z-scores)

  All standardized to mean=0, SD=1 across the full dataset.

  | Variable | Type | Description |
  |----------|------|-------------|
  | `sog_depth_z` | float | Standardized shot depth |
  | `xgoal_depth_z` | float | Standardized xG depth |
  | `cf_depth_z` | float | Standardized Corsi depth |
  | `toi_depth_z` | float | Standardized TOI depth |
  | `sogs_z` | float | Standardized total shots |
  | `xgoal_z` | float | Standardized total xG |
  | `corsi_for_z` | float | Standardized total Corsi-for |

  ---

  ## Latent Depth Variable

  | Variable | Type | Description |
  |----------|------|-------------|
  | `tdi_factor` | float | **Total Depth Index** - Latent factor score from SEM (standardized, mean≈0, SD≈1) |
  | `depth_rolling10` | float | 10-game rolling average of `tdi_factor` (within team-season) |

  **Note:** `tdi_factor` is derived from a confirmatory factor analysis (CFA) where the four depth indicators (shot, xG, Corsi, TOI depth) load onto a single latent construct.

  ---

  ## Missing Data

  - `depth_rolling10`: Missing (NA) for first 10 games of each team-season
  - All other variables: Complete (no missing data)

  ---

  ## Data Processing Notes

  1. **Gini Calculation:** Applied to player-level distributions within each team-game
  2. **Depth Inversion:** All depth metrics are 1 - Gini, so higher values = more balanced
  3. **Z-score Standardization:** Applied across all 19,067 observations
  4. **Rolling Averages:** Calculated within team-season groups, ordered by game_id
  5. **Shootout Shots:** Excluded from shot counts (only regulation + overtime)

  ---

  ## Example Row
```
  game_id: 2010020009
  teamAbbrev: ANA
  season: 20102011
  outcome: 0
  total_sogs: 21.0
  xgoal: 2.668
  corsi_for: 40.0
  sog_gini: 0.664
  sog_depth_raw: 0.336
  sog_depth_z: -2.499
  tdi_factor: -2.148
  depth_rolling10: NA
```
  This represents the Anaheim Ducks' loss in game 2010020009, where they had low depth (concentrated shots among few players).

  ---

  ## Questions?

  Contact: dwiwad@gmail.com
