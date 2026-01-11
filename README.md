# Total Depth Index (TDI) in the NHL: Peer Review Repository

 This repository contains all data, code, and documentation to reproduce the analyses presented in:

 **"Latent Variable Modeling for Intangible Constructs in Sports: The Case of Team Depth in the National Hockey League"**

 Dylan Wiwad, PhD  
 Hockey Decoded  
 Email: dwiwad@gmail.com  
 ---

## Repository Contents

```
.
  ├── README.md                          # This file
  ├── PREPROCESSING.md                   # Complete data pipeline documentation
  ├── CODEBOOK.md                        # Variable definitions
  ├── CITATION.cff                       # Citation metadata
  ├── LICENSE                            # MIT License
  ├── requirements.txt                   # Python dependencies
  ├── data/
  │   ├── final_data_20102025.csv       # Main analysis dataset (19,067 games)
  │   ├── all_games_meta_20102025.csv   # Game metadata from NHL API
  │   ├── all_rosters_20102025.csv      # Player roster data per game
  │   └── samples/                       # Sample raw data
  │       ├── sample_pbp_1000games.csv
  │       ├── sample_shifts_100games.ndjson
  │       └── moneypuck_shots_2024.csv
  ├── scripts/
  │   ├── data_wrangling_all_seasons.py  # Fetch data from NHL API
  │   ├── all_season_shift_depth.py      # Calculate TOI depth
  │   └── creating_final_all_season_df.py # Create final dataset
  └── analysis/
      └── tdi_complete_analysis.R         # Complete R analysis script
```
  ---

  ## Quick Start

  ### 1. Install Dependencies

  **R packages:**
  ```r
  install.packages(c("lavaan", "lavaanPlot", "lme4", "lmerTest", "broom.mixed",
                     "dplyr", "tidyr", "ggplot2", "patchwork", "slider",
                     "psych", "semTools", "yardstick", "ggeffects", "scales"))

  Python packages:
  pip install -r requirements.txt

  2. Run Analysis

  setwd("analysis/")
  source("tdi_complete_analysis.R")

  This reproduces all tables and figures from the paper (~15 min runtime).

  ---
  Data Sources

  Primary Sources

  1. NHL API (api-web.nhle.com, api.nhle.com)
    - Game schedules and metadata
    - Play-by-play data
    - Roster information
    - Shift charts (time-on-ice)
  2. Moneypuck (moneypuck.com)
    - Expected Goals (xG) data for all seasons 2010-2025

  Coverage

  - Seasons: 2010-11 through 2024-25 (15 seasons)
  - Games: 19,067 team-games
  - Players: ~8,000 unique skaters
  - Plays: ~15 million events

  ---
  Key Variables

  - Depth Indicators: Shot depth, xG depth, Corsi depth, TOI depth
  - Latent Measure: tdi_factor (Total Depth Index from SEM)
  - Rolling Average: depth_rolling10 (10-game rolling mean)
  - Outcome: Binary win/loss

  See CODEBOOK.md for complete variable definitions.

  ---
  Key Findings

  1. Depth is measurable: Four indicators load onto a single latent factor (CFI=0.99, RMSEA=0.04)
  2. Depth is unstable: Only 1.9% of variance between teams; lag-1 autocorrelation r=0.013
  3. Rolling averages stabilize: 10-game average predicts next-game depth (β=0.38, p<.001)
  4. Depth predicts winning: LOSO-CV shows improvement over baseline in all 15 seasons
  5. Mechanism is quality: Depth increases xG, not just shot volume (suppression effect)

  ---
  File Sizes

  Some raw files are sampled due to size:

  | File                        | Size   | Status                       |
  |-----------------------------|--------|------------------------------|
  | all_pbp_20102025.csv        | 817 MB | Sample provided (1000 games) |
  | all_shifts_20102025.ndjson  | 5.8 GB | Sample provided (100 games)  |
  | all_rosters_20102025.csv    | 94 MB  | ✓ Full file included         |
  | all_games_meta_20102025.csv | 18 MB  | ✓ Full file included         |
  | final_data_20102025.csv     | 12 MB  | ✓ Full file included         |

  Full raw files available upon request.

  ---
  Citation

  @article{wiwad2025tdi,
    title={Latent Variable Modeling for Intangible Constructs in Sports:
           The Case of Team Depth in the National Hockey League},
    author={Wiwad, Dylan},
    journal={Under Review},
    year={2025}
  }

  ---
  License

  MIT License - see LICENSE file.

  Data from NHL API and Moneypuck subject to their respective terms of use.

  ---
  Contact

  Dylan Wiwad, PhD
  Email: dwiwad@gmail.com
  Website: https://hockeydecoded.com

  ---
  Last Updated: January 2026
