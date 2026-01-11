import json
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import s3fs
from app.core.config import S3_BUCKET

# ----------------------------
# Config
# ----------------------------
PREFIX = "static-ds-analyses/total-depth-index/all-seasons"
SHIFT_OBJECT = "all_shifts_20102025.ndjson"
ROSTER_CSV = "~/dev/hockey_site/data/total-depth-index/all_seasons/all_rosters_20102025.csv"

fs = s3fs.S3FileSystem(anon=False)
shift_path = f"s3://{S3_BUCKET}/{PREFIX}/{SHIFT_OBJECT}"

# ----------------------------
# Helpers
# ----------------------------
def parse_duration_seconds(mmss: str) -> int:
    """
    Parse a MM:SS string to total seconds. Returns 0 on malformed.
    """
    if not isinstance(mmss, str) or ":" not in mmss:
        return 0
    try:
        mm, ss = mmss.strip().split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return 0

def gini(x, eps=1e-9):
    """
    Gini coefficient for a 1D array-like of nonnegative values.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return np.nan
    amin = a.min()
    if amin < 0:
        a = a - amin
    s = a.sum()
    if s <= 0:
        return 0.0
    a = np.sort(a)
    n = a.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return ((2 * idx - n - 1) @ a) / (n * s + eps)

# ----------------------------
# Main loop: 2010 → 2024
# ----------------------------
for year in range(2010, 2025):
    year_str = str(year)
    print(f"\n==== Processing season {year_str} ====")

    # 1) Stream the giant NDJSON once per year and aggregate TOI per (game_id, playerId)
    toi_seconds = defaultdict(int)  # key: (game_id, playerId) -> total seconds
    n_lines = 0
    n_kept = 0

    with fs.open(shift_path, "r") as infile:
        for line in infile:
            n_lines += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            game_id = str(rec.get("game_id", ""))  # store as string for stable prefix checks/joins
            if not game_id.startswith(year_str):
                continue

            # Keep only pure "shift" rows (detailCode == 0)
            if rec.get("detailCode", None) != 0:
                continue

            pid = rec.get("playerId", None)
            if pid is None:
                continue

            dur = rec.get("duration", None)  # "MM:SS"
            sec = parse_duration_seconds(dur)
            if sec <= 0:
                continue

            toi_seconds[(game_id, int(pid))] += sec
            n_kept += 1

    print(f"Read {n_lines:,} lines; kept {n_kept:,} shift rows for {year_str}.")

    # 2) Convert aggregated dict to a small DataFrame
    if not toi_seconds:
        print(f"No shift rows found for {year_str}; skipping write.")
        continue

    toi_rows = (
        {"game_id": k[0], "playerId": k[1], "total_ice_time": v}
        for k, v in toi_seconds.items()
    )
    toi_sums_df = pd.DataFrame(toi_rows)

    # 3) Load roster slice for this year
    #    Keep only columns we need to reduce memory.
    usecols = ["game_id", "playerId", "positionCode", "teamAbbrev"]
    rosters = pd.read_csv(ROSTER_CSV, usecols=usecols, dtype={"game_id": "string"}, low_memory=False)
    mask = rosters["game_id"].astype("string").str.startswith(year_str, na=False)
    rost_year = rosters.loc[mask].copy()
    # Normalize dtypes for merge keys
    rost_year["game_id"] = rost_year["game_id"].astype(str)
    # Some playerId may be floats if CSV has missing—coerce safely
    rost_year["playerId"] = pd.to_numeric(rost_year["playerId"], errors="coerce").astype("Int64")

    # 4) Merge roster with TOI sums, fill missing with 0, drop goalies
    toi_sums_df["playerId"] = pd.to_numeric(toi_sums_df["playerId"], errors="coerce").astype("Int64")
    merged = pd.merge(
        rost_year,
        toi_sums_df,
        on=["game_id", "playerId"],
        how="left",
    )
    merged["total_ice_time"] = merged["total_ice_time"].fillna(0).astype(int)

    shooters = merged[merged["positionCode"] != "G"].copy()

    # 5) Compute Gini per (game_id, teamAbbrev)
    toi_gini = (
        shooters.groupby(["game_id", "teamAbbrev"], as_index=False)["total_ice_time"]
        .apply(gini)
        .rename(columns={"total_ice_time": "toi_gini"})
    )

    # 6) Write to S3
    out_object = f"toi_gini_{year_str}.csv"
    out_path = f"s3://{S3_BUCKET}/{PREFIX}/{out_object}"
    toi_gini.to_csv(out_path, index=False, storage_options={"anon": False})
    print(f"Wrote {len(toi_gini):,} rows to {out_path}")

    # 7) Free memory before the next year
    del toi_seconds, toi_sums_df, rosters, rost_year, merged, shooters, toi_gini
    gc.collect()

print("\nAll seasons complete.")
