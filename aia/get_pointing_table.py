"""
Fetch the AIA master pointing table from JSOC and save it as a CSV.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import drms  # type: ignore[import-untyped]
import pandas as pd

WAVELENGTHS = ("094", "131", "171", "193", "211", "304", "335", "1600", "1700", "4500")
NEEDED_KEYS = ["T_START", "T_STOP"] + [
    f"A_{wl}_{suffix}" for wl in WAVELENGTHS for suffix in ("X0", "Y0", "IMSCALE", "INSTROT")
]
SERIES = "aia.master_pointing3h"


def _build_time_ranges(
    start_date: pd.Timestamp, end_date: pd.Timestamp, months_per_chunk: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges = []
    cur = start_date
    while cur < end_date:
        nxt = min(cur + pd.DateOffset(months=months_per_chunk), end_date)
        ranges.append((cur, nxt))
        cur = nxt
    return ranges


def _query_range(time_range: tuple[pd.Timestamp, pd.Timestamp]) -> pd.DataFrame:
    start, end = time_range
    rec = f"{SERIES}[{start.strftime('%Y-%m-%dT%H:%M:%S')}Z-{end.strftime('%Y-%m-%dT%H:%M:%S')}Z]"
    return cast("pd.DataFrame", drms.Client().query(rec, key=NEEDED_KEYS))


def get_and_save_pointing_table(
    save_path: Path,
    months_per_chunk: int = 12,
    workers: int = 4,
) -> None:
    """
    Get and save the AIA pointing table to CSV quickly and politely.

    Parameters
    ----------
    save_path : Path
        Full file path for the output CSV.
    months_per_chunk : int, optional
        Size of each time chunk in months (default 12).
    workers : int, optional
        Number of parallel workers (default 4).
        Keep small (<= 4) to be kind to JSOC.
    """
    start_date = pd.Timestamp("2010-05-13T00:00:00Z")
    end_date = pd.Timestamp.now(tz="UTC")
    time_ranges = _build_time_ranges(start_date, end_date, months_per_chunk)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        df = pd.concat(ex.map(_query_range, time_ranges))
    # DRMS [t1-t2] ranges include both ends, so each chunk boundary comes back twice
    df = df.drop_duplicates("T_START").sort_values("T_START")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    get_and_save_pointing_table(Path(os.environ["OUTPUT_DIR"]) / "aia_pointing_table.csv")
