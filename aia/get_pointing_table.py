import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import drms
import pandas as pd
from pytz import utc

WAVELENGTHS = ("094", "171", "193", "211", "304", "335", "1600", "1700", "4500")
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


def _query_range(start: pd.Timestamp, end: pd.Timestamp, keys: list[str]) -> pd.DataFrame:
    client = drms.Client()
    rec = f"{SERIES}[{start.strftime('%Y-%m-%dT%H:%M:%S')}Z-{end.strftime('%Y-%m-%dT%H:%M:%S')}Z]"
    df = client.query(rec, key=keys)
    if df.empty:
        msg = f"No data returned for time range {start} to {end}"
        raise ValueError(msg)
    return df.sort_values("T_START")


def get_and_save_pointing_table(
    save_path: Path,
    months_per_chunk: int = 12,
    workers: int = 1,
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
        Number of parallel workers. 1 = sequential (default).
        Keep small (<= 4) to be kind to JSOC.
    """
    start_date = pd.Timestamp("2010-05-13T00:00:00Z")
    end_date = pd.Timestamp.now(tz=utc)
    time_ranges = _build_time_ranges(start_date, end_date, months_per_chunk)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    try:
        if workers <= 1:
            client = drms.Client()
            with save_path.open("w", newline="") as f:
                for start, end in time_ranges:
                    rec = f"{SERIES}[{start.strftime('%Y-%m-%dT%H:%M:%S')}Z-{end.strftime('%Y-%m-%dT%H:%M:%S')}Z]"
                    df = client.query(rec, key=NEEDED_KEYS)
                    if df.empty:
                        msg = f"No data returned for time range {start} to {end}"
                        raise ValueError(msg)  # NOQA: TRY301
                    df = df.sort_values("T_START")
                    df.to_csv(f, index=False, header=not wrote_header)
                    wrote_header = True
        else:
            results: list[pd.DataFrame | None] = [None] * len(time_ranges)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                fut_to_idx = {
                    ex.submit(_query_range, start, end, NEEDED_KEYS): i for i, (start, end) in enumerate(time_ranges)
                }
                for fut in as_completed(fut_to_idx):
                    i = fut_to_idx[fut]
                    results[i] = fut.result()
            with save_path.open("w", newline="") as f:
                for df in results:
                    if df is None or df.empty:
                        continue
                    df.to_csv(f, index=False, header=not wrote_header)
                    wrote_header = True
    except Exception as e:
        msg = f"Unable to create the JSOC table.\nError message: {e}"
        raise OSError(msg) from e


if __name__ == "__main__":
    out_dir = Path(os.environ["OUTPUT_DIR"])
    get_and_save_pointing_table(out_dir / "aia_pointing_table.csv", months_per_chunk=12, workers=4)
