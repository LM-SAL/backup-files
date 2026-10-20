"""
Make ``static/aiapy/aia_chianti11_emissivity.fits`` from CHIANTI with fiasco.

The line and continuum emissivity of every ion, summed, on the grids of
SolarSoft's ``aia_V9_fullemiss.genx`` (1 to 901 angstroms in steps of 0.1,
log T 4 to 9 in steps of 0.05) and with its plasma: the coronal abundances of
Feldman (1992), every element at least as abundant as zinc, the ionization
equilibrium of CHIANTI, a constant pressure of 1e15 K cm^-3, proton rates,
and both members of the He II 304 angstrom doublet boosted by 20 as
``aia_bp_make_emiss`` intends (its V9 file boosted only one, see ``BOOST``).
The emission measure is that of n_e n_H.

Each line is shared between the two grid wavelengths around it, the nearer
one getting the larger share, rather than moved to the nearest one as
SolarSoft does, so folding the file with an effective area interpolated
linearly gives the response at the line's exact wavelength. There is nothing
like ``chiantifix``.

Adapted from ``tools/aia_temperature_response_chianti.py`` of
sun-data/solar-dynamics-observatory (BSD 3-Clause, Roy T. Smart and Jacob D.
Parker), which folds each ion into the channels instead of keeping the
spectrum. As there, each ion is computed in a process of its own and saved in
``RUNDIR`` as it finishes, so an interrupted run resumes and only failed ions
are computed again. The saved ions are keyed by a digest of this script, the
fiasco version and the database, so a run after any of them changes starts
over. Most of the time goes to solving for level populations.

Run by hand, not by the workflow, with fiasco and its CHIANTI database (the
one it reads by default, or ``--database``), numpy, astropy, h5py and psutil
installed::

    python aia/make_chianti_emissivity.py RUNDIR [--workers N] [--memory GB]
        [--database H5] [--output FITS] [--ions "Fe 9,He 2"]
"""

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import hashlib
import logging
import multiprocessing
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import astropy.constants  # type: ignore[import-not-found]
import astropy.units as u  # type: ignore[import-not-found]
import fiasco  # type: ignore[import-not-found]
import fiasco.util.exceptions  # type: ignore[import-not-found]
import numpy as np
from astropy.io import fits  # type: ignore[import-not-found]

OUTPUT = Path(__file__).parent / "static" / "aiapy" / "aia_chianti11_emissivity.fits"

WAVELENGTH = np.round(np.arange(1, 901 + 1e-9, 0.1), 1)
STEP = 0.1
LOGTE = np.round(np.arange(4, 9 + 1e-9, 0.05), 2)
PRESSURE = 1e15 * u.K / u.cm**3
ABUNDANCE = "sun_coronal_1992_feldman_ext"
ABUNDANCE_MIN = 1e-7  # zinc, the least abundant element in the V9 emissivity, is just above
IONIZATION_FRACTION = "chianti"
# aia_bp_make_emiss boosts both members of the He II 304 doublet by 20 for its optical depth
# ("to agree with V&R"). It matches them by exact wavelength, 303.78101 and 303.78601, so the
# V6 and V8 files (CHIANTI 7.1.3) boosted both but the V9 file (CHIANTI 9.0.1, which lists the
# strong 2p 2P3/2 member at 303.780) boosted only the weak one. This follows the intent. Keyed
# by fiasco's level labels, made from CHIANTI's energy levels; CHIANTI 11's line labels are
# wrong for these levels.
BOOST = {("He 2", "1s 2S1/2", "2p 2P1/2"): 20, ("He 2", "1s 2S1/2", "2p 2P3/2"): 20}
WORKER_GIGABYTES = 0.25  # what a process takes before it computes an ion
HC = (astropy.constants.h * astropy.constants.c).to_value(u.erg * u.AA)
UNIT = u.erg * u.cm**3 / u.s / u.AA
MissingData = fiasco.util.exceptions.MissingDatasetException
Result = dict[str, np.ndarray]
Task = tuple[str, bool, float]
Database = tuple[str, str, str]  # CHIANTI version, the fiasco that built the file, its sha256
_state: dict[str, Any] = {}
log = logging.getLogger("make_chianti_emissivity")


def _initialize(proton_electron_ratio: np.ndarray, database: Path) -> None:
    warnings.simplefilter("ignore")
    fiasco.log.setLevel("ERROR")
    _state.update(proton_electron_ratio=proton_electron_ratio, database=database)


def _level(ion: Any, label: str) -> int:  # ruff: ignore[any-type]
    levels = ion.levels
    where = levels.label == label
    if where.sum() != 1:
        msg = f"{ion.ion_name} has {where.sum()} levels {label!r}"
        raise ValueError(msg)
    return int(levels.level[where][0])


def _lines(ion: Any, density: Any) -> tuple[np.ndarray, int]:  # ruff: ignore[any-type]
    # The lines of an ion on the grid, photons per steradian, shape (logte, wavelength), and their number.
    # fiasco walks the whole database for the proton-electron ratio, so compute() hands it to the workers
    ion.__dict__["proton_electron_ratio"] = _state["proton_electron_ratio"]
    g = ion.contribution_function(density, couple_density_to_temperature=True)[:, 0]
    g = g.to_value(u.erg * u.cm**3 / u.s)
    transitions = ion.transitions
    bound = transitions.is_bound_bound
    wavelength = transitions.wavelength[bound].to_value(u.AA)
    factor = np.ones(wavelength.shape)
    for (name, lower, upper), f in BOOST.items():
        if name == ion.ion_name:
            where = (transitions.lower_level[bound] == _level(ion, lower)) & (
                transitions.upper_level[bound] == _level(ion, upper)
            )
            if where.sum() != 1:
                msg = f"{name} has {where.sum()} lines {lower}-{upper}"
                raise ValueError(msg)
            factor[where] = f
    where = (wavelength >= WAVELENGTH[0]) & (wavelength < WAVELENGTH[-1])
    wavelength = wavelength[where]
    photons = g[:, where] * factor[where] * wavelength / HC / (4 * np.pi) / STEP
    # shared linearly between the grid wavelengths either side
    i = np.floor((wavelength - WAVELENGTH[0]) / STEP).astype(int)
    f = np.clip((wavelength - WAVELENGTH[i]) / STEP, 0, 1)
    result = np.empty((LOGTE.size, WAVELENGTH.size))
    for t in range(LOGTE.size):
        result[t] = np.bincount(i, photons[t] * (1 - f), WAVELENGTH.size) + np.bincount(
            i + 1, photons[t] * f, WAVELENGTH.size
        )
    return result, int(where.sum())


def _two_photon(ion: Any, wavelength: Any, density: Any) -> Any:  # ruff: ignore[any-type]
    # The two-photon continuum at one density per temperature, shape (logte, wavelength).
    # With couple_density_to_temperature=True, fiasco 0.8.2 keeps the density axis of the level
    # populations, (T, 1, 1), and divides by a density of shape (T, 1), which fails
    # (wtbarnes/fiasco#475). Handing it the populations without that axis gives the shape the
    # rest of the method expects. Any other call is an error, so a change in fiasco stops this
    # rather than giving wrong populations.
    expected = {"couple_density_to_temperature": True, "include_protons": False}
    populations = ion.level_populations(density, **expected)

    def level_populations(electron_density: Any, **kwargs: object) -> Any:  # ruff: ignore[any-type]
        if kwargs != expected or not np.array_equal(electron_density, density):
            msg = f"fiasco's two_photon no longer asks for these level populations: {kwargs=}"
            raise RuntimeError(msg)
        return populations[:, 0]

    ion.level_populations = level_populations
    try:
        return ion.two_photon(wavelength, density, couple_density_to_temperature=True)[:, 0]
    finally:
        del ion.level_populations


def ion_emissivity(name: str, lines: bool) -> Result:  # ruff: ignore[boolean-type-hint-positional-argument]
    """
    Compute one ion's emissivity per unit emission measure.

    Parameters
    ----------
    name : str
        The ion, as fiasco names it, like ``"Fe 9"``.
    lines : bool
        Whether the ion has lines in range; if not, only its continua are computed.

    Returns
    -------
    dict
        ``lines`` and ``continuum`` in ph cm3 s-1 sr-1 A-1, shape (logte, wavelength); ``n_lines``;
        ``missing``, the kinds of emission the database has no data for; ``seconds``.

    Raises
    ------
    ValueError
        If any value is not finite and >= 0, so that nothing bad is saved and reused.
    """
    start = time.perf_counter()
    temperature = 10**LOGTE * u.K
    density = PRESSURE / temperature
    ion = fiasco.Ion(
        name,
        temperature,
        abundance=ABUNDANCE,
        ionization_fraction=IONIZATION_FRACTION,
        hdf5_dbase_root=_state["database"],
    )
    shape = (LOGTE.size, WAVELENGTH.size)
    result: Result = {"lines": np.zeros(shape), "continuum": np.zeros(shape)}
    missing = []
    n_lines = 0
    if lines:
        try:
            result["lines"], n_lines = _lines(ion, density)
        except MissingData:
            missing.append("lines")
    wavelength = WAVELENGTH * u.AA
    photons = WAVELENGTH / HC  # erg -> photons
    continuum = np.zeros(shape)
    if ion.ionization_stage > 1:
        try:
            # per steradian in fiasco 0.8.2, unlike the lines and the other continua
            # (wtbarnes/fiasco#474), which _check_free_free makes sure of
            ff = ion.free_free(wavelength) * ion.abundance * ion.ionization_fraction[:, np.newaxis]
            continuum += ff.to_value(UNIT) * photons
        except MissingData:
            missing.append("free_free")
    try:
        fb = ion.free_bound(wavelength) * ion.abundance * ion.next_ion().ionization_fraction[:, np.newaxis]
        continuum += fb.to_value(UNIT) * photons / (4 * np.pi)
    except MissingData:
        missing.append("free_bound")
    if ion.hydrogenic or ion.helium_like:
        try:
            tp = _two_photon(ion, wavelength, density) * ion.abundance * ion.ionization_fraction[:, np.newaxis]
            continuum += tp.to_value(UNIT) * photons / (4 * np.pi)
        except MissingData:
            missing.append("two_photon")
    result["continuum"] = continuum
    for component, value in result.items():
        if not np.all(np.isfinite(value) & (value >= 0)):
            msg = f"the {component} of {name} are not finite and >= 0"
            raise ValueError(msg)
    result["n_lines"] = np.array(n_lines)
    result["missing"] = np.array(missing, dtype=str)
    result["seconds"] = np.array(time.perf_counter() - start)
    return result


def _gigabytes(ion: Any) -> float:  # ruff: ignore[any-type]
    # The memory solving an ion's level populations takes, estimated: about eight (logte, level,
    # level) arrays, over the levels of this ion and the next when the two-ion model applies.
    n = ion.n_levels
    if ion._has_dataset("auto") or ion._has_dataset("rrlvl"):  # ruff: ignore[private-member-access]
        with contextlib.suppress(MissingData):
            n += ion.next_ion().n_levels
    return float(8 * LOGTE.size * n**2 * 8 / 1e9)


def _ions(database: Path, only: set[str] | None) -> list[Task]:
    # (name, has lines in range, gigabytes) of every ion abundant enough, the largest first
    result = []
    names = [str(name) for name in fiasco.list_ions(hdf5_dbase_root=database)]
    if only and (unknown := only - set(names)):
        msg = f"not in the database: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    for name in names:
        if only and name not in only:
            continue
        ion = fiasco.Ion(name, 1 * u.MK, abundance=ABUNDANCE, hdf5_dbase_root=database)
        try:
            if ion.abundance < ABUNDANCE_MIN:
                continue
        except MissingData:
            continue
        lines = False
        with contextlib.suppress(MissingData):
            wavelength = ion.transitions.wavelength[ion.transitions.is_bound_bound].to_value(u.AA)
            lines = bool(np.any((wavelength >= WAVELENGTH[0]) & (wavelength < WAVELENGTH[-1])))
        result.append((_gigabytes(ion) if lines else 0.0, name, lines))
    result.sort(reverse=True)
    return [(name, lines, gigabytes) for gigabytes, name, lines in result]


def _check_free_free(database: Path) -> None:
    # fiasco's free-free continuum is per steradian: its radiative loss is 4 pi times its integral
    ion = fiasco.Ion("H 2", [1e6, 1e7, 1e8] * u.K, hdf5_dbase_root=database)
    wavelength = np.geomspace(1e-3, 1e7, 4001) * u.AA
    integral = np.trapezoid(ion.free_free(wavelength), wavelength, axis=-1)
    ratio = (ion.free_free_radiative_loss() / integral).to_value(u.dimensionless_unscaled)
    if not np.allclose(ratio, 4 * np.pi, rtol=0.1):
        msg = f"fiasco's free-free continuum is no longer per steradian (loss/integral = {ratio})"
        raise RuntimeError(msg)


def _database(database: Path) -> Database:
    import h5py  # type: ignore[import-not-found]  # ruff: ignore[import-outside-top-level]

    with h5py.File(database, "r") as f:
        version = str(f["fe/fe_9/wgfa"].attrs["chianti_version"])
        built_with = str(f.attrs.get("fiasco_version", ""))
    with database.open("rb") as f:  # a few seconds for the 2 GB of CHIANTI 11
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    return version, built_with, sha256


def _save(file: Path, result: Result) -> None:
    # written in full before it takes its name, so a run stopped while writing leaves nothing that looks done
    partial = file.with_name(f"{file.name}.part")
    with partial.open("wb") as f:
        np.savez_compressed(
            f,
            lines=result["lines"].astype(np.float32),
            continuum=result["continuum"].astype(np.float32),
            n_lines=result["n_lines"],
            missing=result["missing"],
            seconds=result["seconds"],
        )
    partial.replace(file)


def _run(
    pool: concurrent.futures.ProcessPoolExecutor, todo: list[Task], done: Path, workers: int, memory: float
) -> tuple[dict[str, Result], dict[str, BaseException]]:
    # Submit ions as the memory they need frees up, since the largest together need more than a
    # computer has; save each as it finishes; return the results and the ones that failed.
    results: dict[str, Result] = {}
    failed: dict[str, BaseException] = {}
    running: dict[concurrent.futures.Future[Result], Task] = {}
    start = time.perf_counter()
    k = 0
    while todo or running:
        used = sum(gigabytes for _, _, gigabytes in running.values())
        for task in list(todo):
            name, lines, gigabytes = task
            if len(running) >= workers:
                break
            if running and used + gigabytes > memory:
                continue
            running[pool.submit(ion_emissivity, name, lines)] = task
            used += gigabytes
            todo.remove(task)
        finished, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
        for future in finished:
            name, _, _ = running.pop(future)
            k += 1
            progress = f"{k}/{k + len(todo) + len(running)} {name}"
            elapsed = time.perf_counter() - start
            try:
                result = future.result()
            except Exception as e:
                failed[name] = e
                log.exception("%s: failed, %.0f s elapsed", progress, elapsed)
                continue
            file = done / f"{name.replace(' ', '_')}.npz"
            _save(file, result)
            results[name] = dict(np.load(file))
            log.info("%s: %.0f s, %d lines, %.0f s elapsed", progress, result["seconds"], result["n_lines"], elapsed)
    return results, failed


def compute(
    rundir: Path, workers: int, memory: float, database: Path, only: set[str] | None = None
) -> tuple[Database, dict[str, Result]]:
    """
    Compute every ion into ``rundir``, reusing those already there.

    Parameters
    ----------
    rundir : Path
        Where each ion is saved as it finishes, under a directory named for the CHIANTI version
        and a digest of this script, the fiasco version and the database.
    workers : int
        Processes to run at once.
    memory : float
        Gigabytes the processes may take together.
    database : Path
        The HDF5 database of fiasco.
    only : set of str, optional
        Ion names to compute instead of all of them, for testing.

    Returns
    -------
    tuple, dict
        The database's CHIANTI version, the fiasco that built it and its sha256, and
        ``{name: result}`` in a fixed order.

    Raises
    ------
    ValueError
        If ``memory`` leaves nothing for the ions, or ``only`` names an ion not in the database.
    RuntimeError
        If any ion failed; the rest are saved, so a run again computes only those.
    """
    warnings.simplefilter("ignore")
    fiasco.log.setLevel("ERROR")
    info = _database(database)
    digest = hashlib.sha256(Path(__file__).read_bytes() + f"{fiasco.__version__} {info[2]}".encode()).hexdigest()
    memory_ions = memory - workers * WORKER_GIGABYTES
    if memory_ions <= 0:
        msg = f"{workers} workers take {workers * WORKER_GIGABYTES} GB before computing anything"
        raise ValueError(msg)
    temperature = 10**LOGTE * u.K
    ratio = fiasco.proton_electron_ratio(
        temperature, abundance=ABUNDANCE, ionization_fraction=IONIZATION_FRACTION, hdf5_dbase_root=database
    )
    _check_free_free(database)
    # the parallelism is over ions; a BLAS thread per core in every process slows it to a crawl
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    done = rundir / f"chianti_{info[0]}_{digest[:12]}"
    done.mkdir(parents=True, exist_ok=True)
    tasks = _ions(database, only)
    results: dict[str, Result] = {}
    todo = []
    for task in tasks:
        file = done / f"{task[0].replace(' ', '_')}.npz"
        if file.exists():
            results[task[0]] = dict(np.load(file))
        else:
            todo.append(task)
    log.info("CHIANTI %s: %d ions, %d to compute", info[0], len(tasks), len(todo))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize,
        initargs=(ratio.to_value(u.dimensionless_unscaled), database),
    ) as pool:
        computed, failed = _run(pool, todo, done, workers, memory_ions)
    results.update(computed)
    if failed:
        msg = f"{len(failed)} of {len(tasks)} ions failed: {', '.join(failed)}; run again"
        raise RuntimeError(msg) from next(iter(failed.values()))
    return info, {name: results[name] for name, _, _ in tasks}


def write(info: Database, results: dict[str, Result], database: Path, save_path: Path) -> None:
    """
    Sum the ions and write the FITS file.

    The layout is that of ``aia_V9_emissivity.fits``, without its CHIANTIFIX extension.

    Parameters
    ----------
    info : tuple
        The database's CHIANTI version, the fiasco that built it and its sha256, from `compute`.
    results : dict
        ``{name: result}`` from `compute`.
    database : Path
        The HDF5 database of fiasco, for its name and list of ions.
    save_path : Path
        Full file path for the output FITS file.
    """
    version, built_with, sha256 = info
    total = np.zeros((LOGTE.size, WAVELENGTH.size))
    for result in results.values():
        total += result["lines"].astype(float) + result["continuum"].astype(float)
    primary = fits.PrimaryHDU()
    header = primary.header
    header["CHIANTI"] = (version, "CHIANTI version")
    header["ABUNDANC"] = f"{ABUNDANCE}.abund"
    header["IONEQ"] = f"{IONIZATION_FRACTION}.ioneq"
    header["MODEL"] = ("Constant pressure", "fiasco, density coupled to temperature")
    header["PRESSURE"] = (PRESSURE.to_value(u.K / u.cm**3), "[K cm-3]")
    header["PROTONS"] = (True, "proton rates added")
    header["ABUNDMIN"] = (ABUNDANCE_MIN, "elements less abundant than this left out")
    header["FIASCO"] = (fiasco.__version__, "fiasco version")
    header["DBASE"] = (database.name, "fiasco database")
    header["DBFIASCO"] = (built_with, "fiasco that built the database")
    header["DBSHA"] = sha256
    header["NIONS"] = (len(results), "ions summed")
    header["NLINES"] = (sum(int(r["n_lines"]) for r in results.values()), "lines in 1-901 Angstrom")
    header["DATE"] = (dt.datetime.now(tz=dt.UTC).date().isoformat(), "generated")
    header["GENERATE"] = ("aia/make_chianti_emissivity.py", "generator, LM-SAL/backup-files")
    for (name, lower, upper), factor in BOOST.items():
        header.add_history(f"{name} {lower} - {upper} multiplied by {factor}")
    header.add_history("each line shared linearly between the two grid wavelengths around it")
    names = [str(name) for name in fiasco.list_ions(hdf5_dbase_root=database)]
    for component in ("lines", "free_free", "free_bound", "two_photon"):
        where = [n for n in names if n in results and component in results[n]["missing"]]
        if where:
            header.add_comment(f"no {component} data for {len(where)} ions:")
            for i in range(0, len(where), 9):  # a card holds 72 characters
                header.add_comment(", ".join(where[i : i + 9]))
    emissivity = fits.CompImageHDU(
        total.astype(np.float32), name="EMISSIVITY", compression_type="GZIP_2", quantize_level=0.0
    )
    emissivity.header["BUNIT"] = "ph cm3 s-1 sr-1 Angstrom-1"
    wavelength_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="wavelength", format="D", unit="Angstrom", array=WAVELENGTH)], name="WAVELENGTH"
    )
    temperature_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="logte", format="D", array=LOGTE)], name="TEMPERATURE"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, emissivity, wavelength_hdu, temperature_hdu]).writeto(save_path, overwrite=True)
    with fits.open(save_path) as hdul:
        np.testing.assert_array_equal(hdul["EMISSIVITY"].data, total.astype(np.float32))
    log.info("wrote %s", save_path)


def _memory() -> float:
    import psutil  # type: ignore[import-untyped]  # ruff: ignore[import-outside-top-level]

    return float(0.75 * psutil.virtual_memory().available / 1e9)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("rundir", type=Path, help="where each ion is saved as it finishes")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--memory", type=float, default=_memory(), help="GB the workers may take together")
    parser.add_argument("--database", type=Path, default=Path(fiasco.defaults["hdf5_dbase_root"]))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ions", help="comma-separated ion names to compute instead of all, for testing")
    args = parser.parse_args()
    only = {s.strip() for s in args.ions.split(",")} if args.ions else None
    if args.output.resolve() == OUTPUT.resolve():
        if only:
            sys.exit("--ions writes a partial file; give it an --output")
        if (version := _database(args.database)[0]).split(".")[0] != "11":
            sys.exit(f"{args.database} is CHIANTI {version}; write it elsewhere with --output")
    info, results = compute(args.rundir, args.workers, args.memory, args.database, only)
    write(info, results, args.database, args.output)
