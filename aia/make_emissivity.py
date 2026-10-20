"""
Make ``static/aiapy/aia_V9_emissivity.fits`` from SolarSoft V9 files.

Run by hand, not by the workflow, with numpy, astropy and sunpy installed::

    python aia/make_emissivity.py $SSW/sdo/aia/response
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits  # type: ignore[import-not-found]
from sunpy.io.special import read_genx  # type: ignore[import-not-found]

OUTPUT = Path(__file__).parent / "static" / "aiapy" / "aia_V9_emissivity.fits"


def _provenance(header: fits.Header, key: str, path: Path, genx: dict[str, dict[str, str]]) -> None:
    with path.open("rb") as f:
        sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    header[f"{key}FILE"] = path.name
    header[f"{key}DATE"] = (genx["HEADER"]["CREATION"], "genx creation date")
    header[f"{key}SHA"] = sha256


def make_emissivity_file(response_dir: Path, save_path: Path) -> None:
    """
    Save the V9 summed emissivity, CHIANTI settings and chiantifix as FITS.

    Parameters
    ----------
    response_dir : Path
        The SolarSoft ``sdo/aia/response`` directory.
    save_path : Path
        Full file path for the output FITS file.
    """
    fullemiss_path = response_dir / "aia_V9_fullemiss.genx"
    fix_path = response_dir / "aia_V9_chiantifix.genx"
    fullemiss = read_genx(fullemiss_path)
    fix = read_genx(fix_path)
    total, general = fullemiss["TOTAL"], fullemiss["GENERAL"]
    # The genx values are float32; store the nominal 0.1 A and 0.05 log T steps and pressure
    wavelength = np.round(total["WAVE"].astype(np.float64), 1)
    logte = np.round(total["LOGTE"].astype(np.float64), 2)
    np.testing.assert_allclose(wavelength, total["WAVE"], rtol=0, atol=1e-4)
    np.testing.assert_allclose(logte, total["LOGTE"], rtol=0, atol=1e-5)
    np.testing.assert_array_equal(fix["LOGTE"], total["LOGTE"])

    primary = fits.PrimaryHDU()
    header = primary.header
    header["CHIANTI"] = (general["VERSION"], "CHIANTI version")
    header["ABUNDANC"] = Path(general["ABUNDFILE"]).name
    header["IONEQ"] = Path(general["IONEQ_NAME"]).name
    header["MODEL"] = general["MODEL_NAME"]
    header["PRESSURE"] = (float(f"{general['MODEL_PE']:.7g}"), "[K cm-3]")
    header["PROTONS"] = (general["ADD_PROTONS"], "proton rates added")
    _provenance(header, "EMIS", fullemiss_path, fullemiss)
    _provenance(header, "FIX", fix_path, fix)
    for line in fullemiss["FIXES"]:
        header.add_history(line.strip())
    for line in fullemiss["NOTES"]:
        header.add_comment(line.strip())

    emissivity = fits.CompImageHDU(
        total["EMISSIVITY"].astype(np.float32), name="EMISSIVITY", compression_type="GZIP_2", quantize_level=0.0
    )
    emissivity.header["BUNIT"] = "ph cm3 s-1 sr-1 Angstrom-1"
    wavelength_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="wavelength", format="D", unit="Angstrom", array=wavelength)], name="WAVELENGTH"
    )
    temperature_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="logte", format="D", array=logte)], name="TEMPERATURE"
    )
    # Keyed like aiapy's Channel.name ("94"), not the genx "A94"
    chiantifix_hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="channel", format="4A", array=[c.removeprefix("A") for c in fix["CHANNELS"]]),
            fits.Column(
                name="chiantifix",
                format=f"{logte.size}D",
                unit="DN cm5 s-1 pix-1",
                array=fix["EMPIRICAL_MINUS_RAW"],
            ),
        ],
        name="CHIANTIFIX",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, emissivity, wavelength_hdu, temperature_hdu, chiantifix_hdu]).writeto(
        save_path, overwrite=True
    )
    with fits.open(save_path) as hdul:
        np.testing.assert_array_equal(hdul["EMISSIVITY"].data, total["EMISSIVITY"].astype(np.float32))


if __name__ == "__main__":
    make_emissivity_file(Path(sys.argv[1]), OUTPUT)
