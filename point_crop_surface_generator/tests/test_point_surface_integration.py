"""Opt-in cloud integration check for the one-cell Python static generator.

Run with ECLM_RUN_CLOUD_INTEGRATION=1 and the cloud S3 environment configured.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import netCDF4
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SURFDATA_TEMPLATE = REPO_ROOT / (
    "input_clm/surfdata_1x1_DE-RuS_hist_78pfts_CMIP6_simyr2005_c240813_WWrainfed.nc"
)
DOMAIN_TEMPLATE = REPO_ROOT / "input_clm/domain.lnd.DE-RuS.240717.nc"
EXAMPLE_BOUNDS = (50.860906, 50.870906, 6.4421445, 6.4521445)
SECOND_AOI_BOUNDS = (48.500, 48.510, 7.500, 7.510)
EXCLUDED_SURFDATA_VARIABLES = {"SLOPE", "STD_ELEV"}
CLOUD_SURFDATA_VARIABLES = {
    "PCT_SAND",
    "PCT_CLAY",
    "ORGANIC",
    "zbedrock",
    "peatf",
    "FMAX",
    "SOIL_COLOR",
    "MONTHLY_LAI",
    "MONTHLY_SAI",
    "MONTHLY_HEIGHT_TOP",
    "MONTHLY_HEIGHT_BOT",
    "SLOPE",
    "STD_ELEV",
    "EF1_BTR",
    "EF1_FET",
    "EF1_FDT",
    "EF1_SHR",
    "EF1_GRS",
    "EF1_CRP",
    "gdp",
    "abm",
    "CONST_HARVEST_VH1",
    "CONST_HARVEST_VH2",
    "CONST_HARVEST_SH1",
    "CONST_HARVEST_SH2",
    "CONST_HARVEST_SH3",
    "CONST_GRAZING",
    "CONST_FERTNITRO_CFT",
    "UNREPRESENTED_PFT_LULCC",
    "UNREPRESENTED_CFT_LULCC",
}


def _compare_netcdf_values(
    actual_path: Path,
    expected_path: Path,
    *,
    excluded_variables: set[str] | None = None,
) -> None:
    excluded = excluded_variables or set()
    mismatches: list[str] = []
    with (
        netCDF4.Dataset(actual_path) as actual,
        netCDF4.Dataset(expected_path) as expected,
    ):
        assert set(actual.dimensions) == set(expected.dimensions)
        for name, dimension in expected.dimensions.items():
            assert len(actual.dimensions[name]) == len(dimension), name
            assert actual.dimensions[name].isunlimited() == dimension.isunlimited(), (
                name
            )

        assert set(actual.variables) == set(expected.variables)
        for name, expected_variable in expected.variables.items():
            if name in excluded:
                continue
            actual_variable = actual.variables[name]
            assert actual_variable.dimensions == expected_variable.dimensions, name
            expected_values = np.ma.asarray(expected_variable[...])
            actual_values = np.ma.asarray(actual_variable[...])
            assert actual_values.shape == expected_values.shape, name
            np.testing.assert_array_equal(
                np.ma.getmaskarray(actual_values),
                np.ma.getmaskarray(expected_values),
                err_msg=f"{name} mask differs",
            )

            expected_values = expected_values.compressed()
            actual_values = actual_values.compressed()
            try:
                if expected_variable.dtype.kind in "fc":
                    np.testing.assert_allclose(
                        actual_values,
                        expected_values,
                        rtol=5e-5,
                        atol=1e-8,
                        err_msg=f"{name} differs",
                    )
                else:
                    np.testing.assert_array_equal(
                        actual_values,
                        expected_values,
                        err_msg=f"{name} differs",
                    )
            except AssertionError as exc:
                detail = str(exc).splitlines()
                mismatches.append(f"{name}: {detail[0] if detail else 'values differ'}")
    assert not mismatches, "NetCDF value mismatches:\n" + "\n".join(mismatches)


def _run_generator(
    temporary_path: Path,
    bounds: tuple[float, float, float, float],
    *,
    fertilizer_rate: float | None = None,
) -> tuple[Path, Path]:
    generated_surfdata = temporary_path / "surfdata.nc"
    generated_domain = temporary_path / "domain.nc"
    # Existing destinations must be replaced by the default CLI behavior.
    generated_surfdata.write_bytes(b"old surfdata")
    generated_domain.write_bytes(b"old domain")

    command = [
        sys.executable,
        "-m",
        "point_crop_surface_generator",
        "--surfdata-template",
        str(SURFDATA_TEMPLATE),
        "--south",
        str(bounds[0]),
        "--north",
        str(bounds[1]),
        "--west",
        str(bounds[2]),
        "--east",
        str(bounds[3]),
        "--crop-type",
        "21",
        "--output-surfdata",
        str(generated_surfdata),
        "--output-domain",
        str(generated_domain),
    ]
    if fertilizer_rate is not None:
        command.extend(["--fertilizer-rate", str(fertilizer_rate)])
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        str(PROJECT_ROOT / "src"),
                        os.environ.get("PYTHONPATH"),
                    ),
                )
            ),
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return generated_surfdata, generated_domain


@pytest.mark.skipif(
    os.environ.get("ECLM_RUN_CLOUD_INTEGRATION") != "1",
    reason="set ECLM_RUN_CLOUD_INTEGRATION=1 to read the cloud Zarr/GeoParquet stores",
)
def test_example_aoi_generator_matches_all_values_except_topography():
    """Run the CLI at DE-RuS and compare every data variable with the example."""
    assert SURFDATA_TEMPLATE.is_file(), (
        f"missing surfdata template: {SURFDATA_TEMPLATE}"
    )
    assert DOMAIN_TEMPLATE.is_file(), f"missing domain example: {DOMAIN_TEMPLATE}"

    with tempfile.TemporaryDirectory(prefix="eclm-point-surface-") as temporary:
        temporary_path = Path(temporary)
        generated_surfdata, generated_domain = _run_generator(
            temporary_path,
            EXAMPLE_BOUNDS,
            # The checked-in example has a manual 30 gN/m2/yr crop rate.
            fertilizer_rate=30,
        )

        _compare_netcdf_values(
            generated_surfdata,
            SURFDATA_TEMPLATE,
            excluded_variables=EXCLUDED_SURFDATA_VARIABLES,
        )
        _compare_netcdf_values(generated_domain, DOMAIN_TEMPLATE)


@pytest.mark.skipif(
    os.environ.get("ECLM_RUN_CLOUD_INTEGRATION") != "1",
    reason="set ECLM_RUN_CLOUD_INTEGRATION=1 to read the cloud Zarr/GeoParquet stores",
)
def test_second_aoi_generates_complete_files_without_site_data():
    """Confirm a new AOI needs bounds and crop type, not prepared local data."""
    with tempfile.TemporaryDirectory(
        prefix="eclm-point-surface-second-aoi-"
    ) as temporary:
        generated_surfdata, generated_domain = _run_generator(
            Path(temporary), SECOND_AOI_BOUNDS
        )

        center_lat = sum(SECOND_AOI_BOUNDS[:2]) / 2.0
        center_lon = sum(SECOND_AOI_BOUNDS[2:]) / 2.0
        with netCDF4.Dataset(generated_surfdata) as surfdata:
            assert CLOUD_SURFDATA_VARIABLES <= set(surfdata.variables)
            np.testing.assert_allclose(surfdata.variables["LATIXY"][:], center_lat)
            np.testing.assert_allclose(surfdata.variables["LONGXY"][:], center_lon)
            for name in CLOUD_SURFDATA_VARIABLES:
                values = np.ma.asarray(surfdata.variables[name][:]).compressed()
                assert values.size, f"{name} has no unmasked output values"
                assert np.all(np.isfinite(values)), f"{name} is non-finite"
            np.testing.assert_allclose(surfdata.variables["PCT_CROP"][:], 100.0)
            cft_indices = np.asarray(surfdata.variables["cft"][:], dtype=np.int64)
            crop_row = int(np.flatnonzero(cft_indices == 21)[0])
            crop_fractions = np.asarray(surfdata.variables["PCT_CFT"][:])
            assert crop_fractions[crop_row, 0, 0] == pytest.approx(100.0)

        with netCDF4.Dataset(generated_domain) as domain:
            np.testing.assert_allclose(domain.variables["yc"][:], center_lat)
            np.testing.assert_allclose(domain.variables["xc"][:], center_lon)
            assert np.all(np.asarray(domain.variables["mask"][:]) == 1)
            assert np.all(np.asarray(domain.variables["frac"][:]) == 1.0)
