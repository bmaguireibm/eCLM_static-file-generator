from __future__ import annotations

import netCDF4
import numpy as np
import pytest

from point_crop_surface_generator.point_topography import _face_overlaps
from point_crop_surface_generator.s3 import S3Config, parse_s3_uri
from point_crop_surface_generator.surface import (
    _apply_cloud_canopy,
    _apply_cloud_crop_fertilizer,
    _normalize_bounds,
    _prescribe_crop,
    _read_monthly_lai,
    _spherical_area,
    _write_domain,
)


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        ("1,2,3,4,5,6,7,8,9,10,11,12\n", np.arange(1, 13)),
        (
            "month,lai\n"
            + "\n".join(f"{month},{month / 2}" for month in range(12, 0, -1)),
            np.arange(1, 13) / 2,
        ),
        ("lai\n" + "\n".join(str(month) for month in range(1, 13)), np.arange(1, 13)),
    ],
)
def test_read_monthly_lai_accepts_supported_csv_shapes(tmp_path, contents, expected):
    path = tmp_path / "lai.csv"
    path.write_text(contents, encoding="utf-8")

    np.testing.assert_array_equal(_read_monthly_lai(path), expected)


@pytest.mark.parametrize(
    "contents",
    [
        "1,2,3\n",
        "month,lai\n" + "\n".join(f"{month},{month}" for month in range(1, 12)),
        "month,lai\n"
        + "\n".join(f"{month},{month}" for month in range(1, 12))
        + "\n12,-1\n",
        "month,lai\n"
        + "\n".join(f"{month},{month}" for month in range(1, 12))
        + "\n12,nan\n",
    ],
)
def test_read_monthly_lai_rejects_invalid_csv(tmp_path, contents):
    path = tmp_path / "lai.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        _read_monthly_lai(path)


def test_normalize_bounds_handles_longitudes_in_360_degree_convention():
    assert _normalize_bounds(48.5, 48.51, 349.0, 350.0) == (
        48.5,
        48.51,
        -11.0,
        -10.0,
    )


def test_spherical_area_is_positive_for_valid_cell():
    assert _spherical_area((48.5, 48.51, 7.5, 7.51)) > 0


def test_write_domain_uses_cell_bounds(tmp_path):
    path = tmp_path / "domain.nc"
    bounds = (48.5, 48.51, 7.5, 7.51)

    _write_domain(path, bounds)

    with netCDF4.Dataset(path) as domain:
        assert domain.variables["xc"][0, 0] == pytest.approx(7.505)
        assert domain.variables["yc"][0, 0] == pytest.approx(48.505)
        assert domain.variables["area"][0, 0] == pytest.approx(_spherical_area(bounds))
        np.testing.assert_array_equal(domain.variables["mask"][:], [[1]])
        np.testing.assert_array_equal(domain.variables["frac"][:], [[1.0]])


def _create_crop_template(path):
    with netCDF4.Dataset(path, "w") as dataset:
        for name, size in (
            ("lsmlat", 1),
            ("lsmlon", 1),
            ("cft", 2),
            ("natpft", 2),
            ("glc_mec", 10),
            ("lsmpft", 78),
            ("time", 12),
        ):
            dataset.createDimension(name, size)
        dataset.createVariable("cft", "i4", ("cft",))[:] = [15, 21]
        dataset.createVariable("natpft", "i4", ("natpft",))[:] = [13, 14]

        for name in (
            "PCT_CROP",
            "PCT_NATVEG",
            "PCT_LAKE",
            "PCT_WETLAND",
            "PCT_GLACIER",
            "PCT_URBAN",
        ):
            dataset.createVariable(name, "f8", ("lsmlat", "lsmlon"))[:] = 0
        dataset.createVariable("PCT_GLC_MEC", "f8", ("glc_mec", "lsmlat", "lsmlon"))[
            :, 0, 0
        ] = [100, *([0] * 9)]
        dataset.createVariable("PCT_NAT_PFT", "f8", ("natpft", "lsmlat", "lsmlon"))[
            :
        ] = 0
        dataset.createVariable("PCT_CFT", "f8", ("cft", "lsmlat", "lsmlon"))[:] = 0
        dataset.createVariable(
            "CONST_FERTNITRO_CFT", "f8", ("cft", "lsmlat", "lsmlon")
        )[:, 0, 0] = [7, 11]
        for name in ("LANDFRAC_PFT", "PFTDATA_MASK"):
            dataset.createVariable(name, "f8", ("lsmpft", "lsmlat", "lsmlon"))[:] = 0
        dataset.createVariable(
            "MONTHLY_LAI", "f8", ("time", "lsmpft", "lsmlat", "lsmlon")
        )[:] = -1


def test_prescribe_crop_sets_active_cft_and_preserves_inactive_fertilizer(tmp_path):
    path = tmp_path / "surfdata.nc"
    _create_crop_template(path)

    _prescribe_crop(path, crop_type=21, monthly_lai=None, fertilizer_rate=30)

    with netCDF4.Dataset(path) as dataset:
        assert dataset.variables["PCT_CROP"][0, 0] == 100
        np.testing.assert_array_equal(dataset.variables["PCT_CFT"][:, 0, 0], [0, 100])
        np.testing.assert_array_equal(dataset.variables["PCT_NATVEG"][:], [[0]])
        np.testing.assert_array_equal(
            dataset.variables["PCT_NAT_PFT"][:, 0, 0], [100, 0]
        )
        np.testing.assert_array_equal(
            dataset.variables["CONST_FERTNITRO_CFT"][:, 0, 0], [7, 30]
        )
        np.testing.assert_array_equal(
            dataset.variables["PCT_GLC_MEC"][:, 0, 0], [100, *([0] * 9)]
        )
        assert np.all(dataset.variables["LANDFRAC_PFT"][:] == 1)
        assert np.all(dataset.variables["PFTDATA_MASK"][:] == 1)


def test_prescribe_crop_writes_monthly_lai_to_global_cft_row(tmp_path):
    path = tmp_path / "surfdata.nc"
    monthly_lai = np.arange(1, 13, dtype=np.float64)
    _create_crop_template(path)

    _prescribe_crop(path, crop_type=21, monthly_lai=monthly_lai, fertilizer_rate=None)

    with netCDF4.Dataset(path) as dataset:
        np.testing.assert_array_equal(
            dataset.variables["MONTHLY_LAI"][:, 21, 0, 0], monthly_lai
        )
        assert np.all(dataset.variables["MONTHLY_LAI"][:, :21, 0, 0] == -1)


def test_cloud_fertilizer_updates_only_the_selected_cft(tmp_path):
    path = tmp_path / "surfdata.nc"
    _create_crop_template(path)

    _apply_cloud_crop_fertilizer(path, crop_type=21, source_values=[4, 8])

    with netCDF4.Dataset(path) as dataset:
        np.testing.assert_array_equal(
            dataset.variables["CONST_FERTNITRO_CFT"][:, 0, 0], [7, 8]
        )


def test_cloud_canopy_maps_source_rows_and_irrigated_crop(tmp_path):
    path = tmp_path / "surfdata.nc"
    source = np.arange(12 * 16, dtype=np.float64).reshape(12, 16)
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 12)
        dataset.createDimension("lsmpft", 17)
        dataset.createDimension("lsmlat", 1)
        dataset.createDimension("lsmlon", 1)
        for name in (
            "MONTHLY_LAI",
            "MONTHLY_SAI",
            "MONTHLY_HEIGHT_TOP",
            "MONTHLY_HEIGHT_BOT",
        ):
            dataset.createVariable(name, "f8", ("time", "lsmpft", "lsmlat", "lsmlon"))[
                :
            ] = -1

    fields = {
        name: source
        for name in (
            "MONTHLY_LAI",
            "MONTHLY_SAI",
            "MONTHLY_HEIGHT_TOP",
            "MONTHLY_HEIGHT_BOT",
        )
    }
    _apply_cloud_canopy(path, fields)

    with netCDF4.Dataset(path) as dataset:
        np.testing.assert_array_equal(
            dataset.variables["MONTHLY_LAI"][:, :16, 0, 0], source
        )
        np.testing.assert_array_equal(
            dataset.variables["MONTHLY_LAI"][:, 16, 0, 0], source[:, 15]
        )


def test_topography_face_overlap_weights_cover_cell():
    overlap_rows, weights = _face_overlaps(
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([[0, 1, 2, 3]]),
        (0.0, 1.0, 0.0, 1.0),
    )

    np.testing.assert_array_equal(overlap_rows, [0])
    np.testing.assert_allclose(weights, [1.0])


def test_s3_uri_parser_and_environment_aliases(monkeypatch):
    monkeypatch.setenv("AWS_S3_ENDPOINT", "test-s3-endpoint")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "region-1")

    config = S3Config.from_env()

    assert config.endpoint_url == "test-s3-endpoint"
    assert config.access_key_id == "access"
    assert config.secret_access_key == "secret"
    assert config.region_name == "region-1"
    assert parse_s3_uri("s3://bucket/prefix/data.zarr") == (
        "bucket",
        "prefix/data.zarr",
    )
    with pytest.raises(ValueError, match="expected an s3:// URI"):
        parse_s3_uri("https://bucket/path")
