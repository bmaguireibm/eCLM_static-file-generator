from __future__ import annotations

import json
import subprocess
from pathlib import Path

import netCDF4
import numpy as np
import pyarrow.parquet as parquet
import pytest
import zarr

from point_crop_surface_generator.download import (
    DEFAULT_GRID_FILES,
    _remote_size,
    download_sources,
)
from point_crop_surface_generator.point_soil import _axis_overlaps
from point_crop_surface_generator.prepare import (
    _convert_hydro1k,
    _convert_netcdf_group,
    _convert_scrip_grid,
    _validate_lai_dimensions,
    prepare_stores,
)


def _create_raw_netcdf(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lat", 2)
        dataset.createDimension("lon", 3)
        dataset.createDimension("levsoi", 2)
        lat = dataset.createVariable("LATIXY", "f8", ("lat", "lon"))
        lon = dataset.createVariable("LONGXY", "f8", ("lat", "lon"))
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        lat[:] = [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]]
        lon[:] = [[6.0, 7.0, 8.0], [6.0, 7.0, 8.0]]
        field = dataset.createVariable("SOIL", "f4", ("levsoi", "lat", "lon"))
        field.units = "kg m-3"
        field[:] = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        station_name = dataset.createVariable("station_name", str, ("lat",))
        station_name[0] = "north"
        station_name[1] = "south"
        dataset.title = "raw test field"


def _create_lai_netcdf(path: Path, *, lat_size: int = 1, lon_size: int = 3) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("lat", lat_size)
        dataset.createDimension("lon", lon_size)
        lat = dataset.createVariable("LAT", "f8", ("lat",))
        lon = dataset.createVariable("LON", "f8", ("lon",))
        lat.units = "degrees_north"
        lon.units = "degrees_east"
        lat[:] = np.linspace(50.0, 51.0, lat_size)
        lon[:] = np.linspace(6.0, 8.0, lon_size)
        dataset.createVariable("MONTHLY_LAI", "f4", ("lat", "lon"))[:] = 1.0


def _create_scrip_netcdf(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("grid_rank", 2)
        dataset.createDimension("grid_size", 3)
        dataset.createDimension("grid_corners", 4)
        dataset.createVariable("grid_dims", "i4", ("grid_rank",))[:] = [3, 1]
        dataset.createVariable("grid_center_lon", "f8", ("grid_size",))[:] = [
            359.5,
            0.5,
            180.0,
        ]
        dataset.createVariable("grid_center_lat", "f8", ("grid_size",))[:] = [
            0.0,
            1.0,
            2.0,
        ]
        corner_lon = dataset.createVariable(
            "grid_corner_lon", "f8", ("grid_size", "grid_corners")
        )
        corner_lat = dataset.createVariable(
            "grid_corner_lat", "f8", ("grid_size", "grid_corners")
        )
        corner_lon[:] = [[359, 360, 360, 359], [0, 1, 1, 0], [179, 181, 181, 179]]
        corner_lat[:] = [
            [-0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, 1.5, 1.5],
            [1.5, 1.5, 2.5, 2.5],
        ]
        dataset.createVariable("grid_imask", "i4", ("grid_size",))[:] = [1, 1, 1]
        dataset.createVariable("grid_area", "f8", ("grid_size",))[:] = [1.0, 1.0, 1.0]


def _create_hydro_netcdf(path: Path) -> None:
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("nface", 2)
        dataset.createDimension("nnode", 6)
        dataset.createDimension("max_face_nodes", 4)
        for name, values in (
            ("landmesh_face_x", [359.5, 0.5]),
            ("landmesh_face_y", [0.5, 0.5]),
        ):
            variable = dataset.createVariable(name, "f8", ("nface",))
            variable.units = "degrees"
            variable[:] = values
        for name, values in (
            ("landmesh_node_x", [359.0, 360.0, 360.0, 359.0, 1.0, 1.0]),
            ("landmesh_node_y", [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ):
            variable = dataset.createVariable(name, "f8", ("nnode",))
            variable.units = "degrees"
            variable[:] = values
        connectivity = dataset.createVariable(
            "landmesh_face_node",
            "i4",
            ("nface", "max_face_nodes"),
            fill_value=0,
        )
        connectivity[:] = [[1, 2, 3, 4], [2, 5, 6, 3]]
        dataset.createVariable("landmesh_face_area", "f8", ("nface",))[:] = [1.0, 1.0]
        dataset.createVariable("landmesh_face_mask", "i4", ("nface",))[:] = [1, 1]


def test_convert_raw_field_records_spatial_metadata_and_values(tmp_path: Path):
    source = tmp_path / "raw.nc"
    output = tmp_path / "stores"
    _create_raw_netcdf(source)

    _convert_netcdf_group(
        source,
        str(output),
        relative_path="raw/test.nc.zarr",
        face_chunk_size=2,
        spatial_raw=True,
    )

    group = zarr.open_group(str(output / "raw/test.nc.zarr"), mode="r")
    assert group.attrs["dimensions"] == {"lat": 2, "lon": 3, "levsoi": 2}
    assert group.attrs["spatial"] == {
        "latitude_variable": "LATIXY",
        "longitude_variable": "LONGXY",
        "row_dimension": "lat",
        "column_dimension": "lon",
        "curvilinear": True,
    }
    assert group["SOIL"].attrs["dimensions"] == ["levsoi", "lat", "lon"]
    np.testing.assert_array_equal(group["SOIL"][:], np.arange(12).reshape(2, 2, 3))
    assert "station_name" not in group
    assert (
        group.attrs["variables"]["station_name"]["skipped"]
        == "variable-length or object dtype"
    )
    assert group.attrs["source_file"] == "raw.nc"
    index = group["spatial_index"]
    assert index["row_start"][:].tolist() == [0]
    assert index["column_stop"][:].tolist() == [3]
    assert index["lat_min"][:].tolist() == [50.0]


def test_longitude_overlap_wraps_cloud_coordinates_from_zero_to_360(tmp_path: Path):
    group = zarr.open_group(str(tmp_path / "longitude.zarr"), mode="w")
    try:
        group.create_dataset("longitude", data=[308.5, 309.5, 310.5, 311.5])
    except AttributeError:  # zarr 3 renamed this method
        group.create_array("longitude", data=[308.5, 309.5, 310.5, 311.5])

    overlaps = _axis_overlaps(
        group,
        "longitude",
        dimension_size=4,
        chunk_start=1,
        chunk_stop=3,
        lower=-51.0,
        upper=-49.0,
        latitude=False,
    )

    assert set(overlaps) == {1, 2}


def test_convert_scrip_grid_writes_cloud_chunk_index(tmp_path: Path):
    source = tmp_path / "scrip.nc"
    output = tmp_path / "stores"
    _create_scrip_netcdf(source)

    _convert_scrip_grid(
        source,
        str(output),
        relative_path="grids/scrip.nc.zarr",
        face_chunk_size=2,
    )

    group = zarr.open_group(str(output / "grids/scrip.nc.zarr"), mode="r")
    np.testing.assert_array_equal(group["grid_center_lon"][:], [359.5, 0.5, 180.0])
    index = group["spatial_index"]
    assert index["chunk_start"][:].tolist() == [0, 2]
    assert index["chunk_stop"][:].tolist() == [2, 3]
    assert index["lon_width"][0] < 2.1
    assert index.attrs["source"] == "grid corner bounds"
    assert group.attrs["kind"] == "SCRIP"


def test_prepare_lai_profile_builds_compatible_raw_and_scrip_stores(tmp_path: Path):
    input_root = tmp_path / "inputdata"
    raw_relative = (
        "lnd/clm2/rawdata/pftcftdynharv.0.05x0.05.LUH2.histsimyr2005.c190116/"
        "mksrf_lai_histclm52deg005_earthstatmirca_2005.c190119.nc"
    )
    raw_file = input_root / raw_relative
    raw_file.parent.mkdir(parents=True)
    _create_lai_netcdf(raw_file)

    grid_file = (
        input_root / "lnd/clm2/mappingdata/grids/SCRIPgrid_3minx3min_MODISv2_c190503.nc"
    )
    grid_file.parent.mkdir(parents=True)
    _create_scrip_netcdf(grid_file)

    raw_manifest = tmp_path / "raw.txt"
    raw_manifest.write_text(f"{raw_relative}\n", encoding="utf-8")
    grid_manifest = tmp_path / "grids.txt"
    grid_manifest.write_text(
        "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/"
        "mappingdata/grids/SCRIPgrid_3minx3min_MODISv2_c190503.nc\n",
        encoding="utf-8",
    )
    output = tmp_path / "cloud-optimized"

    prepare_stores(
        input_root,
        str(output),
        raw_manifest=raw_manifest,
        grid_manifest=grid_manifest,
        face_chunk_size=2,
        lai_only=True,
    )

    assert (output / f"raw/{raw_relative}.zarr/.zgroup").is_file()
    assert (
        output / "grids/SCRIPgrid_3minx3min_MODISv2_c190503.nc.zarr/.zgroup"
    ).is_file()
    assert not (output / "grids/hydro1k.zarr").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [result["kind"] for result in manifest["results"]] == [
        "raw-field",
        "SCRIP",
    ]


def test_validate_lai_dimensions_rejects_an_aoi_cropped_field(tmp_path: Path):
    lai = tmp_path / "lai.nc"
    grid = tmp_path / "grid.nc"
    _create_lai_netcdf(lai, lat_size=2)
    _create_scrip_netcdf(grid)

    with pytest.raises(ValueError, match="AoI crop"):
        _validate_lai_dimensions(lai, grid)


def test_lai_only_cannot_skip_its_raw_field(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_stores(
            tmp_path,
            str(tmp_path / "out"),
            skip_raw=True,
            lai_only=True,
        )


def test_convert_hydro1k_writes_zero_based_mesh_and_geoparquet(tmp_path: Path):
    source = tmp_path / "hydro.nc"
    output = tmp_path / "stores"
    _create_hydro_netcdf(source)

    _convert_hydro1k(source, str(output), face_chunk_size=1, tile_size=1.0)

    mesh = zarr.open_group(str(output / "grids/hydro1k.zarr"), mode="r")
    assert mesh.attrs["face_count"] == 2
    np.testing.assert_array_equal(mesh["face_node"][:], [[0, 1, 2, 3], [1, 4, 5, 2]])
    np.testing.assert_array_equal(mesh["face_lon"][:], [-0.5, 0.5])

    parquet_files = list((output / "grids/hydro1k-face-index").glob("**/*.parquet"))
    assert len(parquet_files) == 2
    table = parquet.read_table(parquet_files[0])
    assert {"face_id", "min_lon", "max_lon", "min_lat", "max_lat", "geometry"} <= set(
        table.column_names
    )
    geo = json.loads(table.schema.metadata[b"geo"])
    assert geo["primary_column"] == "geometry"
    assert geo["columns"]["geometry"]["encoding"] == "WKB"


def test_convert_hydro1k_keeps_zero_based_connectivity_across_chunks(tmp_path: Path):
    source = tmp_path / "hydro-zero-based.nc"
    with netCDF4.Dataset(source, "w") as dataset:
        dataset.createDimension("nface", 2)
        dataset.createDimension("nnode", 5)
        dataset.createDimension("max_face_nodes", 4)
        dataset.createVariable("landmesh_face_x", "f8", ("nface",))[:] = [
            0.5,
            1.5,
        ]
        dataset.createVariable("landmesh_face_y", "f8", ("nface",))[:] = [
            0.5,
            0.5,
        ]
        dataset.createVariable("landmesh_node_x", "f8", ("nnode",))[:] = [
            0.0,
            1.0,
            1.0,
            0.0,
            2.0,
        ]
        dataset.createVariable("landmesh_node_y", "f8", ("nnode",))[:] = [
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
        ]
        dataset.createVariable("landmesh_face_node", "i4", ("nface", "max_face_nodes"))[
            :
        ] = [[0, 1, 2, 3], [1, 2, 4, 3]]

    output = tmp_path / "stores"
    _convert_hydro1k(source, str(output), face_chunk_size=1, tile_size=1.0)

    mesh = zarr.open_group(str(output / "grids/hydro1k.zarr"), mode="r")
    np.testing.assert_array_equal(mesh["face_node"][:], [[0, 1, 2, 3], [1, 2, 4, 3]])
    parquet_files = list((output / "grids/hydro1k-face-index").glob("**/*.parquet"))
    assert sum(parquet.read_table(path).num_rows for path in parquet_files) == 2


def test_convert_hydro1k_skips_incomplete_faces_in_geoparquet(tmp_path: Path):
    source = tmp_path / "hydro-incomplete.nc"
    _create_hydro_netcdf(source)
    with netCDF4.Dataset(source, "a") as dataset:
        dataset.variables["landmesh_face_node"][1, :] = [2, 3, 4, 0]

    output = tmp_path / "stores"
    _convert_hydro1k(source, str(output), face_chunk_size=1, tile_size=1.0)

    mesh = zarr.open_group(str(output / "grids/hydro1k.zarr"), mode="r")
    assert mesh["face_node"][1].tolist() == [1, 2, 3, -1]
    parquet_files = list((output / "grids/hydro1k-face-index").glob("**/*.parquet"))
    assert sum(parquet.read_table(path).num_rows for path in parquet_files) == 1


def test_downloader_selects_default_cloud_grid_profile(tmp_path: Path, monkeypatch):
    manifest = (
        Path(__file__).resolve().parents[1]
        / "src/point_crop_surface_generator/data/clm_mappingfiles.txt"
    )
    input_root = tmp_path / "inputdata"
    requested = []

    def fake_size(url: str) -> int:
        requested.append(url.rsplit("/", 1)[-1])
        return 3

    def fake_download(url: str, destination: Path, size: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"abc")

    monkeypatch.setattr("point_crop_surface_generator.download._remote_size", fake_size)
    monkeypatch.setattr(
        "point_crop_surface_generator.download._download_one", fake_download
    )
    download_sources(
        input_root,
        raw_manifest=tmp_path / "unused-raw.txt",
        grid_manifest=manifest,
        skip_raw=True,
    )

    assert set(requested) == DEFAULT_GRID_FILES
    assert len(requested) == 12
    assert all(
        (input_root / "lnd/clm2/mappingdata/grids" / name).is_file()
        for name in requested
    )


def test_download_size_uses_svn_list_when_xml_omits_size(monkeypatch):
    calls = []

    def fake_run(arguments, **kwargs):
        calls.append(arguments[1:3])
        if arguments[1] == "info":
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout='<info><entry kind="file" /></info>',
                stderr="",
            )
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout="123 brian 456789 Oct 01 12:34 mksrf_lanwat.050425.nc\n",
            stderr="",
        )

    monkeypatch.setattr(
        "point_crop_surface_generator.download.subprocess.run", fake_run
    )

    assert _remote_size("https://source.example/mksrf_lanwat.050425.nc") == 456789
    assert calls == [["info", "--xml"], ["ls", "-v"]]


def test_source_download_populates_raw_and_mapping_grid_paths(
    tmp_path: Path, monkeypatch
):
    input_root = tmp_path / "inputdata"
    raw_relative = "lnd/clm2/rawdata/test-field.nc"
    raw_manifest = tmp_path / "raw.txt"
    raw_manifest.write_text(f"{raw_relative}\n", encoding="utf-8")
    grid_name = "SCRIPgrid_3minx3min_MODISv2_c190503.nc"
    grid_manifest = tmp_path / "grids.txt"
    grid_manifest.write_text(
        "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/lnd/clm2/"
        f"mappingdata/grids/{grid_name}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "point_crop_surface_generator.download._remote_size", lambda _url: 3
    )

    def write_source(_url: str, destination: Path, _size: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"abc")

    monkeypatch.setattr(
        "point_crop_surface_generator.download._download_one", write_source
    )
    download_sources(
        input_root,
        raw_manifest=raw_manifest,
        grid_manifest=grid_manifest,
    )

    assert (input_root / raw_relative).read_bytes() == b"abc"
    assert (
        input_root / "lnd/clm2/mappingdata/grids" / grid_name
    ).read_bytes() == b"abc"
