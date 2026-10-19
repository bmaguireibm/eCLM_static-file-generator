"""Convert canonical CLM NetCDF inputs into spatially indexed cloud stores.

The outputs deliberately retain the names and source ordering consumed by
``mkmapdata.generate_mapping`` and the one-cell point samplers. Regular grids
and raw fields are chunked Zarr groups; HYDRO1K additionally gets a tiled
GeoParquet face index so regional readers do not scan the complete mesh.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse

import netCDF4
import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet
import zarr
from numcodecs import Blosc

from .download import DEFAULT_GRID_FILES, LAI_GRID_FILE, LAI_RAW_MARKER
from .s3 import S3Config, parse_s3_uri

LOG = logging.getLogger(__name__)
DEFAULT_FACE_CHUNK_SIZE = 250_000
DEFAULT_TILE_SIZE = 1.0
DEFAULT_RAW_MANIFEST = files("point_crop_surface_generator").joinpath(
    "data/rawdata_files.txt"
)
DEFAULT_GRID_MANIFEST = files("point_crop_surface_generator").joinpath(
    "data/clm_mappingfiles.txt"
)
DEFAULT_COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
LAI_RAW_PATH = Path(
    "lnd/clm2/rawdata/pftcftdynharv.0.05x0.05.LUH2.histsimyr2005.c190116/"
    "mksrf_lai_histclm52deg005_earthstatmirca_2005.c190119.nc"
)


def _json_value(value):
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _attribute_dict(variable) -> dict[str, object]:
    return {
        str(name): _json_value(variable.getncattr(name))
        for name in variable.ncattrs()
        if name != "_FillValue"
    }


def _safe_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError(f"invalid store path component: {name!r}")
    return name


def _chunk_shape(
    shape: tuple[int, ...],
    dimensions: tuple[str, ...],
    dtype: np.dtype,
    *,
    row_dimension: str | None = None,
    column_dimension: str | None = None,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
    max_chunk_bytes: int = 8 * 1024 * 1024,
) -> tuple[int, ...] | None:
    if not shape:
        return None
    chunks = [max(1, size) for size in shape]
    for axis, (dimension, size) in enumerate(zip(dimensions, shape)):
        lowered = dimension.lower()
        if dimension in {row_dimension, column_dimension}:
            chunks[axis] = min(size, 256)
        elif lowered in {"grid_size", "num_pixels", "nface", "nfaces", "face"}:
            chunks[axis] = min(size, face_chunk_size)
        elif lowered in {"time", "month", "months", "pft", "natpft", "cft"}:
            chunks[axis] = min(
                size, 1 if lowered in {"time", "month", "months"} else 16
            )
        else:
            chunks[axis] = min(size, 64)

    item_size = max(1, np.dtype(dtype).itemsize)
    while math.prod(chunks) * item_size > max_chunk_bytes:
        largest = max(range(len(chunks)), key=chunks.__getitem__)
        if chunks[largest] <= 1:
            break
        chunks[largest] = (chunks[largest] + 1) // 2
    return tuple(chunks)


def _slices(
    shape: tuple[int, ...], chunks: tuple[int, ...]
) -> Iterator[tuple[slice, ...]]:
    counts = tuple((size + chunk - 1) // chunk for size, chunk in zip(shape, chunks))
    for block in np.ndindex(*counts):
        yield tuple(
            slice(index * chunk, min((index + 1) * chunk, size))
            for index, chunk, size in zip(block, chunks, shape)
        )


def _variable_dtype(variable) -> np.dtype:
    dtype = np.dtype(variable.dtype)
    if "scale_factor" in variable.ncattrs() or "add_offset" in variable.ncattrs():
        scale_dtype = (
            np.asarray(variable.getncattr("scale_factor")).dtype
            if "scale_factor" in variable.ncattrs()
            else np.dtype("f4")
        )
        offset_dtype = (
            np.asarray(variable.getncattr("add_offset")).dtype
            if "add_offset" in variable.ncattrs()
            else np.dtype("f4")
        )
        return np.result_type(dtype, scale_dtype, offset_dtype)
    return dtype


def _copy_variable(
    group,
    variable,
    name: str,
    dimensions: tuple[str, ...],
    *,
    row_dimension: str | None = None,
    column_dimension: str | None = None,
    coordinate_kind: str | None = None,
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
) -> bool:
    dtype = _variable_dtype(variable)
    if dtype.kind in "OU":
        return False
    source_units = str(getattr(variable, "units", "degrees"))
    if coordinate_kind and "radian" in source_units.lower():
        dtype = np.result_type(dtype, np.float64)
    shape = tuple(int(size) for size in variable.shape)
    chunks = _chunk_shape(
        shape,
        dimensions,
        dtype,
        row_dimension=row_dimension,
        column_dimension=column_dimension,
        face_chunk_size=face_chunk_size,
    )
    fill_value = getattr(variable, "_FillValue", None)
    if dtype != np.dtype(variable.dtype) and fill_value is not None:
        fill_value = np.nan
    target = group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        compressor=DEFAULT_COMPRESSOR,
        fill_value=_json_value(fill_value),
        overwrite=True,
    )
    attributes = _attribute_dict(variable)
    if coordinate_kind and "radian" in source_units.lower():
        attributes["units"] = (
            "degrees_north" if coordinate_kind == "latitude" else "degrees_east"
        )
    if dtype != np.dtype(variable.dtype):
        attributes.pop("scale_factor", None)
        attributes.pop("add_offset", None)
        attributes.pop("_Unsigned", None)
        if fill_value is not None:
            attributes["_FillValue"] = float("nan")
    target.attrs.update(attributes)
    target.attrs["dimensions"] = list(dimensions)

    if not shape:
        target[()] = variable[()]
        return True
    assert chunks is not None
    for selection in _slices(shape, chunks):
        values = variable[selection]
        if np.ma.isMaskedArray(values):
            replacement = (
                np.nan
                if np.issubdtype(dtype, np.floating)
                and dtype != np.dtype(variable.dtype)
                else fill_value
            )
            if replacement is None:
                replacement = 0
            values = np.ma.filled(values, replacement)
        if coordinate_kind and "radian" in source_units.lower():
            values = np.rad2deg(values)
        target[selection] = np.asarray(values, dtype=dtype)
    return True


def _write_metadata(
    group, dataset, *, row_dimension=None, column_dimension=None
) -> None:
    dimensions = {
        str(name): int(len(value)) for name, value in dataset.dimensions.items()
    }
    variables = {}
    for name, variable in dataset.variables.items():
        entry = {
            "dimensions": list(variable.dimensions),
            "shape": [int(size) for size in variable.shape],
        }
        if hasattr(variable, "_FillValue"):
            entry["fill_value"] = _json_value(variable.getncattr("_FillValue"))
        variables[str(name)] = entry
    metadata = {
        str(name): _json_value(dataset.getncattr(name))
        for name in dataset.ncattrs()
        if name not in {"dimensions", "variables", "spatial"}
    }
    metadata.update(
        dimensions=dimensions,
        variables=variables,
    )
    if row_dimension and column_dimension:
        metadata["spatial"] = {
            "row_dimension": row_dimension,
            "column_dimension": column_dimension,
        }
    group.attrs.update(metadata)


def _coordinate_metadata(dataset) -> tuple[str, str, str, str] | None:
    latitude_names = {"lat", "latitude", "latixy", "yc", "nav_lat"}
    longitude_names = {"lon", "longitude", "longxy", "xc", "nav_lon"}
    latitudes = []
    longitudes = []
    for name, variable in dataset.variables.items():
        lower = name.lower()
        standard_name = str(getattr(variable, "standard_name", "")).lower()
        axis = str(getattr(variable, "axis", "")).upper()
        units = str(getattr(variable, "units", "")).lower()
        score = (
            4 * (standard_name == "latitude")
            + 3 * (axis == "Y")
            + 2 * ("degrees_north" in units or "radian_north" in units)
            + (lower in latitude_names)
        )
        if score:
            if variable.ndim in {1, 2}:
                latitudes.append((score, name, tuple(variable.dimensions)))
        score = (
            4 * (standard_name == "longitude")
            + 3 * (axis == "X")
            + 2 * ("degrees_east" in units or "radian_east" in units)
            + (lower in longitude_names)
        )
        if score:
            if variable.ndim in {1, 2}:
                longitudes.append((score, name, tuple(variable.dimensions)))

    matches = []
    for lat_score, lat_name, lat_dims in latitudes:
        for lon_score, lon_name, lon_dims in longitudes:
            if len(lat_dims) == len(lon_dims) == 1 and lat_dims[0] != lon_dims[0]:
                matches.append(
                    (
                        lat_score + lon_score,
                        lat_name,
                        lon_name,
                        lat_dims[0],
                        lon_dims[0],
                    )
                )
            elif len(lat_dims) == len(lon_dims) == 2 and lat_dims == lon_dims:
                matches.append(
                    (
                        lat_score + lon_score,
                        lat_name,
                        lon_name,
                        lat_dims[-2],
                        lat_dims[-1],
                    )
                )
    if not matches:
        return None
    _, lat_name, lon_name, row_dimension, column_dimension = max(matches)
    return lat_name, lon_name, row_dimension, column_dimension


def _degrees(values: np.ndarray, units: str) -> np.ndarray:
    if "radian" in units.lower():
        return np.rad2deg(values)
    return values


def _normalize_lon(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _add_raw_spatial_index(group, row_dimension: str, column_dimension: str) -> None:
    spatial = group.attrs["spatial"]
    lat_name = spatial["latitude_variable"]
    lon_name = spatial["longitude_variable"]
    row_size = int(group.attrs["dimensions"][row_dimension])
    column_size = int(group.attrs["dimensions"][column_dimension])
    row_chunk = min(256, row_size)
    column_chunk = min(256, column_size)
    rows: dict[str, list[float | int]] = {
        key: []
        for key in (
            "row_start",
            "row_stop",
            "column_start",
            "column_stop",
            "lat_min",
            "lat_max",
            "lon_min",
            "lon_max",
        )
    }
    lat_array = group[lat_name]
    lon_array = group[lon_name]
    lat_dims = tuple(lat_array.attrs["dimensions"])
    lon_dims = tuple(lon_array.attrs["dimensions"])
    for row_start in range(0, row_size, row_chunk):
        row_stop = min(row_start + row_chunk, row_size)
        for column_start in range(0, column_size, column_chunk):
            column_stop = min(column_start + column_chunk, column_size)
            index_row_start = max(0, row_start - 1)
            index_row_stop = min(row_size, row_stop + 1)
            index_column_start = max(0, column_start - 1)
            index_column_stop = min(column_size, column_stop + 1)

            def coordinate_window(array, dimensions):
                selection = tuple(
                    slice(index_row_start, index_row_stop)
                    if dim == row_dimension
                    else slice(index_column_start, index_column_stop)
                    if dim == column_dimension
                    else slice(None)
                    for dim in dimensions
                )
                return np.asarray(array[selection], dtype=np.float64)

            lat = _degrees(
                coordinate_window(lat_array, lat_dims),
                str(lat_array.attrs.get("units", "degrees")),
            )
            lon = _normalize_lon(
                _degrees(
                    coordinate_window(lon_array, lon_dims),
                    str(lon_array.attrs.get("units", "degrees")),
                )
            )
            finite_lon = lon[np.isfinite(lon)]
            finite_lat = lat[np.isfinite(lat)]
            lon_min, lon_max = (
                (float(np.min(finite_lon)), float(np.max(finite_lon)))
                if finite_lon.size
                else (-180.0, 180.0)
            )
            lat_min, lat_max = (
                (float(np.min(finite_lat)), float(np.max(finite_lat)))
                if finite_lat.size
                else (-90.0, 90.0)
            )
            base = {
                "row_start": row_start,
                "row_stop": row_stop,
                "column_start": column_start,
                "column_stop": column_stop,
                "lat_min": lat_min,
                "lat_max": lat_max,
            }
            if lon_max - lon_min > 180.0:
                positive = lon[lon >= 0.0]
                negative = lon[lon < 0.0]
                for lower, upper in (
                    (float(np.min(positive)), 180.0),
                    (-180.0, float(np.max(negative))),
                ):
                    for key, value in {
                        **base,
                        "lon_min": lower,
                        "lon_max": upper,
                    }.items():
                        rows[key].append(value)
            else:
                for key, value in {
                    **base,
                    "lon_min": lon_min,
                    "lon_max": lon_max,
                }.items():
                    rows[key].append(value)

    index = group.create_group("spatial_index")
    index.attrs.update(
        {
            "version": 1,
            "source": "raw-field coordinate bounds",
            "row_dimension": row_dimension,
            "column_dimension": column_dimension,
        }
    )
    for name, values in rows.items():
        dtype = np.int64 if name.endswith(("start", "stop")) else np.float64
        index.create_dataset(name, data=np.asarray(values, dtype=dtype), chunks=True)


def _convert_netcdf_group(
    source: Path,
    output_uri: str,
    *,
    relative_path: str,
    face_chunk_size: int,
    spatial_raw: bool = False,
) -> int:
    target_uri = f"{output_uri.rstrip('/')}/{relative_path.lstrip('/')}"
    group = _open_group_for_write(target_uri)
    with netCDF4.Dataset(source) as dataset:
        coordinate_metadata = _coordinate_metadata(dataset) if spatial_raw else None
        row_dimension, column_dimension = (
            coordinate_metadata[2:] if coordinate_metadata else (None, None)
        )
        _write_metadata(
            group,
            dataset,
            row_dimension=row_dimension,
            column_dimension=column_dimension,
        )
        group.attrs.update(
            {
                "source_file": source.name,
                "source_format": "NetCDF",
                "kind": "raw-field" if spatial_raw else "SCRIP",
            }
        )
        if coordinate_metadata:
            lat_name, lon_name = coordinate_metadata[:2]
            spatial = {
                "latitude_variable": lat_name,
                "longitude_variable": lon_name,
                "row_dimension": row_dimension,
                "column_dimension": column_dimension,
                "curvilinear": dataset.variables[lat_name].ndim == 2,
            }
            group.attrs["spatial"] = spatial
        variable_metadata = dict(group.attrs["variables"])
        copied = 0
        for name, variable in dataset.variables.items():
            coordinate_kind = None
            if coordinate_metadata and name == coordinate_metadata[0]:
                coordinate_kind = "latitude"
            elif coordinate_metadata and name == coordinate_metadata[1]:
                coordinate_kind = "longitude"
            was_copied = _copy_variable(
                group,
                variable,
                str(name),
                tuple(variable.dimensions),
                row_dimension=row_dimension,
                column_dimension=column_dimension,
                coordinate_kind=coordinate_kind,
                face_chunk_size=face_chunk_size,
            )
            entry = dict(variable_metadata[str(name)])
            if was_copied:
                entry["chunks"] = [int(size) for size in group[str(name)].chunks]
                copied += 1
            else:
                entry["skipped"] = "variable-length or object dtype"
            variable_metadata[str(name)] = entry
        group.attrs["variables"] = variable_metadata
        if coordinate_metadata:
            _add_raw_spatial_index(group, row_dimension, column_dimension)
    LOG.info("converted %s -> %s", source, target_uri)
    return copied


def _uri_filesystem(uri: str):
    if uri.startswith("s3://"):
        bucket, prefix = parse_s3_uri(uri)
        return S3Config.from_env().filesystem(), f"{bucket}/{prefix}"
    import fsspec

    fs, path = fsspec.core.url_to_fs(uri)
    return fs, path


def _open_group_for_write(uri: str):
    if uri.startswith("s3://"):
        fs, store_path = _uri_filesystem(uri)
        if fs.exists(store_path):
            fs.rm(store_path, recursive=True)
        return zarr.open_group(store=fs.get_mapper(store_path), mode="w")
    path = Path(uri.removeprefix("file://"))
    if path.exists():
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_group(str(path), mode="w")


def _read_grid_source_names(manifest: Path) -> list[str]:
    output = []
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = urlparse(line)
        output.append(Path(parsed.path if parsed.scheme else line).name)
    return output


def _convert_scrip_grid(
    source: Path,
    output_uri: str,
    *,
    relative_path: str,
    face_chunk_size: int,
) -> int:
    target_uri = f"{output_uri.rstrip('/')}/{relative_path.lstrip('/')}"
    group = _open_group_for_write(target_uri)
    with netCDF4.Dataset(source) as dataset:
        if (
            "grid_size" not in dataset.dimensions
            or "grid_center_lon" not in dataset.variables
        ):
            raise ValueError(f"{source}: not a SCRIP grid")
        _write_metadata(group, dataset)
        group.attrs.update(
            {
                "source_file": source.name,
                "source_format": "NetCDF",
                "kind": "SCRIP",
            }
        )
        copied = 0
        variable_metadata = dict(group.attrs["variables"])
        for name, variable in dataset.variables.items():
            was_copied = _copy_variable(
                group,
                variable,
                str(name),
                tuple(variable.dimensions),
                face_chunk_size=face_chunk_size,
            )
            entry = dict(variable_metadata[str(name)])
            if was_copied:
                entry["chunks"] = [int(size) for size in group[str(name)].chunks]
                copied += 1
            else:
                entry["skipped"] = "variable-length or object dtype"
            variable_metadata[str(name)] = entry
        group.attrs["variables"] = variable_metadata
        lon_var = dataset.variables["grid_center_lon"]
        lat_var = dataset.variables["grid_center_lat"]
        corner_lon_var = dataset.variables.get("grid_corner_lon")
        corner_lat_var = dataset.variables.get("grid_corner_lat")
        size = int(lon_var.shape[0])
        chunk_size = min(face_chunk_size, size)
        record_values = {
            name: []
            for name in (
                "chunk_start",
                "chunk_stop",
                "lon_start",
                "lon_width",
                "lat_min",
                "lat_max",
            )
        }
        for start in range(0, size, chunk_size):
            stop = min(start + chunk_size, size)
            if corner_lon_var is not None and corner_lat_var is not None:
                lon_values = np.ma.asarray(corner_lon_var[start:stop])
                lat_values = np.ma.asarray(corner_lat_var[start:stop])
                lon = _degrees(
                    np.asarray(np.ma.filled(lon_values, np.nan), dtype=np.float64),
                    str(getattr(corner_lon_var, "units", "degrees")),
                )
                lat = _degrees(
                    np.asarray(np.ma.filled(lat_values, np.nan), dtype=np.float64),
                    str(getattr(corner_lat_var, "units", "degrees")),
                )
            else:
                lon = _degrees(
                    np.asarray(lon_var[start:stop], dtype=np.float64),
                    str(getattr(lon_var, "units", "degrees")),
                )
                lat = _degrees(
                    np.asarray(lat_var[start:stop], dtype=np.float64),
                    str(getattr(lat_var, "units", "degrees")),
                )
            finite_lon = np.isfinite(lon)
            finite_lat = np.isfinite(lat)
            arc_start, arc_width = (
                _minimum_longitude_arc(lon[finite_lon])
                if np.any(finite_lon)
                else (0.0, 360.0)
            )
            values = {
                "chunk_start": start,
                "chunk_stop": stop,
                "lon_start": arc_start,
                "lon_width": arc_width,
                "lat_min": float(np.min(lat[finite_lat]))
                if np.any(finite_lat)
                else -90.0,
                "lat_max": float(np.max(lat[finite_lat]))
                if np.any(finite_lat)
                else 90.0,
            }
            for name, value in values.items():
                record_values[name].append(value)
        index = group.create_group("spatial_index")
        index.attrs.update(
            {
                "version": 1,
                "source": "grid corner bounds"
                if corner_lon_var is not None and corner_lat_var is not None
                else "grid center bounds",
                "longitude": "start and width in [0, 360), increasing clockwise",
                "chunk_size": int(chunk_size),
            }
        )
        for name, values in record_values.items():
            dtype = np.int64 if name.startswith("chunk_") else np.float64
            index.create_dataset(
                name, data=np.asarray(values, dtype=dtype), chunks=True
            )
    LOG.info("converted SCRIP grid %s -> %s", source, target_uri)
    return copied


def _minimum_longitude_arc(longitudes: np.ndarray) -> tuple[float, float]:
    normalized = np.sort(np.asarray(longitudes, dtype=np.float64) % 360.0)
    if normalized.size <= 1:
        return float(normalized[0]) if normalized.size else 0.0, 0.0
    gaps = np.diff(np.concatenate((normalized, normalized[:1] + 360.0)))
    largest_gap = int(np.argmax(gaps))
    start = float(normalized[(largest_gap + 1) % normalized.size])
    width = float(360.0 - gaps[largest_gap])
    return start, width


def _ugrid_connectivity(
    variable,
    start: int,
    stop: int,
    face_count: int,
    node_count: int,
    fill_value,
    index_base: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    if variable.shape[0] == face_count:
        raw = np.ma.asarray(variable[start:stop, :])
    elif variable.shape[1] == face_count:
        raw = np.ma.asarray(variable[:, start:stop]).T
    else:
        raise ValueError("HYDRO1K face-node array is not aligned with face coordinates")
    valid = ~np.ma.getmaskarray(raw)
    values = np.asarray(
        np.ma.filled(raw, fill_value if fill_value is not None else -1),
        dtype=np.int64,
    )
    if fill_value is not None:
        valid &= values != int(fill_value)
    if index_base is None:
        if "start_index" in variable.ncattrs():
            index_base = int(variable.getncattr("start_index"))
        elif fill_value is not None and int(fill_value) == 0:
            index_base = 1
        else:
            index_base = 0 if np.any(values[valid] == 0) else 1
    if index_base not in (0, 1):
        raise ValueError(f"unsupported HYDRO1K connectivity start index: {index_base}")
    values -= index_base
    valid &= (values >= 0) & (values < node_count)
    values[~valid] = -1
    return values, index_base, np.all(valid, axis=1)


def _create_zarr_array(group, name: str, shape, dtype, chunks, attrs=None):
    return group.create_dataset(
        name,
        shape=tuple(shape),
        dtype=dtype,
        chunks=chunks,
        compressor=DEFAULT_COMPRESSOR,
        overwrite=True,
    )


def _write_parquet_index(
    fs,
    target_path: str,
    table: pa.Table,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> None:
    crs84 = {"name": "OGC:CRS84"}
    metadata = {
        "version": "1.0.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "crs": crs84,
                "bbox": [lon_min, lat_min, lon_max, lat_max],
            }
        },
        "bbox": [lon_min, lat_min, lon_max, lat_max],
    }
    table = table.replace_schema_metadata(
        {b"geo": json.dumps(metadata).encode("utf-8")}
    )
    fs.makedirs(target_path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(target_path, "wb") as handle:
        parquet.write_table(table, handle, compression="zstd", row_group_size=65_536)


def _convert_hydro1k(
    source: Path,
    output_uri: str,
    *,
    face_chunk_size: int,
    tile_size: float,
) -> tuple[int, int]:
    from shapely.geometry import Polygon
    from shapely.wkb import dumps as wkb_dumps

    mesh_uri = f"{output_uri.rstrip('/')}/grids/hydro1k.zarr"
    mesh = _open_group_for_write(mesh_uri)
    hydro_fs, hydro_root = _uri_filesystem(
        f"{output_uri.rstrip('/')}/grids/hydro1k-face-index"
    )
    if hydro_fs.exists(hydro_root):
        hydro_fs.rm(hydro_root, recursive=True)
    with netCDF4.Dataset(source) as dataset:
        names = {
            "face_lon": "landmesh_face_x",
            "face_lat": "landmesh_face_y",
            "node_lon": "landmesh_node_x",
            "node_lat": "landmesh_node_y",
            "face_node": "landmesh_face_node",
            "face_area": "landmesh_face_area",
            "face_mask": "landmesh_face_mask",
        }
        required = ("face_lon", "face_lat", "node_lon", "node_lat", "face_node")
        missing = [
            names[key] for key in required if names[key] not in dataset.variables
        ]
        if missing:
            raise ValueError(
                f"{source}: missing HYDRO1K variables: {', '.join(missing)}"
            )
        face_lon_var = dataset.variables[names["face_lon"]]
        face_lat_var = dataset.variables[names["face_lat"]]
        node_lon_var = dataset.variables[names["node_lon"]]
        node_lat_var = dataset.variables[names["node_lat"]]
        connectivity_var = dataset.variables[names["face_node"]]
        face_count, node_count = int(face_lon_var.shape[0]), int(node_lon_var.shape[0])
        connectivity_chunk = min(face_chunk_size, face_count)
        node_chunk = min(face_chunk_size, node_count)
        corner_count = int(
            connectivity_var.shape[1]
            if connectivity_var.shape[0] == face_count
            else connectivity_var.shape[0]
        )
        face_chunks = (connectivity_chunk,)
        _create_zarr_array(mesh, "face_lon", (face_count,), "f8", face_chunks)
        _create_zarr_array(mesh, "face_lat", (face_count,), "f8", face_chunks)
        _create_zarr_array(
            mesh,
            "face_node",
            (face_count, corner_count),
            "i8",
            (connectivity_chunk, corner_count),
        )
        _create_zarr_array(mesh, "node_lon", (node_count,), "f8", (node_chunk,))
        _create_zarr_array(mesh, "node_lat", (node_count,), "f8", (node_chunk,))
        for key in ("face_area", "face_mask"):
            if names[key] in dataset.variables:
                dtype = np.dtype("f8") if key == "face_area" else np.dtype("i4")
                _create_zarr_array(mesh, key, (face_count,), dtype, face_chunks)
        mesh.attrs.update(
            {
                "face_count": face_count,
                "node_count": node_count,
                "tile_size_degrees": float(tile_size),
                "source": str(source),
                "source_file": source.name,
                "source_format": "NetCDF UGRID",
                "kind": "UGRID",
            }
        )
        mesh["face_lon"].attrs["units"] = "degrees"
        mesh["face_lat"].attrs["units"] = "degrees"
        mesh["node_lon"].attrs["units"] = "degrees"
        mesh["node_lat"].attrs["units"] = "degrees"
        fill_value = getattr(connectivity_var, "_FillValue", None)
        start_index = getattr(connectivity_var, "start_index", None)
        if start_index is not None:
            index_base = int(start_index)
        elif fill_value is not None and int(fill_value) == 0:
            index_base = 1
        else:
            first_connectivity = np.ma.asarray(
                connectivity_var[0:1, :]
                if connectivity_var.shape[0] == face_count
                else connectivity_var[:, 0:1]
            )
            first_values = np.asarray(
                np.ma.filled(
                    first_connectivity,
                    fill_value if fill_value is not None else -1,
                ),
                dtype=np.int64,
            )
            first_valid = ~np.ma.getmaskarray(first_connectivity)
            if fill_value is not None:
                first_valid &= first_values != int(fill_value)
            index_base = 0 if np.any(first_values[first_valid] == 0) else 1
        parquet_count = 0
        for start in range(0, face_count, connectivity_chunk):
            stop = min(start + connectivity_chunk, face_count)
            face_lon = _normalize_lon(
                _degrees(
                    np.asarray(face_lon_var[start:stop], dtype=np.float64),
                    str(getattr(face_lon_var, "units", "degrees")),
                )
            )
            face_lat = _degrees(
                np.asarray(face_lat_var[start:stop], dtype=np.float64),
                str(getattr(face_lat_var, "units", "degrees")),
            )
            connectivity, _, valid_faces = _ugrid_connectivity(
                connectivity_var,
                start,
                stop,
                face_count,
                node_count,
                fill_value,
                index_base,
            )
            mesh["face_lon"][start:stop] = face_lon
            mesh["face_lat"][start:stop] = face_lat
            mesh["face_node"][start:stop] = connectivity
            if "face_area" in mesh:
                mesh["face_area"][start:stop] = np.asarray(
                    dataset.variables[names["face_area"]][start:stop], dtype=np.float64
                )
            if "face_mask" in mesh:
                mesh["face_mask"][start:stop] = np.asarray(
                    dataset.variables[names["face_mask"]][start:stop], dtype=np.int32
                )

            if not np.any(valid_faces):
                continue
            local = np.flatnonzero(valid_faces)
            selected_connectivity = connectivity[local]
            selected_face_lon = face_lon[local]
            selected_face_lat = face_lat[local]
            node_ids = np.unique(selected_connectivity)
            node_lon = _normalize_lon(
                _degrees(
                    np.asarray(node_lon_var[node_ids], dtype=np.float64),
                    str(getattr(node_lon_var, "units", "degrees")),
                )
            )
            node_lat = _degrees(
                np.asarray(node_lat_var[node_ids], dtype=np.float64),
                str(getattr(node_lat_var, "units", "degrees")),
            )
            # Node chunks are sparsely written as faces arrive. Repeated node
            # IDs have identical coordinates, so this is safe and avoids a
            # second full pass over the global mesh.
            mesh["node_lon"][node_ids] = node_lon
            mesh["node_lat"][node_ids] = node_lat

            bbox_min_lon, bbox_max_lon, bbox_min_lat, bbox_max_lat = [], [], [], []
            geometries = []
            for face, center_lon in zip(selected_connectivity, selected_face_lon):
                lon = node_lon[np.searchsorted(node_ids, face)].astype(np.float64)
                lat = node_lat[np.searchsorted(node_ids, face)].astype(np.float64)
                lon = center_lon + ((lon - center_lon + 180.0) % 360.0) - 180.0
                polygon = Polygon(np.column_stack((lon, lat)))
                min_lon, min_lat, max_lon, max_lat = polygon.bounds
                bbox_min_lon.append(min_lon)
                bbox_max_lon.append(max_lon)
                bbox_min_lat.append(min_lat)
                bbox_max_lat.append(max_lat)
                geometries.append(wkb_dumps(polygon))

            face_ids = np.arange(start, stop, dtype=np.int64)[local]
            tile_lat = np.floor(selected_face_lat / tile_size).astype(np.int64)
            tile_lon = np.floor(selected_face_lon / tile_size).astype(np.int64)
            tile_keys = np.column_stack((tile_lat, tile_lon))
            order = np.lexsort((tile_lon, tile_lat))
            sorted_keys = tile_keys[order]
            group_starts = np.r_[
                0, np.flatnonzero(np.any(np.diff(sorted_keys, axis=0), axis=1)) + 1
            ]
            group_stops = np.r_[group_starts[1:], order.size]
            for group_start, group_stop in zip(group_starts, group_stops):
                tile = sorted_keys[group_start]
                selected = order[group_start:group_stop]
                ids = face_ids[selected]
                min_lon = np.asarray(bbox_min_lon)[selected]
                max_lon = np.asarray(bbox_max_lon)[selected]
                min_lat = np.asarray(bbox_min_lat)[selected]
                max_lat = np.asarray(bbox_max_lat)[selected]
                geometry = pa.array(
                    [geometries[int(index)] for index in selected], type=pa.binary()
                )
                table = pa.table(
                    {
                        "face_id": pa.array(ids, type=pa.int64()),
                        "min_lon": pa.array(min_lon, type=pa.float64()),
                        "max_lon": pa.array(max_lon, type=pa.float64()),
                        "min_lat": pa.array(min_lat, type=pa.float64()),
                        "max_lat": pa.array(max_lat, type=pa.float64()),
                        "geometry": geometry,
                    }
                )
                tile_path = (
                    f"{hydro_root}/tile_lat={int(tile[0])}/tile_lon={int(tile[1])}/"
                    f"part-{start:012d}.parquet"
                )
                bounds = (
                    float(np.min(min_lon)),
                    float(np.max(max_lon)),
                    float(np.min(min_lat)),
                    float(np.max(max_lat)),
                )
                _write_parquet_index(
                    hydro_fs,
                    tile_path,
                    table,
                    lon_min=bounds[0],
                    lon_max=bounds[1],
                    lat_min=bounds[2],
                    lat_max=bounds[3],
                )
                parquet_count += int(selected.size)

        LOG.info(
            "converted HYDRO1K mesh %s -> %s (%s faces)", source, mesh_uri, face_count
        )
    return face_count, parquet_count


def _grid_files(manifest: Path, *, all_grids: bool, lai_only: bool) -> list[str]:
    names = _read_grid_source_names(manifest)
    if lai_only:
        return [name for name in names if name == LAI_GRID_FILE]
    if all_grids:
        return names
    return [name for name in names if name in DEFAULT_GRID_FILES]


def _raw_files(manifest: Path, *, lai_only: bool) -> list[str]:
    entries = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if lai_only:
        entries = [entry for entry in entries if LAI_RAW_MARKER in entry]
        if not entries:
            required_field = (
                "raw-data manifest does not list the required LAI field: "
                f"{LAI_RAW_PATH}"
            )
            raise ValueError(required_field)
    return entries


def _validate_lai_dimensions(raw_path: Path, scrip_path: Path) -> None:
    """Ensure the global LAI field uses the full matching SCRIP grid."""
    with netCDF4.Dataset(raw_path) as raw, netCDF4.Dataset(scrip_path) as scrip:
        lat = next(
            (
                raw.variables[name]
                for name in ("LAT", "lat", "latitude")
                if name in raw.variables
            ),
            None,
        )
        lon = next(
            (
                raw.variables[name]
                for name in ("LON", "lon", "longitude")
                if name in raw.variables
            ),
            None,
        )
        if (
            lat is None
            or lon is None
            or len(lat.dimensions) != 1
            or len(lon.dimensions) != 1
        ):
            raise ValueError(
                f"{raw_path}: expected one-dimensional LAT/LON coordinates"
            )
        actual = (
            len(raw.dimensions[lon.dimensions[0]]),
            len(raw.dimensions[lat.dimensions[0]]),
        )
        if "grid_dims" not in scrip.variables:
            raise ValueError(f"{scrip_path}: missing grid_dims")
        expected_values = np.asarray(
            scrip.variables["grid_dims"][:], dtype=np.int64
        ).reshape(-1)
        if expected_values.size != 2:
            raise ValueError(
                f"{scrip_path}: expected two-dimensional MODISv2 grid_dims"
            )
        expected = (int(expected_values[0]), int(expected_values[1]))
    if actual != expected:
        raise ValueError(
            f"{raw_path}: spatial dimensions {actual} do not match "
            f"{scrip_path} {expected}. The input appears to be an AoI crop, "
            "not the canonical global LAI file. Refresh the raw cache from "
            "the CLM SVN source before preparing Zarr stores."
        )


def _write_preparation_manifest(
    output_uri: str, results: list[dict[str, object]]
) -> None:
    manifest_uri = f"{output_uri.rstrip('/')}/manifest.json"
    filesystem, path = _uri_filesystem(manifest_uri)
    parent = path.rpartition("/")[0]
    if parent:
        filesystem.makedirs(parent, exist_ok=True)
    metadata = {
        "format": "eCLM cloud-optimized preparation inputs",
        "version": "0.3.0",
        "s3_environment": {
            "endpoint": "AWS_ENDPOINT_URL or AWS_ENDPOINT_URL_S3 or "
            "AWS_S3_ENDPOINT or S3_ENDPOINT_URL or S3_ENDPOINT",
            "credentials": "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, "
            "AWS_SESSION_TOKEN",
            "region": "AWS_REGION or AWS_DEFAULT_REGION",
        },
        "results": results,
    }
    with filesystem.open(path, "wb") as handle:
        handle.write((json.dumps(metadata, indent=2) + "\n").encode("utf-8"))


def prepare_stores(
    input_root: Path,
    output_uri: str,
    *,
    raw_manifest: Path = Path(str(DEFAULT_RAW_MANIFEST)),
    grid_manifest: Path = Path(str(DEFAULT_GRID_MANIFEST)),
    face_chunk_size: int = DEFAULT_FACE_CHUNK_SIZE,
    tile_size: float = DEFAULT_TILE_SIZE,
    all_grids: bool = False,
    skip_raw: bool = False,
    lai_only: bool = False,
) -> list[dict[str, object]]:
    if lai_only and skip_raw:
        raise ValueError("--lai-only cannot be combined with --skip-raw")
    if face_chunk_size <= 0 or tile_size <= 0:
        raise ValueError("face chunk size and tile size must be positive")
    input_root = input_root.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    raw_relatives = []
    if not skip_raw:
        if not raw_manifest.is_file():
            raise FileNotFoundError(f"raw-data manifest does not exist: {raw_manifest}")
        raw_relatives = _raw_files(raw_manifest, lai_only=lai_only)
    if not grid_manifest.is_file():
        raise FileNotFoundError(f"grid manifest does not exist: {grid_manifest}")
    grid_root = input_root / "lnd/clm2/mappingdata/grids"
    names = _grid_files(grid_manifest, all_grids=all_grids, lai_only=lai_only)
    if lai_only and LAI_GRID_FILE not in names:
        raise ValueError(
            f"grid manifest does not list the required LAI grid: {LAI_GRID_FILE}"
        )

    raw_sources = [input_root / relative for relative in raw_relatives]
    grid_sources = [grid_root / name for name in names]
    for source in raw_sources:
        if not source.is_file():
            raise FileNotFoundError(f"raw input is missing: {source}")
    for source in grid_sources:
        if not source.is_file():
            raise FileNotFoundError(f"mapping grid is missing: {source}")

    lai_relative = next(
        (relative for relative in raw_relatives if LAI_RAW_MARKER in relative), None
    )
    lai_grid_source = grid_root / LAI_GRID_FILE
    if lai_relative and lai_grid_source.is_file():
        _validate_lai_dimensions(input_root / lai_relative, lai_grid_source)
    elif lai_only:
        raise FileNotFoundError(f"LAI mapping grid is missing: {lai_grid_source}")

    results: list[dict[str, object]] = []
    for relative, source in zip(raw_relatives, raw_sources):
        relative_path = f"raw/{relative}.zarr"
        LOG.info("converting raw field %s", relative)
        count = _convert_netcdf_group(
            source,
            output_uri,
            relative_path=relative_path,
            face_chunk_size=face_chunk_size,
            spatial_raw=True,
        )
        results.append(
            {
                "source": relative,
                "output": f"{output_uri.rstrip('/')}/{relative_path}",
                "kind": "raw-field",
                "count": count,
            }
        )

    for name, source in zip(names, grid_sources):
        LOG.info("converting mapping grid %s", name)
        if name.startswith("UGRID_"):
            face_count, parquet_count = _convert_hydro1k(
                source,
                output_uri,
                face_chunk_size=face_chunk_size,
                tile_size=tile_size,
            )
            results.extend(
                (
                    {
                        "source": name,
                        "output": f"{output_uri.rstrip('/')}/grids/hydro1k.zarr",
                        "kind": "UGRID-Zarr",
                        "count": face_count,
                    },
                    {
                        "source": name,
                        "output": f"{output_uri.rstrip('/')}/grids/hydro1k-face-index",
                        "kind": "GeoParquet face index",
                        "count": parquet_count,
                    },
                )
            )
        else:
            relative_path = f"grids/{name}.zarr"
            count = _convert_scrip_grid(
                source,
                output_uri,
                relative_path=relative_path,
                face_chunk_size=face_chunk_size,
            )
            results.append(
                {
                    "source": name,
                    "output": f"{output_uri.rstrip('/')}/{relative_path}",
                    "kind": "SCRIP",
                    "count": count,
                }
            )

    _write_preparation_manifest(output_uri, results)
    LOG.info("prepared %d stores", len(results))
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert CLM raw NetCDF and mapping grids to cloud Zarr/GeoParquet stores"
        )
    )
    parser.add_argument(
        "--input-root", type=Path, required=True, help="CSMDATA input-data root"
    )
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument(
        "--output-uri", help="S3 or local output root, for example s3://bucket/prefix"
    )
    output.add_argument("--output-dir", type=Path, help="local output directory")
    parser.add_argument(
        "--raw-manifest", type=Path, default=Path(str(DEFAULT_RAW_MANIFEST))
    )
    parser.add_argument(
        "--grid-manifest", type=Path, default=Path(str(DEFAULT_GRID_MANIFEST))
    )
    parser.add_argument("--face-chunk-size", type=int, default=DEFAULT_FACE_CHUNK_SIZE)
    parser.add_argument("--tile-size", type=float, default=DEFAULT_TILE_SIZE)
    parser.add_argument(
        "--all-grids", action="store_true", help="convert all grids in the manifest"
    )
    parser.add_argument(
        "--skip-raw", action="store_true", help="convert mapping grids only"
    )
    parser.add_argument(
        "--lai-only",
        action="store_true",
        help="convert only the LAI field and its SCRIP grid",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.lai_only and args.skip_raw:
        raise SystemExit("--lai-only cannot be combined with --skip-raw")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_uri = args.output_uri or str(args.output_dir.expanduser().resolve())
    prepare_stores(
        args.input_root,
        output_uri,
        raw_manifest=args.raw_manifest,
        grid_manifest=args.grid_manifest,
        face_chunk_size=args.face_chunk_size,
        tile_size=args.tile_size,
        all_grids=args.all_grids,
        skip_raw=args.skip_raw,
        lai_only=args.lai_only,
    )


if __name__ == "__main__":
    main()
