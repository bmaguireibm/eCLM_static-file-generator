"""Sample CLM topographic statistics from the cloud Hydro1K mesh."""

from __future__ import annotations

import math

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as parquet
import zarr
from shapely.geometry import Polygon

from .s3 import S3Config, parse_s3_uri

TOPOGRAPHY_STORE = (
    "raw/lnd/clm2/rawdata/"
    "mksrf_topostats_1km-merge-10min_HYDRO1K-merge-nomask_simyr2000.c130402.nc.zarr"
)


def _read_indices(array, indices: np.ndarray) -> np.ndarray:
    """Read sparse first-axis indices with at most one read per Zarr chunk."""
    indices = np.asarray(indices, dtype=np.int64)
    output = np.empty((indices.size,) + tuple(array.shape[1:]), dtype=array.dtype)
    chunk_size = int(array.chunks[0])
    chunk_ids = indices // chunk_size
    for chunk_id in np.unique(chunk_ids):
        positions = np.flatnonzero(chunk_ids == chunk_id)
        start = int(chunk_id) * chunk_size
        stop = min(start + chunk_size, int(array.shape[0]))
        chunk = np.asarray(array[start:stop])
        output[positions] = chunk[indices[positions] - start]
    return output


def _candidate_faces(
    fs,
    bucket_prefix: str,
    bounds: tuple[float, float, float, float],
    tile_size: float,
) -> np.ndarray:
    south, north, west, east = bounds
    if east < west:
        raise ValueError("Hydro1K sampling does not support cells crossing 180 degrees")

    index_root = f"{bucket_prefix.rstrip('/')}/grids/hydro1k-face-index"
    # A mesh face is assigned to a tile by its center, which can lie outside
    # the target tile even when the face itself crosses into the target.
    margin = tile_size
    lat_tiles = range(
        math.floor((south - margin) / tile_size),
        math.floor((north + margin) / tile_size) + 1,
    )
    lon_tiles = range(
        math.floor((west - margin) / tile_size),
        math.floor((east + margin) / tile_size) + 1,
    )
    candidate_ids: list[np.ndarray] = []
    for lat_tile in lat_tiles:
        for lon_tile in lon_tiles:
            prefix = f"{index_root}/tile_lat={lat_tile}/tile_lon={lon_tile}"
            for path in fs.find(prefix):
                if not path.endswith(".parquet"):
                    continue
                with fs.open(path, "rb") as handle:
                    table = parquet.read_table(
                        handle,
                        columns=[
                            "face_id",
                            "min_lon",
                            "max_lon",
                            "min_lat",
                            "max_lat",
                        ],
                    )
                overlaps = pc.and_kleene(
                    pc.and_kleene(
                        pc.greater_equal(table["max_lon"], west),
                        pc.less_equal(table["min_lon"], east),
                    ),
                    pc.and_kleene(
                        pc.greater_equal(table["max_lat"], south),
                        pc.less_equal(table["min_lat"], north),
                    ),
                )
                selected = table.filter(overlaps)
                if selected.num_rows:
                    candidate_ids.append(
                        selected["face_id"].to_numpy().astype(np.int64)
                    )
    if not candidate_ids:
        raise ValueError("GeoParquet index found no Hydro1K faces near the target")
    return np.unique(np.concatenate(candidate_ids))


def _unwrap_longitudes(longitudes: np.ndarray, reference: float) -> np.ndarray:
    return reference + ((longitudes - reference + 180.0) % 360.0) - 180.0


def _project(longitudes: np.ndarray, latitudes: np.ndarray, reference_lat: float):
    return np.column_stack(
        (
            np.deg2rad(longitudes) * math.cos(math.radians(reference_lat)),
            np.deg2rad(latitudes),
        )
    )


def _face_overlaps(
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    connectivity: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    south, north, west, east = bounds
    reference_lon = (west + east) / 2.0
    reference_lat = (south + north) / 2.0
    target_lon = _unwrap_longitudes(
        np.asarray([west, east, east, west], dtype=np.float64), reference_lon
    )
    target_lat = np.asarray([south, south, north, north], dtype=np.float64)
    target_polygon = Polygon(_project(target_lon, target_lat, reference_lat))
    target_area = float(target_polygon.area)
    if target_area <= 0.0:
        raise ValueError("target cell has zero projected area")

    weights: list[float] = []
    valid_rows: list[int] = []
    for row, node_ids in enumerate(connectivity):
        lon = _unwrap_longitudes(node_lon[node_ids], reference_lon)
        lat = node_lat[node_ids]
        polygon = Polygon(_project(lon, lat, reference_lat))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        overlap_area = float(polygon.intersection(target_polygon).area)
        if overlap_area > target_area * 1e-14:
            valid_rows.append(row)
            weights.append(overlap_area / target_area)
    if not weights:
        raise ValueError("Hydro1K faces do not overlap the target cell")
    return np.asarray(valid_rows, dtype=np.int64), np.asarray(weights, dtype=np.float64)


def sample_point_topography_fields(
    cloud_root: str,
    bounds: tuple[float, float, float, float],
    *,
    config: S3Config | None = None,
) -> dict[str, float]:
    """Calculate the CLM ``SLOPE`` and ``STD_ELEV`` values for one cell.

    The GeoParquet index locates intersecting mesh faces. Only the Zarr chunks
    containing those faces and their nodes are read. ``STD_ELEV`` uses CLM's
    population, area-weighted standard-deviation formula. ``MEAN_ELEVATION``
    is returned as a diagnostic; it is not a variable in the CLM surfdata
    schema.
    """
    bucket, prefix = parse_s3_uri(cloud_root)
    fs = (config or S3Config.from_env()).filesystem()
    bucket_prefix = f"{bucket}/{prefix.rstrip('/')}"
    grid_path = f"{bucket_prefix.rstrip('/')}/grids/hydro1k.zarr"
    grid = zarr.open_group(store=fs.get_mapper(grid_path), mode="r")

    face_ids = _candidate_faces(
        fs,
        bucket_prefix,
        bounds,
        float(grid.attrs.get("tile_size_degrees", 1.0)),
    )
    node_count = int(grid.attrs["node_count"])
    connectivity = _read_indices(grid["face_node"], face_ids).astype(np.int64)
    if np.any(connectivity < 0) or np.any(connectivity >= node_count):
        raise ValueError("GeoParquet index returned a face with invalid connectivity")
    node_ids = np.unique(connectivity.reshape(-1))
    node_lon = _read_indices(grid["node_lon"], node_ids).astype(np.float64)
    node_lat = _read_indices(grid["node_lat"], node_ids).astype(np.float64)
    local_connectivity = np.searchsorted(node_ids, connectivity)

    overlap_rows, weights = _face_overlaps(
        node_lon, node_lat, local_connectivity, bounds
    )
    overlapping_ids = face_ids[overlap_rows]
    raw_path = f"{bucket_prefix.rstrip('/')}/{TOPOGRAPHY_STORE.lstrip('/')}"
    raw = zarr.open_group(store=fs.get_mapper(raw_path), mode="r")
    if int(raw.attrs["dimensions"]["num_pixels"]) != int(grid.attrs["face_count"]):
        raise ValueError(
            "Hydro1K mesh and topography fields have different face counts"
        )

    elevation = _read_indices(raw["ELEVATION"], overlapping_ids).astype(np.float64)
    slope = _read_indices(raw["SLOPE"], overlapping_ids).astype(np.float64)
    if not np.all(np.isfinite(elevation)) or not np.all(np.isfinite(slope)):
        raise ValueError("Hydro1K topography fields contain non-finite values")
    if np.any(slope < 0.0) or np.any(slope > 90.0):
        raise ValueError("Hydro1K slope values are outside the CLM range 0–90 degrees")

    weight_sum = float(np.sum(weights))
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError(
            f"Hydro1K faces cover {weight_sum:.6f} of the target cell; "
            "full coverage is required"
        )
    destination_fraction = min(weight_sum, 1.0)
    mean_elevation = float(np.dot(weights, elevation) / destination_fraction)
    mean_slope = float(np.dot(weights, slope) / destination_fraction)
    elevation_variance = float(np.dot(weights, (elevation - mean_elevation) ** 2))
    std_elevation = math.sqrt(elevation_variance / weight_sum)
    return {
        "SLOPE": mean_slope,
        "STD_ELEV": std_elevation,
        "MEAN_ELEVATION": mean_elevation,
    }
