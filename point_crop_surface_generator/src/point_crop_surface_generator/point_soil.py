"""Sample CLM soil and canopy fields from cloud Zarr stores for one cell."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import zarr

from .s3 import S3Config, parse_s3_uri

SOIL_STORES = {
    "texture": "raw/lnd/clm2/rawdata/mksrf_soitex.10level.c010119.nc.zarr",
    "organic": (
        "raw/lnd/clm2/rawdata/"
        "mksrf_organic_10level_5x5min_ISRIC-WISE-NCSCD_nlev7_c120830.nc.zarr"
    ),
    "depth": (
        "raw/lnd/clm2/rawdata/"
        "mksf_soilthk_5x5min_ORNL-Soil_simyr1900-2015_c170630.nc.zarr"
    ),
    "peat": "raw/lnd/clm2/rawdata/mksrf_peatf_0.5x0.5_AVHRR_simyr2000.c130228.nc.zarr",
    "fmax": "raw/lnd/clm2/rawdata/mksrf_fmax_3x3min_USGS_c120911.nc.zarr",
    "color": "raw/lnd/clm2/rawdata/pftcftdynharv.0.05x0.05.LUH2.histsimyr2005.c190116/"
    "mksrf_soilcolor_histclm52deg005_earthstatmirca_2005.c190116.nc.zarr",
}
CANOPY_STORE = (
    "raw/lnd/clm2/rawdata/"
    "pftcftdynharv.0.05x0.05.LUH2.histsimyr2005.c190116/"
    "mksrf_lai_histclm52deg005_earthstatmirca_2005.c190119.nc.zarr"
)
CANOPY_FIELDS = {
    "MONTHLY_LAI": "MONTHLY_LAI",
    "MONTHLY_SAI": "MONTHLY_SAI",
    "MONTHLY_HEIGHT_TOP": "MONTHLY_HEIGHT_TOP",
    "MONTHLY_HEIGHT_BOT": "MONTHLY_HEIGHT_BOT",
}
ANCILLARY_STORES = {
    "vocef": "raw/lnd/clm2/rawdata/mksrf_vocef_0.5x0.5_simyr2000.c110531.nc.zarr",
    "gdp": "raw/lnd/clm2/rawdata/mksrf_gdp_0.5x0.5_AVHRR_simyr2000.c130228.nc.zarr",
    "abm": "raw/lnd/clm2/rawdata/mksrf_abm_0.5x0.5_AVHRR_simyr2000.c130201.nc.zarr",
    "harvest": (
        "raw/lnd/clm2/rawdata/"
        "pftcftdynharv.0.25x0.25.LUH2.histsimyr1850-2015.c170629/"
        "mksrf_landuse_histclm50_LUH2_2005.c170629.nc.zarr"
    ),
}
VOCEF_FIELDS = {
    "EF1_BTR": "ef_btr",
    "EF1_FET": "ef_fet",
    "EF1_FDT": "ef_fdt",
    "EF1_SHR": "ef_shr",
    "EF1_GRS": "ef_grs",
    "EF1_CRP": "ef_crp",
}
HARVEST_FIELDS = {
    "CONST_HARVEST_VH1": "HARVEST_VH1",
    "CONST_HARVEST_VH2": "HARVEST_VH2",
    "CONST_HARVEST_SH1": "HARVEST_SH1",
    "CONST_HARVEST_SH2": "HARVEST_SH2",
    "CONST_HARVEST_SH3": "HARVEST_SH3",
    "CONST_GRAZING": "GRAZING",
}


def _open_group(fs, bucket_prefix: str, relative_path: str):
    mapper = fs.get_mapper(f"{bucket_prefix.rstrip('/')}/{relative_path.lstrip('/')}")
    return zarr.open_group(store=mapper, mode="r")


def _index_records(
    group, bounds: tuple[float, float, float, float]
) -> list[dict[str, int]]:
    slat, nlat, wlon, elon = bounds
    index = group["spatial_index"]
    values = {name: np.asarray(index[name][:]) for name in index.array_keys()}
    matches = (
        (values["lat_max"] >= slat)
        & (values["lat_min"] <= nlat)
        & (values["lon_max"] >= wlon)
        & (values["lon_min"] <= elon)
    )
    return [
        {name: int(value[i]) for name, value in values.items()}
        for i in np.flatnonzero(matches)
    ]


def _axis_overlaps(
    group,
    variable_name: str,
    dimension_size: int,
    chunk_start: int,
    chunk_stop: int,
    lower: float,
    upper: float,
    *,
    latitude: bool,
) -> dict[int, float]:
    read_start = max(0, chunk_start - 1)
    read_stop = min(dimension_size, chunk_stop + 1)
    centers = np.asarray(group[variable_name][read_start:read_stop], dtype=np.float64)
    if not latitude:
        reference = (lower + upper) / 2.0
        centers = reference + ((centers - reference + 180.0) % 360.0) - 180.0
    overlaps: dict[int, float] = {}
    for cell in range(chunk_start, chunk_stop):
        local = cell - read_start
        if local == 0 or local + 1 >= centers.size:
            continue
        first = (centers[local - 1] + centers[local]) / 2.0
        second = (centers[local] + centers[local + 1]) / 2.0
        cell_lower, cell_upper = min(first, second), max(first, second)
        overlap_lower = max(lower, cell_lower)
        overlap_upper = min(upper, cell_upper)
        if overlap_upper <= overlap_lower:
            continue
        if latitude:
            weight = math.sin(math.radians(overlap_upper)) - math.sin(
                math.radians(overlap_lower)
            )
        else:
            weight = math.radians(overlap_upper - overlap_lower)
        if weight > 0.0:
            overlaps[cell] = weight
    return overlaps


def _regular_overlaps(
    group, bounds: tuple[float, float, float, float]
) -> list[tuple[int, int, float]]:
    slat, nlat, wlon, elon = bounds
    spatial = group.attrs["spatial"]
    dimensions = group.attrs["dimensions"]
    records = _index_records(group, bounds)
    cells: dict[tuple[int, int], float] = {}
    for record in records:
        rows = _axis_overlaps(
            group,
            spatial["latitude_variable"],
            int(dimensions[spatial["row_dimension"]]),
            record["row_start"],
            record["row_stop"],
            slat,
            nlat,
            latitude=True,
        )
        columns = _axis_overlaps(
            group,
            spatial["longitude_variable"],
            int(dimensions[spatial["column_dimension"]]),
            record["column_start"],
            record["column_stop"],
            wlon,
            elon,
            latitude=False,
        )
        for row, lat_weight in rows.items():
            for column, lon_weight in columns.items():
                cells[(row, column)] = lat_weight * lon_weight
    if not cells:
        raise ValueError(
            "spatial index found no source cells overlapping the target cell"
        )
    total = sum(cells.values())
    target_area = math.radians(elon - wlon) * (
        math.sin(math.radians(nlat)) - math.sin(math.radians(slat))
    )
    if not math.isclose(
        total,
        target_area,
        rel_tol=0.0,
        abs_tol=max(target_area * 1e-4, 1e-14),
    ):
        raise ValueError("source grid does not fully cover the target cell")
    return [(row, column, area / total) for (row, column), area in cells.items()]


def _curvilinear_containing_cell(
    group, bounds: tuple[float, float, float, float]
) -> tuple[int, int, float]:
    """Find a containing curvilinear cell; reject targets requiring polygon clipping."""
    slat, nlat, wlon, elon = bounds
    spatial = group.attrs["spatial"]
    candidates: list[tuple[float, int, int, float, float, float, float]] = []
    target_lat = (slat + nlat) / 2.0
    target_lon = (wlon + elon) / 2.0
    for record in _index_records(group, bounds):
        r0, r1 = record["row_start"], record["row_stop"]
        c0, c1 = record["column_start"], record["column_stop"]
        lat = np.asarray(
            group[spatial["latitude_variable"]][r0:r1, c0:c1], dtype=np.float64
        )
        lon = np.asarray(
            group[spatial["longitude_variable"]][r0:r1, c0:c1], dtype=np.float64
        )
        longitude_delta = ((lon - target_lon + 180.0) % 360.0) - 180.0
        distance = (lat - target_lat) ** 2 + (
            np.cos(np.deg2rad(lat)) * longitude_delta
        ) ** 2
        local_row, local_column = np.unravel_index(
            np.nanargmin(distance), distance.shape
        )
        row, column = r0 + int(local_row), c0 + int(local_column)
        center_lat = float(lat[local_row, local_column])
        center_lon = (
            target_lon
            + ((float(lon[local_row, local_column]) - target_lon + 180.0) % 360.0)
            - 180.0
        )
        lat_step = float(np.nanmedian(np.abs(np.diff(lat[:, local_column]))))
        longitude_row = (
            target_lon + ((lon[local_row, :] - target_lon + 180.0) % 360.0) - 180.0
        )
        lon_step = float(np.nanmedian(np.abs(np.diff(longitude_row))))
        candidates.append(
            (
                float(np.nanmin(distance)),
                row,
                column,
                center_lat,
                center_lon,
                lat_step,
                lon_step,
            )
        )
    if not candidates:
        raise ValueError(
            "spatial index found no source cells overlapping the target cell"
        )
    _, row, column, center_lat, center_lon, lat_step, lon_step = min(candidates)
    tolerance = 1e-5
    if (
        max(abs(center_lat - slat), abs(center_lat - nlat)) > lat_step / 2 + tolerance
        or max(abs(center_lon - wlon), abs(center_lon - elon))
        > lon_step / 2 + tolerance
    ):
        raise ValueError(
            "target crosses a curvilinear source-cell boundary; "
            "polygon overlap is required"
        )
    return row, column, 1.0


def _axis_weights_from_centers(
    centers: np.ndarray,
    first_index: int,
    start: int,
    stop: int,
    lower: float,
    upper: float,
    *,
    latitude: bool,
) -> dict[int, float]:
    """Calculate overlap weights for an axis represented by cell centers."""
    weights: dict[int, float] = {}
    for index in range(start, stop):
        local = index - first_index
        if local <= 0 or local + 1 >= centers.size:
            continue
        first = (centers[local - 1] + centers[local]) / 2.0
        second = (centers[local] + centers[local + 1]) / 2.0
        cell_lower, cell_upper = min(first, second), max(first, second)
        overlap_lower = max(lower, cell_lower)
        overlap_upper = min(upper, cell_upper)
        if overlap_upper <= overlap_lower:
            continue
        if latitude:
            weight = math.sin(math.radians(overlap_upper)) - math.sin(
                math.radians(overlap_lower)
            )
        else:
            weight = math.radians(overlap_upper - overlap_lower)
        if weight > 0.0:
            weights[index] = weight
    return weights


def _curvilinear_axis_overlaps(
    group, bounds: tuple[float, float, float, float]
) -> list[tuple[int, int, float]] | None:
    """Use separable 2-D coordinate arrays as regular geographic axes.

    Some CLM raw files store regular lat/lon centers as 2-D ``LATIXY`` and
    ``LONGXY`` arrays. Supporting their cell edges avoids falling back to a
    single nearest cell when the target crosses a source-cell boundary.
    ``None`` indicates a genuinely curvilinear grid.
    """
    south, north, west, east = bounds
    spatial = group.attrs["spatial"]
    records = _index_records(group, bounds)
    row_weights: dict[int, float] = {}
    column_weights: dict[int, float] = {}
    lat_name = spatial["latitude_variable"]
    lon_name = spatial["longitude_variable"]

    for record in records:
        r0, r1 = record["row_start"], record["row_stop"]
        c0, c1 = record["column_start"], record["column_stop"]
        lat_block = np.asarray(group[lat_name][r0:r1, c0:c1], dtype=np.float64)
        lon_block = np.asarray(group[lon_name][r0:r1, c0:c1], dtype=np.float64)
        reference_lon = (west + east) / 2.0
        lon_block = (
            reference_lon + ((lon_block - reference_lon + 180.0) % 360.0) - 180.0
        )
        if not np.allclose(
            lat_block, lat_block[:, :1], rtol=0.0, atol=1e-7
        ) or not np.allclose(lon_block, lon_block[:1, :], rtol=0.0, atol=1e-7):
            return None

        lat_start = max(0, r0 - 1)
        lat_stop = min(int(group.attrs["dimensions"][spatial["row_dimension"]]), r1 + 1)
        lat_centers = np.asarray(
            group[lat_name][lat_start:lat_stop, c0], dtype=np.float64
        )
        row_weights.update(
            _axis_weights_from_centers(
                lat_centers,
                lat_start,
                r0,
                r1,
                south,
                north,
                latitude=True,
            )
        )

        lon_start = max(0, c0 - 1)
        lon_stop = min(
            int(group.attrs["dimensions"][spatial["column_dimension"]]), c1 + 1
        )
        lon_centers = np.asarray(
            group[lon_name][r0, lon_start:lon_stop], dtype=np.float64
        )
        reference_lon = (west + east) / 2.0
        lon_centers = (
            reference_lon + ((lon_centers - reference_lon + 180.0) % 360.0) - 180.0
        )
        column_weights.update(
            _axis_weights_from_centers(
                lon_centers,
                lon_start,
                c0,
                c1,
                west,
                east,
                latitude=False,
            )
        )

    if not row_weights or not column_weights:
        raise ValueError(
            "source grid does not fully cover the target cell on both axes"
        )
    cells = [
        (row, column, lat_weight * lon_weight)
        for row, lat_weight in row_weights.items()
        for column, lon_weight in column_weights.items()
    ]
    total = sum(weight for _, _, weight in cells)
    target_area = math.radians(east - west) * (
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )
    if not math.isclose(
        total,
        target_area,
        rel_tol=0.0,
        abs_tol=max(target_area * 1e-4, 1e-14),
    ):
        raise ValueError("source grid does not fully cover the target cell")
    return [(row, column, weight / total) for row, column, weight in cells]


def _overlapping_cells(
    group, bounds: tuple[float, float, float, float]
) -> list[tuple[int, int, float]]:
    spatial = group.attrs.get("spatial")
    if spatial is None:
        raise ValueError("raw Zarr store has no spatial coordinate metadata")
    if spatial.get("curvilinear"):
        axis_overlaps = _curvilinear_axis_overlaps(group, bounds)
        if axis_overlaps is not None:
            return axis_overlaps
        return [_curvilinear_containing_cell(group, bounds)]
    return _regular_overlaps(group, bounds)


def _field_value(group, name: str, row: int, column: int) -> np.ndarray:
    spatial = group.attrs["spatial"]
    dimensions = group.attrs["variables"][name]["dimensions"]
    index = tuple(
        row
        if dim == spatial["row_dimension"]
        else column
        if dim == spatial["column_dimension"]
        else slice(None)
        for dim in dimensions
    )
    return np.asarray(group[name][index])


def _weighted_mean(
    group, field: str, cells: list[tuple[int, int, float]]
) -> np.ndarray:
    """Area-average a field while reading its local spatial window once."""
    spatial = group.attrs["spatial"]
    dimensions = group.attrs["variables"][field]["dimensions"]
    row_axis = dimensions.index(spatial["row_dimension"])
    column_axis = dimensions.index(spatial["column_dimension"])
    row_start = min(row for row, _, _ in cells)
    row_stop = max(row for row, _, _ in cells) + 1
    column_start = min(column for _, column, _ in cells)
    column_stop = max(column for _, column, _ in cells) + 1
    selection: list[slice] = [slice(None)] * len(dimensions)
    selection[row_axis] = slice(row_start, row_stop)
    selection[column_axis] = slice(column_start, column_stop)
    window = np.asarray(group[field][tuple(selection)], dtype=np.float64)
    result: np.ndarray | None = None
    for row, column, weight in cells:
        cell_index: list[slice | int] = [slice(None)] * len(dimensions)
        cell_index[row_axis] = row - row_start
        cell_index[column_axis] = column - column_start
        value = window[tuple(cell_index)]
        if result is None:
            result = np.zeros_like(value, dtype=np.float64)
        result += value * weight
    if result is None:
        raise ValueError(f"no source values found for {field}")
    return result


def _soil_texture(
    group, cells: list[tuple[int, int, float]]
) -> tuple[np.ndarray, np.ndarray]:
    by_mapunit: dict[int, float] = {}
    for row, column, weight in cells:
        mapunit = int(round(float(_field_value(group, "MAPUNITS", row, column))))
        by_mapunit[mapunit] = by_mapunit.get(mapunit, 0.0) + weight
    ranked = sorted(by_mapunit.items(), key=lambda item: item[1], reverse=True)
    mapunit = ranked[0][0]
    if mapunit == 0:
        mapunit = next((candidate for candidate, _ in ranked[1:] if candidate != 0), 0)
    if mapunit == 0:
        return np.full(10, 43.0), np.full(10, 18.0)

    sand_table = group["PCT_SAND"]
    clay_table = group["PCT_CLAY"]
    if mapunit < 1 or mapunit > sand_table.shape[1]:
        raise ValueError(f"soil map-unit index {mapunit} is outside the texture table")
    sand = np.asarray(sand_table[:, mapunit - 1], dtype=np.float64)
    clay = np.asarray(clay_table[:, mapunit - 1], dtype=np.float64)
    # The Fortran generator applies NINT to these positive percentages.
    return np.floor(sand + 0.5), np.floor(clay + 0.5)


def _dominant_soil_color(group, cells: list[tuple[int, int, float]]) -> int:
    weights: dict[int, float] = {}
    for row, column, area_weight in cells:
        color = int(round(float(_field_value(group, "SOIL_COLOR", row, column))))
        landmask = float(_field_value(group, "LANDMASK", row, column))
        color = color if landmask > 0 else 0
        weights[color] = weights.get(color, 0.0) + area_weight
    nonzero = {color: weight for color, weight in weights.items() if color > 0}
    if not nonzero:
        return 15
    return max(nonzero, key=nonzero.get)


def sample_point_soil_fields(
    cloud_root: str,
    bounds: tuple[float, float, float, float],
    *,
    config: S3Config | None = None,
) -> dict[str, Any]:
    """Sample the CLM soil fields used by surfdata for a single target cell.

    ``bounds`` is ``(south, north, west, east)`` in degrees. Regular source
    grids use spherical rectangle overlap weights. The 5-minute soil-depth
    product stores separable latitude and longitude axes as 2-D arrays; their
    cell edges are used to weight overlaps. A genuinely curvilinear fallback
    accepts only targets contained within one source cell.
    """
    bucket, prefix = parse_s3_uri(cloud_root)
    fs = (config or S3Config.from_env()).filesystem()
    bucket_prefix = f"{bucket}/{prefix.rstrip('/')}"

    def open_store(relative_path: str):
        return _open_group(fs, bucket_prefix, relative_path)

    texture = open_store(SOIL_STORES["texture"])
    texture_cells = _overlapping_cells(texture, bounds)
    sand, clay = _soil_texture(texture, texture_cells)

    organic = open_store(SOIL_STORES["organic"])
    organic_cells = _overlapping_cells(organic, bounds)
    organic_profile = _weighted_mean(organic, "ORGANIC", organic_cells)

    depth = open_store(SOIL_STORES["depth"])
    depth_cells = _overlapping_cells(depth, bounds)
    bedrock = float(_weighted_mean(depth, "Avg_Depth_Median", depth_cells))

    peat = open_store(SOIL_STORES["peat"])
    peat_cells = _overlapping_cells(peat, bounds)
    peat_fraction = float(_weighted_mean(peat, "peatf", peat_cells))

    fmax = open_store(SOIL_STORES["fmax"])
    fmax_cells = _overlapping_cells(fmax, bounds)
    max_saturated_fraction = float(_weighted_mean(fmax, "FMAX", fmax_cells))

    color = open_store(SOIL_STORES["color"])
    color_cells = _overlapping_cells(color, bounds)
    soil_color = _dominant_soil_color(color, color_cells)

    return {
        "PCT_SAND": sand,
        "PCT_CLAY": clay,
        "ORGANIC": organic_profile,
        "zbedrock": bedrock,
        "peatf": peat_fraction,
        "FMAX": max_saturated_fraction,
        "SOIL_COLOR": soil_color,
    }


def sample_point_canopy_fields(
    cloud_root: str,
    bounds: tuple[float, float, float, float],
    *,
    config: S3Config | None = None,
) -> dict[str, np.ndarray]:
    """Map the source LAI, SAI, and canopy-height rows onto one target cell.

    The source product contains 16 rows for PFT indices 0 through 15. The
    Fortran generator uses these as output rows 0 through 15; its separate
    irrigated C3-crop row (16) copies source row 15. Crop rows 17 onward have
    no satellite canopy input and remain zero.
    """
    bucket, prefix = parse_s3_uri(cloud_root)
    fs = (config or S3Config.from_env()).filesystem()
    group = _open_group(
        fs,
        f"{bucket}/{prefix.rstrip('/')}",
        CANOPY_STORE,
    )
    spatial = group.attrs["spatial"]
    spatial_dimensions = {
        spatial["row_dimension"],
        spatial["column_dimension"],
    }
    pft_count = int(group.attrs["dimensions"]["pft"])
    pft_indices = np.asarray(group["pft"][:], dtype=np.int64)
    if not np.array_equal(pft_indices, np.arange(pft_count)):
        raise ValueError("LAI source PFT coordinate must use zero-based CLM indices")
    if int(group.attrs["dimensions"]["time"]) != 12:
        raise ValueError("LAI source must contain 12 monthly records")

    cells = _overlapping_cells(group, bounds)
    mapped: dict[str, np.ndarray] = {}
    for output_name, source_name in CANOPY_FIELDS.items():
        dimensions = tuple(group.attrs["variables"][source_name]["dimensions"])
        value_dimensions = tuple(
            dimension for dimension in dimensions if dimension not in spatial_dimensions
        )
        if value_dimensions != ("time", "pft"):
            raise ValueError(
                f"{source_name} must have time,pft dimensions before spatial dimensions"
            )
        values = _weighted_mean(group, source_name, cells)
        if values.shape != (12, pft_count):
            raise ValueError(f"{source_name} mapped to unexpected shape {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source_name} contains non-finite mapped values")
        mapped[output_name] = values
    return mapped


def sample_point_ancillary_fields(
    cloud_root: str,
    bounds: tuple[float, float, float, float],
    *,
    config: S3Config | None = None,
) -> dict[str, Any]:
    """Sample remaining active, location-dependent fields for a one-cell file.

    Crop and natural-vegetation fractions are deliberately excluded because
    the point generator prescribes a monocrop. Harvest/LULCC arrays and
    fertilizer are retained from the CLM 2005 source product.
    """
    bucket, prefix = parse_s3_uri(cloud_root)
    fs = (config or S3Config.from_env()).filesystem()
    bucket_prefix = f"{bucket}/{prefix.rstrip('/')}"

    def open_store(name: str):
        return _open_group(fs, bucket_prefix, ANCILLARY_STORES[name])

    values: dict[str, Any] = {}

    vocef = open_store("vocef")
    vocef_cells = _overlapping_cells(vocef, bounds)
    for output_name, source_name in VOCEF_FIELDS.items():
        values[output_name] = float(_weighted_mean(vocef, source_name, vocef_cells))

    gdp = open_store("gdp")
    gdp_cells = _overlapping_cells(gdp, bounds)
    values["gdp"] = float(_weighted_mean(gdp, "gdp", gdp_cells))

    abm = open_store("abm")
    abm_cells = _overlapping_cells(abm, bounds)
    month_weights: dict[int, float] = {}
    for row, column, weight in abm_cells:
        month = int(round(float(_field_value(abm, "abm", row, column))))
        if 1 <= month <= 12:
            month_weights[month] = month_weights.get(month, 0.0) + weight
    values["abm"] = max(month_weights, key=month_weights.get) if month_weights else 13

    harvest = open_store("harvest")
    harvest_cells = _overlapping_cells(harvest, bounds)
    for output_name, source_name in HARVEST_FIELDS.items():
        values[output_name] = _weighted_mean(harvest, source_name, harvest_cells)
    for output_name, source_name in (
        ("CONST_FERTNITRO_CFT", "FERTNITRO_CFT"),
        ("UNREPRESENTED_PFT_LULCC", "UNREPRESENTED_PFT_LULCC"),
        ("UNREPRESENTED_CFT_LULCC", "UNREPRESENTED_CFT_LULCC"),
    ):
        values[output_name] = _weighted_mean(harvest, source_name, harvest_cells)

    for name, value in values.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"cloud field {name} contains non-finite values")
    return values
