"""Create a one-cell crop surfdata file from a compatible NetCDF template.

Generate one-cell CLM surface and domain files from AOI bounds and a crop CFT.
Cloud data is used by default. The input surfdata file supplies the CLM schema
and version-specific inactive urban/glacier parameter tables; active
location-dependent fields are sampled from cloud Zarr.

Example:
    uv run --project point_crop_surface_generator point-crop-surface-generator \
        --surfdata-template input_clm/surfdata_1x1_example.nc \
        --south 50.860906 --north 50.870906 \
        --west 6.4421445 --east 6.4521445 \
        --crop-type 21 --fertilizer-rate 30

Output files use coordinate-based names in ``generated_static_files/`` and
are replaced when the same AOI is generated.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import tempfile
from pathlib import Path

import netCDF4
import numpy as np

from .point_soil import (
    sample_point_ancillary_fields,
    sample_point_canopy_fields,
    sample_point_soil_fields,
)
from .point_topography import sample_point_topography_fields

C3_GRASS_NATPFT = 13
EARTH_RADIUS_KM = 6371.22


def _read_monthly_lai(path: Path) -> np.ndarray:
    """Read 12 LAI values or 12 ``month,lai`` records from a CSV file."""
    rows: list[list[float]] = []
    skipped_header = False
    with path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            cells = [cell.strip() for cell in row]
            if not cells or not any(cells) or cells[0].startswith("#"):
                continue
            try:
                values = [float(cell) for cell in cells if cell]
            except ValueError as exc:
                if not rows and not skipped_header:
                    skipped_header = True
                    continue
                raise ValueError(
                    f"{path}:{line_number}: expected numeric LAI values"
                ) from exc
            if len(values) != len(cells):
                raise ValueError(f"{path}:{line_number}: empty CSV field")
            rows.append(values)

    if len(rows) == 1 and len(rows[0]) == 12:
        lai = np.asarray(rows[0], dtype=np.float64)
    elif len(rows) == 12 and all(len(row) == 1 for row in rows):
        lai = np.asarray([row[0] for row in rows], dtype=np.float64)
    elif len(rows) == 12 and all(len(row) == 2 for row in rows):
        month_values = np.asarray([row[0] for row in rows])
        months = month_values.astype(np.int64)
        if not np.array_equal(month_values, months) or sorted(months.tolist()) != list(
            range(1, 13)
        ):
            raise ValueError(
                f"{path}: month column must contain each month 1 through 12 once"
            )
        lai = np.empty(12, dtype=np.float64)
        for month, value in rows:
            lai[int(month) - 1] = value
    else:
        raise ValueError(
            f"{path}: expected 12 monthly LAI values (one column) or 12 month,LAI rows"
        )

    if not np.all(np.isfinite(lai)) or np.any(lai < 0):
        raise ValueError(f"{path}: LAI values must be finite and non-negative")
    return lai


def _write_if_changed(variable: netCDF4.Variable, values: np.ndarray) -> bool:
    """Avoid rewriting values that already match the requested prescription."""
    desired = np.asarray(values, dtype=variable.dtype)
    current = np.asarray(variable[:])
    if current.shape != desired.shape:
        raise ValueError(
            f"{variable.name}: expected shape {current.shape}, got {desired.shape}"
        )
    if np.array_equal(current, desired):
        return False
    variable[:] = desired
    return True


def _temporary_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".nc",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        shutil.copy2(source, temporary_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _prescribe_crop(
    surfdata_path: Path,
    crop_type: int,
    monthly_lai: np.ndarray | None,
    fertilizer_rate: float | None,
) -> list[str]:
    changed: list[str] = []
    with netCDF4.Dataset(surfdata_path, "r+") as ds:
        for dimension in ("lsmlat", "lsmlon"):
            if dimension not in ds.dimensions or len(ds.dimensions[dimension]) != 1:
                raise ValueError(
                    "surfdata template must have exactly one lsmlat and lsmlon cell"
                )

        required = (
            "cft",
            "natpft",
            "PCT_CROP",
            "PCT_NATVEG",
            "PCT_LAKE",
            "PCT_WETLAND",
            "PCT_GLACIER",
            "PCT_GLC_MEC",
            "PCT_URBAN",
            "PCT_NAT_PFT",
            "PCT_CFT",
            "CONST_FERTNITRO_CFT",
            "LANDFRAC_PFT",
            "PFTDATA_MASK",
        )
        missing = [name for name in required if name not in ds.variables]
        if missing:
            raise ValueError(
                f"surfdata template is missing variables: {', '.join(missing)}"
            )

        cft_indices = np.asarray(ds.variables["cft"][:], dtype=np.int64)
        crop_rows = np.flatnonzero(cft_indices == crop_type)
        if crop_rows.size != 1:
            raise ValueError(
                f"CFT {crop_type} must occur exactly once in the cft coordinate"
            )
        crop_row = int(crop_rows[0])

        natpft_indices = np.asarray(ds.variables["natpft"][:], dtype=np.int64)
        grass_rows = np.flatnonzero(natpft_indices == C3_GRASS_NATPFT)
        if grass_rows.size != 1:
            raise ValueError(
                f"natpft index {C3_GRASS_NATPFT} (C3 grass) is not present"
            )

        zero_fractions = (
            "PCT_NATVEG",
            "PCT_LAKE",
            "PCT_WETLAND",
            "PCT_GLACIER",
            "PCT_URBAN",
        )
        for name in zero_fractions:
            variable = ds.variables[name]
            if _write_if_changed(variable, np.zeros(variable.shape)):
                changed.append(name)

        crop_fraction = ds.variables["PCT_CROP"]
        if _write_if_changed(crop_fraction, np.full(crop_fraction.shape, 100.0)):
            changed.append("PCT_CROP")

        natural_pft = np.zeros(ds.variables["PCT_NAT_PFT"].shape)
        natural_pft[int(grass_rows[0]), ...] = 100.0
        if _write_if_changed(ds.variables["PCT_NAT_PFT"], natural_pft):
            changed.append("PCT_NAT_PFT")

        crop_pft = np.zeros(ds.variables["PCT_CFT"].shape)
        crop_pft[crop_row, ...] = 100.0
        if _write_if_changed(ds.variables["PCT_CFT"], crop_pft):
            changed.append("PCT_CFT")

        for name in ("LANDFRAC_PFT", "PFTDATA_MASK"):
            variable = ds.variables[name]
            desired = np.ones(variable.shape, dtype=variable.dtype)
            if _write_if_changed(variable, desired):
                changed.append(name)

        if fertilizer_rate is not None:
            variable = ds.variables["CONST_FERTNITRO_CFT"]
            desired = np.asarray(variable[:])
            desired[crop_row, ...] = fertilizer_rate
            if _write_if_changed(variable, desired):
                changed.append("CONST_FERTNITRO_CFT")

        if monthly_lai is not None:
            if "MONTHLY_LAI" not in ds.variables:
                raise ValueError("surfdata template has no MONTHLY_LAI variable")
            lai_variable = ds.variables["MONTHLY_LAI"]
            expected_dimensions = ("time", "lsmpft", "lsmlat", "lsmlon")
            if lai_variable.dimensions != expected_dimensions:
                raise ValueError(
                    "MONTHLY_LAI dimensions must be " + ", ".join(expected_dimensions)
                )
            if (
                len(ds.dimensions["time"]) != 12
                or len(ds.dimensions["lsmpft"]) <= crop_type
            ):
                raise ValueError(
                    "template must have 12 months and an lsmpft row "
                    "for the selected CFT"
                )
            desired_lai = np.asarray(lai_variable[:])
            # CLM's combined lsmpft axis uses the global PFT/CFT index; this
            # differs from the shorter cft coordinate used by PCT_CFT.
            desired_lai[:, crop_type, 0, 0] = monthly_lai
            if _write_if_changed(lai_variable, desired_lai):
                changed.append("MONTHLY_LAI")

    return changed


def _normalize_bounds(
    south: float, north: float, west: float, east: float
) -> tuple[float, float, float, float]:
    values = np.asarray((south, north, west, east), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("AOI bounds must be finite numbers")
    if south < -90.0 or north > 90.0 or south >= north:
        raise ValueError("latitude bounds must satisfy -90 <= south < north <= 90")
    if 0.0 <= west < east <= 360.0 and east > 180.0:
        west = ((west + 180.0) % 360.0) - 180.0
        east = ((east + 180.0) % 360.0) - 180.0
    if west < -180.0 or east > 180.0 or west >= east:
        raise ValueError(
            "longitude bounds must satisfy -180 <= west < east <= 180; "
            "AOIs crossing the antimeridian are not supported"
        )
    return float(south), float(north), float(west), float(east)


def _spherical_area(bounds: tuple[float, float, float, float]) -> float:
    south, north, west, east = bounds
    return math.radians(east - west) * (
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )


def _coordinate_slug(value: float) -> str:
    return f"{value:.8f}".replace("-", "m").replace(".", "p")


def _output_paths(args, bounds: tuple[float, float, float, float]):
    south, north, west, east = bounds
    slug = (
        f"s{_coordinate_slug(south)}_n{_coordinate_slug(north)}_"
        f"w{_coordinate_slug(west)}_e{_coordinate_slug(east)}_"
        f"cft{args.crop_type}"
    )
    surfdata = args.output_surfdata or (args.output_dir / f"surfdata_1x1_{slug}.nc")
    domain = args.output_domain or (args.output_dir / f"domain.lnd_{slug}.nc")
    return surfdata.resolve(), domain.resolve()


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".nc",
        dir=destination.parent,
        delete=False,
    ) as handle:
        return Path(handle.name)


def _write_domain(path: Path, bounds: tuple[float, float, float, float]) -> None:
    south, north, west, east = bounds
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.title = "CESM domain data:"
        ds.Conventions = "CF-1.0"
        ds.source = "generated by point-crop-surface-generator"
        ds.history = (
            "single-cell agricultural domain; land mask and fraction set to one"
        )
        for name, size in (("n", 1), ("ni", 1), ("nj", 1), ("nv", 4)):
            ds.createDimension(name, size)

        xc = ds.createVariable("xc", "f8", ("nj", "ni"))
        xc.long_name = "longitude of grid cell center"
        xc.units = "degrees_east"
        xc.bounds = "xv"
        xc[:] = center_lon

        yc = ds.createVariable("yc", "f8", ("nj", "ni"))
        yc.long_name = "latitude of grid cell center"
        yc.units = "degrees_north"
        yc.bounds = "yv"
        yc[:] = center_lat

        xv = ds.createVariable("xv", "f8", ("nj", "ni", "nv"))
        xv.long_name = "longitude of grid cell vertices"
        xv.units = "degrees_east"
        xv[0, 0, :] = (west, east, east, west)

        yv = ds.createVariable("yv", "f8", ("nj", "ni", "nv"))
        yv.long_name = "latitude of grid cell vertices"
        yv.units = "degrees_north"
        yv[0, 0, :] = (south, south, north, north)

        mask = ds.createVariable("mask", "i4", ("nj", "ni"))
        mask.long_name = "domain mask"
        mask.note = "unitless"
        mask.coordinates = "xc yc"
        mask.comment = "0 value indicates cell is not active"
        mask[:] = 1

        area = ds.createVariable("area", "f8", ("nj", "ni"))
        area.long_name = "area of grid cell in radians squared"
        area.coordinates = "xc yc"
        area.units = "radian2"
        area[:] = _spherical_area(bounds)

        fraction = ds.createVariable("frac", "f8", ("nj", "ni"))
        fraction.long_name = "fraction of grid cell that is active"
        fraction.coordinates = "xc yc"
        fraction.units = "unitless"
        fraction[:] = 1.0


def _apply_grid_location(
    surfdata_path: Path, bounds: tuple[float, float, float, float]
) -> list[str]:
    south, north, west, east = bounds
    values = {
        "LATIXY": (south + north) / 2.0,
        "LONGXY": (west + east) / 2.0,
        "AREA": _spherical_area(bounds) * EARTH_RADIUS_KM**2,
    }
    changed: list[str] = []
    with netCDF4.Dataset(surfdata_path, "r+") as ds:
        for name, value in values.items():
            if name not in ds.variables:
                raise ValueError(f"surfdata template is missing {name}")
            variable = ds.variables[name]
            if variable.shape != (1, 1):
                raise ValueError(f"{name} must have one lsmlat by one lsmlon cell")
            if _write_if_changed(variable, np.full(variable.shape, value)):
                changed.append(name)
    return changed


def _apply_cloud_fields(surfdata_path: Path, values: dict[str, object]) -> list[str]:
    changed: list[str] = []
    with netCDF4.Dataset(surfdata_path, "r+") as ds:
        for name, value in values.items():
            if name not in ds.variables:
                raise ValueError(f"surfdata template is missing cloud field {name}")
            variable = ds.variables[name]
            desired = np.asarray(value)
            # Point samplers return the source's non-spatial dimensions only;
            # the surfdata template appends its singleton lsmlat/lsmlon axes.
            # Require those leading axes to agree instead of accepting any
            # arbitrary same-sized array and silently scrambling its meaning.
            leading_shape = variable.shape[: desired.ndim]
            trailing_shape = variable.shape[desired.ndim :]
            if (
                desired.ndim > variable.ndim
                or leading_shape != desired.shape
                or any(size != 1 for size in trailing_shape)
            ):
                raise ValueError(
                    f"cloud field {name} has shape {desired.shape}; "
                    f"template variable has incompatible shape {variable.shape}"
                )
            if _write_if_changed(variable, desired.reshape(variable.shape)):
                changed.append(name)
    return changed


def _apply_cloud_crop_fertilizer(
    surfdata_path: Path, crop_type: int, source_values: object
) -> list[str]:
    """Apply the cloud-derived rate only to the selected, active CFT row.

    Fertilizer rates for inactive CFTs are not relevant to a monocrop file and
    may contain case-specific management overrides in the template.
    """
    changed: list[str] = []
    with netCDF4.Dataset(surfdata_path, "r+") as ds:
        cft_indices = np.asarray(ds.variables["cft"][:], dtype=np.int64)
        crop_rows = np.flatnonzero(cft_indices == crop_type)
        if crop_rows.size != 1:
            raise ValueError(
                f"CFT {crop_type} must occur exactly once in the cft coordinate"
            )
        variable = ds.variables["CONST_FERTNITRO_CFT"]
        expected_dimensions = ("cft", "lsmlat", "lsmlon")
        if variable.dimensions != expected_dimensions:
            raise ValueError(
                "CONST_FERTNITRO_CFT dimensions must be "
                + ", ".join(expected_dimensions)
            )
        rates = np.asarray(source_values, dtype=np.float64)
        if rates.shape != cft_indices.shape or variable.shape[0] != cft_indices.size:
            raise ValueError(
                "cloud fertilizer rates must match the surfdata cft coordinate"
            )
        crop_row = int(crop_rows[0])
        desired = np.asarray(variable[:])
        desired[crop_row, ...] = rates[crop_row]
        if _write_if_changed(variable, desired):
            changed.append("CONST_FERTNITRO_CFT")
    return changed


def _apply_cloud_canopy(
    surfdata_path: Path, source_fields: dict[str, np.ndarray]
) -> list[str]:
    """Apply mapped source PFT fields using CLM's combined PFT/CFT row order."""
    source_crop_pft = 15
    irrigated_crop_pft = 16
    changed: list[str] = []
    with netCDF4.Dataset(surfdata_path, "r+") as ds:
        expected_dimensions = ("time", "lsmpft", "lsmlat", "lsmlon")
        for name, source_values in source_fields.items():
            if name not in ds.variables:
                raise ValueError(f"surfdata template is missing canopy field {name}")
            variable = ds.variables[name]
            if variable.dimensions != expected_dimensions:
                raise ValueError(
                    f"{name} dimensions must be " + ", ".join(expected_dimensions)
                )
            if len(ds.dimensions["time"]) != 12:
                raise ValueError("surfdata template must have 12 monthly records")
            if any(
                len(ds.dimensions[dimension]) != 1 for dimension in ("lsmlat", "lsmlon")
            ):
                raise ValueError("canopy mapping requires a one-cell surfdata template")

            values = np.asarray(source_values, dtype=np.float64)
            if values.ndim != 2 or values.shape[0] != 12:
                raise ValueError(f"{name} source values must have shape (12, pft)")
            output_pft_count = len(ds.dimensions["lsmpft"])
            if (
                values.shape[1] <= source_crop_pft
                or output_pft_count <= irrigated_crop_pft
            ):
                raise ValueError(f"{name} template does not include CLM crop PFT rows")
            if values.shape[1] > output_pft_count:
                raise ValueError(f"{name} source has more PFT rows than the template")

            desired = np.zeros(variable.shape, dtype=np.float64)
            desired[:, : values.shape[1], 0, 0] = values
            desired[:, irrigated_crop_pft, 0, 0] = values[:, source_crop_pft]
            if _write_if_changed(variable, desired):
                changed.append(name)
    return changed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surfdata-template",
        type=Path,
        required=True,
        help="compatible one-cell CLM surfdata file used as the output schema",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("generated_static_files")
    )
    parser.add_argument("--output-surfdata", type=Path)
    parser.add_argument("--output-domain", type=Path)
    parser.add_argument(
        "--south",
        type=float,
        required=True,
        help="southern latitude of the single grid cell, in degrees",
    )
    parser.add_argument(
        "--north",
        type=float,
        required=True,
        help="northern latitude of the single grid cell, in degrees",
    )
    parser.add_argument(
        "--west",
        type=float,
        required=True,
        help="western longitude of the single grid cell, in degrees",
    )
    parser.add_argument(
        "--east",
        type=float,
        required=True,
        help="eastern longitude of the single grid cell, in degrees",
    )
    parser.add_argument(
        "--crop-type",
        type=int,
        required=True,
        help="global CLM CFT index, e.g. 21 for rainfed winter wheat",
    )
    parser.add_argument(
        "--lai-csv",
        type=Path,
        help=(
            "12 monthly LAI values, or month,LAI rows; updates only "
            "the crop's MONTHLY_LAI"
        ),
    )
    parser.add_argument(
        "--cloud-data-root",
        default=os.environ.get("ECLM_CLOUD_DATA_ROOT"),
        help="cloud-optimized CLM raw-data root (or set ECLM_CLOUD_DATA_ROOT)",
    )
    parser.add_argument(
        "--fertilizer-rate",
        type=float,
        help=(
            "override the cloud-derived annual fertilizer rate for the selected "
            "crop, in gN/m2/yr"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.cloud_data_root:
        parser.error("pass --cloud-data-root or set ECLM_CLOUD_DATA_ROOT")
    try:
        bounds = _normalize_bounds(args.south, args.north, args.west, args.east)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.surfdata_template.is_file():
        raise SystemExit(f"surfdata template does not exist: {args.surfdata_template}")
    if args.fertilizer_rate is not None and (
        not math.isfinite(args.fertilizer_rate) or args.fertilizer_rate < 0.0
    ):
        raise SystemExit("--fertilizer-rate must be finite and non-negative")

    output_surfdata, output_domain = _output_paths(args, bounds)
    if output_surfdata == args.surfdata_template.resolve():
        raise SystemExit("output surfdata path must differ from the template")
    if output_domain == args.surfdata_template.resolve():
        raise SystemExit("output domain path must differ from the surfdata template")
    if output_domain == output_surfdata:
        raise SystemExit("surfdata and domain output paths must differ")

    monthly_lai = None
    if args.lai_csv is not None:
        if not args.lai_csv.is_file():
            raise SystemExit(f"LAI CSV does not exist: {args.lai_csv}")
        try:
            monthly_lai = _read_monthly_lai(args.lai_csv)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    temporary_surfdata: Path | None = None
    temporary_domain: Path | None = None
    cloud_fields: list[str] = []
    mean_elevation: float | None = None
    stage = "initializing"
    try:
        stage = "copying surfdata template"
        temporary_surfdata = _temporary_copy(args.surfdata_template, output_surfdata)
        temporary_domain = _temporary_path(output_domain)
        changed: list[str] = []
        stage = "sampling soil fields"
        soil_values = sample_point_soil_fields(args.cloud_data_root, bounds)
        stage = "sampling canopy fields"
        canopy_values = sample_point_canopy_fields(args.cloud_data_root, bounds)
        stage = "sampling ancillary fields"
        ancillary_values = sample_point_ancillary_fields(args.cloud_data_root, bounds)
        cloud_fertilizer_rates = ancillary_values.pop("CONST_FERTNITRO_CFT")
        stage = "sampling topography fields"
        topography_values = sample_point_topography_fields(args.cloud_data_root, bounds)
        mean_elevation = topography_values.pop("MEAN_ELEVATION")
        cloud_fields = [
            *soil_values,
            *canopy_values,
            *ancillary_values,
            "CONST_FERTNITRO_CFT",
            *topography_values,
        ]
        scalar_values = {
            **soil_values,
            **ancillary_values,
            **topography_values,
        }
        stage = "writing surfdata variables"
        changed.extend(_apply_cloud_fields(temporary_surfdata, scalar_values))
        changed.extend(
            _apply_cloud_crop_fertilizer(
                temporary_surfdata, args.crop_type, cloud_fertilizer_rates
            )
        )
        changed.extend(_apply_cloud_canopy(temporary_surfdata, canopy_values))
        changed.extend(_apply_grid_location(temporary_surfdata, bounds))
        # Apply a site LAI climatology after the mapped cloud fields so it
        # remains the selected crop's prescription. A rate override likewise
        # follows the cloud-mapped historical crop fertilization.
        changed.extend(
            _prescribe_crop(
                temporary_surfdata,
                args.crop_type,
                monthly_lai,
                args.fertilizer_rate,
            )
        )
        stage = "writing the domain file"
        _write_domain(temporary_domain, bounds)
        stage = "replacing output files"
        os.replace(temporary_surfdata, output_surfdata)
        temporary_surfdata = None
        os.replace(temporary_domain, output_domain)
        temporary_domain = None
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(f"cannot create static files while {stage}: {exc}") from exc
    finally:
        if temporary_surfdata is not None:
            temporary_surfdata.unlink(missing_ok=True)
        if temporary_domain is not None:
            temporary_domain.unlink(missing_ok=True)

    print(f"surfdata: {output_surfdata}")
    print(f"domain: {output_domain}")
    print(
        f"AOI bounds: south={bounds[0]}, north={bounds[1]}, "
        f"west={bounds[2]}, east={bounds[3]}"
    )
    print(f"crop CFT: {args.crop_type}")
    print(
        "updated variables: "
        + (", ".join(changed) if changed else "none (already matches prescription)")
    )
    if args.lai_csv is not None:
        print(f"monthly LAI source: {args.lai_csv}")
    print(f"cloud source: {args.cloud_data_root}")
    print("cloud fields: " + ", ".join(cloud_fields))
    print(
        "area-weighted mean elevation (diagnostic; not stored in surfdata): "
        f"{mean_elevation:.3f} m"
    )
    if args.fertilizer_rate is not None:
        print(f"fertilizer override: {args.fertilizer_rate:g} gN/m2/yr")
    print("existing output files are overwritten (each replacement is atomic)")
    return 0
