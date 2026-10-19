"""Download canonical CLM raw fields and mapping grids from the UCAR SVN tree."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlparse

LOG = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata"
DEFAULT_RAW_MANIFEST = files("point_crop_surface_generator").joinpath(
    "data/rawdata_files.txt"
)
DEFAULT_GRID_MANIFEST = files("point_crop_surface_generator").joinpath(
    "data/clm_mappingfiles.txt"
)
LAI_RAW_MARKER = "mksrf_lai_histclm52deg005_earthstatmirca_2005.c190119.nc"
LAI_GRID_FILE = "SCRIPgrid_3minx3min_MODISv2_c190503.nc"

# These names mirror the default cloud source profile used by
# ``mkmapdata.generate_mapping``. ``--all-grids`` opts into the remaining
# source grids listed in clm_mappingfiles.txt.
DEFAULT_GRID_FILES = {
    "SCRIPgrid_0.5x0.5_AVHRR_c110228.nc",
    "SCRIPgrid_0.25x0.25_MODIS_c170321.nc",
    "SCRIPgrid_3minx3min_LandScan2004_c120517.nc",
    LAI_GRID_FILE,
    "SCRIPgrid_3minx3min_MODISwcspsea_c151020.nc",
    "SCRIPgrid_3x3_USGS_c120912.nc",
    "SCRIPgrid_5x5min_IGBP-GSDP_c110228.nc",
    "SCRIPgrid_5x5min_ISRIC-WISE_c111114.nc",
    "SCRIPgrid_5x5min_ORNL-Soil_c170630.nc",
    "SCRIPgrid_10x10min_nomask_c110228.nc",
    "SCRIPgrid_3minx3min_GLOBE-Gardner_c120922.nc",
    "UGRID_1km-merge-10min_HYDRO1K-merge-nomask_c130402.nc",
}


def _manifest_entries(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    if not entries:
        raise ValueError(f"manifest contains no paths: {path}")
    return entries


def _relative_path(entry: str, base_url: str) -> str:
    parsed = urlparse(entry)
    if parsed.scheme:
        base_path = urlparse(base_url).path.rstrip("/")
        if not parsed.path.startswith(f"{base_path}/"):
            raise ValueError(f"manifest URL is outside the configured source: {entry}")
        relative = parsed.path[len(base_path) + 1 :]
    else:
        relative = entry
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe path in manifest: {entry}")
    return path.as_posix()


def _remote_size(url: str) -> int:
    result = subprocess.run(
        ["svn", "info", "--xml", url],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        try:
            entry = ET.fromstring(result.stdout).find("entry")
            size = int(entry.attrib["size"])
        except (ET.ParseError, AttributeError, KeyError, ValueError):
            pass
        else:
            if size > 0:
                return size

    # Some SVN servers omit file sizes from `info --xml`. `svn ls -v` reports
    # the byte count as its third whitespace-separated column.
    listing = subprocess.run(
        ["svn", "ls", "-v", url],
        check=False,
        capture_output=True,
        text=True,
    )
    if listing.returncode == 0:
        for line in listing.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[0].isdigit() and fields[2].isdigit():
                size = int(fields[2])
                if size > 0:
                    return size
    detail = result.stderr.strip() or listing.stderr.strip()
    raise RuntimeError(f"could not determine source size for {url}: {detail}")


def _download_one(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".eclm-download-", dir=destination.parent
    ) as folder:
        temporary = Path(folder) / destination.name
        subprocess.run(["svn", "export", "--force", url, str(temporary)], check=True)
        actual_size = temporary.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"download size mismatch for {destination}: expected "
                f"{expected_size}, got {actual_size}"
            )
        os.replace(temporary, destination)


def download_manifest(
    manifest: Path,
    input_root: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    allowed_names: set[str] | None = None,
) -> list[Path]:
    """Download listed files after verifying source sizes, replacing bad copies."""
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    root = input_root.expanduser().resolve()
    base_url = base_url.rstrip("/")
    paths = [
        _relative_path(entry, base_url)
        for entry in _manifest_entries(manifest)
        if allowed_names is None or Path(urlparse(entry).path).name in allowed_names
    ]
    if not paths:
        raise ValueError(f"manifest has no selected source files: {manifest}")

    LOG.info("checking source sizes for %d files from %s", len(paths), manifest)
    sizes = [_remote_size(f"{base_url}/{relative}") for relative in paths]
    downloaded = []
    for relative, expected_size in zip(paths, sizes):
        destination = root / relative
        if destination.is_file() and destination.stat().st_size == expected_size:
            LOG.info("cached %s", relative)
            continue
        LOG.info("downloading %s (%d bytes)", relative, expected_size)
        _download_one(f"{base_url}/{relative}", destination, expected_size)
        downloaded.append(destination)
    LOG.info(
        "download complete: %d downloaded, %d already cached",
        len(downloaded),
        len(paths) - len(downloaded),
    )
    return downloaded


def download_sources(
    input_root: Path,
    *,
    raw_manifest: Path = Path(str(DEFAULT_RAW_MANIFEST)),
    grid_manifest: Path = Path(str(DEFAULT_GRID_MANIFEST)),
    base_url: str = DEFAULT_BASE_URL,
    all_grids: bool = False,
    lai_only: bool = False,
    skip_raw: bool = False,
    skip_grids: bool = False,
) -> None:
    if not skip_raw:
        allowed_raw = {Path(entry).name for entry in _manifest_entries(raw_manifest)}
        if lai_only:
            allowed_raw = {name for name in allowed_raw if LAI_RAW_MARKER in name}
        download_manifest(
            raw_manifest,
            input_root,
            base_url=base_url,
            allowed_names=allowed_raw,
        )
    if not skip_grids:
        allowed_grids = None
        if lai_only:
            allowed_grids = {LAI_GRID_FILE}
        elif not all_grids:
            allowed_grids = DEFAULT_GRID_FILES
        download_manifest(
            grid_manifest,
            input_root,
            base_url=base_url,
            allowed_names=allowed_grids,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download CLM raw input files and mapping grids from UCAR SVN"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--raw-manifest", type=Path, default=Path(str(DEFAULT_RAW_MANIFEST))
    )
    parser.add_argument(
        "--grid-manifest", type=Path, default=Path(str(DEFAULT_GRID_MANIFEST))
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--all-grids", action="store_true")
    parser.add_argument("--lai-only", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-grids", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = _parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    download_sources(
        args.input_root,
        raw_manifest=args.raw_manifest,
        grid_manifest=args.grid_manifest,
        base_url=args.base_url,
        all_grids=args.all_grids,
        lai_only=args.lai_only,
        skip_raw=args.skip_raw,
        skip_grids=args.skip_grids,
    )


if __name__ == "__main__":
    main()
