# Point crop surface generator

This is a standalone `uv` project for generating one-cell eCLM domain and
surfdata NetCDF files. It samples soil, canopy, topography, and ancillary
fields from the configured cloud-optimized Zarr and GeoParquet stores, then
applies a single-crop prescription. The compatible surfdata template provides
the CLM schema and inactive parameter tables.

## Install and run

Install the project and its command-line entry point:

```bash
uv sync --project point_crop_surface_generator
```

Provide bounds, a global CLM CFT index, a compatible one-cell surfdata
template, and the cloud S3 environment variables. The project includes the
DE-RuS example schema template at
`src/point_crop_surface_generator/data/surfdata_template.nc`:

```bash
uv run --project point_crop_surface_generator point-crop-surface-generator \
  --surfdata-template point_crop_surface_generator/src/point_crop_surface_generator/data/surfdata_template.nc \
  --south 50.860906 --north 50.870906 \
  --west 6.4421445 --east 6.4521445 \
  --crop-type 21 --fertilizer-rate 30 --cloud-data-root s3://YOUR_BUCKET/eclm/cloud-optimized/
```

By default, output files go to `generated_static_files/`, named from the
bounds and CFT. A rerun replaces matching outputs. `--lai-csv` optionally
sets twelve monthly LAI values for the selected crop. Set
`ECLM_CLOUD_DATA_ROOT` or pass `--cloud-data-root` to use a different cloud
prefix. S3 settings use the standard AWS environment variables, including
`AWS_ENDPOINT_URL` or `AWS_S3_ENDPOINT` for compatible S3 services.

The cloud stores must already exist and cover the requested cell. The project
also owns the one-time preparation commands that build these stores from the
canonical CLM input-data tree. The downloader checks source file sizes before
replacing incomplete local copies. The converter writes Zarr stores and a
tiled HYDRO1K GeoParquet index directly to a local directory or S3.

For a source-to-store local run, first download the default raw fields and
mapping grids, then convert them:

```bash
uv run --project point_crop_surface_generator eclm-regional-download \
  --input-root /data/inputdata
uv run --project point_crop_surface_generator eclm-regional-prep \
  --input-root /data/inputdata --output-uri s3://YOUR_BUCKET/eclm/cloud-optimized/
```

The default profile prepares the twelve source grids used by the cloud
mapping profile. `--all-grids` includes every grid in the manifest;
`--lai-only` downloads or converts only the high-resolution LAI field and its
MODISv2 source grid. The prep command overwrites each matching store. For S3,
pass `--output-uri s3://YOUR_BUCKET/prefix` and configure AWS credentials plus an
endpoint when using an S3-compatible service. Raw and grid manifests ship
inside the project and can be overridden with `--raw-manifest` and
`--grid-manifest`.

When the LAI raw field is selected, preparation checks its global NetCDF
dimensions against the MODISv2 SCRIP grid before writing any stores. It then
writes a `manifest.json` summary at the output root. The raw and SCRIP stores
retain variable dimensions, shapes,
chunks, and skip metadata for unsupported variable-length strings. SCRIP
spatial bounds use cell corners so boundary cells remain discoverable.
HYDRO1K connectivity is normalized once for all chunks; incomplete faces are
kept in the mesh array but omitted from the GeoParquet lookup index.

The point generator reads the prepared stores directly. It does not create
regional NetCDF subsets for the legacy Fortran tools.

The standalone container includes the SVN client used by the downloader and
both preparation commands. Build it from this project directory with:

```bash
docker build \
  -t point-crop-surface-generator:latest \
  .
```

This image includes the downloader, converter, and point generator. Load it into
the local image store available to each workflow node. For a `kind` cluster:

```bash
kind load docker-image point-crop-surface-generator:latest
```


- [`../argo/eclm-regional-input-preparation-workflow.yaml`](../argo/eclm-regional-input-preparation-workflow.yaml)
  downloads canonical inputs and prepares the cloud-optimized stores.
- [`../argo/eclm-static-file-generator-workflow.yaml`](../argo/eclm-static-file-generator-workflow.yaml)
  reads those stores and generates the domain and surfdata files.

Both workflows default to `point-crop-surface-generator:latest`. Replace the
`YOUR_BUCKET` placeholder in their S3 URI parameters with the bucket containing
your prepared data. The static workflow also needs an Argo artifact repository
for its output archive. Configure the `s3-credentials` Kubernetes secret with
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`; if your S3 service needs a
custom endpoint, provide it through the runtime environment.

The preparation workflow uses a 250 GiB PVC for raw inputs, then writes the
cloud-optimized stores directly to S3. Its `refresh-rawdata` parameter defaults
to `true`, so it can bootstrap from the public CLM input-data source without a
cache artifact.

## Development

```bash
cd point_crop_surface_generator
uv sync --group test
uv run pytest
uvx ruff check .
uvx ruff format --check .
```

The cloud integration checks compare the DE-RuS example and exercise a second
AOI. With cloud access configured, run them with
`ECLM_RUN_CLOUD_INTEGRATION=1 uv run pytest -q tests/test_point_surface_integration.py`.
The DE-RuS comparison allows differences in `SLOPE` and `STD_ELEV`, whose
cloud source differs from the checked-in example.

The implementation and reusable helpers live under `src/`.
