FROM ghcr.io/astral-sh/uv:0.12-debian AS uv

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates subversion \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY src/point_crop_surface_generator/data /opt/eclm-inputdata
RUN uv sync --locked --no-dev

ENTRYPOINT ["uv", "run", "--no-dev", "point-crop-surface-generator"]
