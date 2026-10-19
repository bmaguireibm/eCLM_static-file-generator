"""Small S3 adapter using conventional AWS environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class S3Config:
    endpoint_url: str | None
    region_name: str | None
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None

    @classmethod
    def from_env(cls) -> "S3Config":
        return cls(
            endpoint_url=_first_env(
                "AWS_ENDPOINT_URL",
                "AWS_ENDPOINT_URL_S3",
                "AWS_S3_ENDPOINT",
                "S3_ENDPOINT_URL",
                "S3_ENDPOINT",
            ),
            region_name=_first_env("AWS_REGION", "AWS_DEFAULT_REGION"),
            access_key_id=_first_env("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY"),
            secret_access_key=_first_env("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_KEY"),
            session_token=os.environ.get("AWS_SESSION_TOKEN"),
        )

    def filesystem(self):
        """Return an fsspec S3 filesystem configured from this object."""
        import s3fs

        client_kwargs: dict[str, str] = {}
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            client_kwargs["region_name"] = self.region_name
        kwargs: dict[str, object] = {"client_kwargs": client_kwargs}
        if self.access_key_id:
            kwargs["key"] = self.access_key_id
        if self.secret_access_key:
            kwargs["secret"] = self.secret_access_key
        if self.session_token:
            kwargs["token"] = self.session_token
        return s3fs.S3FileSystem(**kwargs)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")
