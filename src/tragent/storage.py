"""Files in the S3-compatible object store: research artifacts, uploaded statements, and exports.

Every machine running the API or a worker reads and writes the same bucket, so a report a worker
writes on one machine is the report the API serves on another. Locally the store is SeaweedFS.
"""

from __future__ import annotations

from functools import cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from tragent.config import load_settings


def put(
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
    download_name: str | None = None,
) -> None:
    """Store an object; one with a download name is saved under that name when downloaded."""

    client, bucket = _bucket()
    extra = {}
    if download_name is not None:
        # Kept with the object, since SeaweedFS rejects it as a presigned link's parameter.
        extra["ContentDisposition"] = f'attachment; filename="{download_name}"'
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type, **extra)


def get(key: str) -> bytes:
    client, bucket = _bucket()
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def exists(key: str) -> bool:
    client, bucket = _bucket()
    try:
        client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response["Error"]["Code"] == "404":
            return False
        raise
    return True


def download_url(key: str, seconds: int = 600) -> str:
    """A link that downloads the object straight from the store, without passing through the API."""

    _client, bucket = _bucket()
    return _signer().generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=seconds
    )


@cache
def _bucket() -> tuple[object, str]:
    """The process's one client, with its bucket created on first use."""

    settings = load_settings()
    client = _client(str(settings.s3_endpoint_url))
    try:
        client.create_bucket(Bucket=settings.s3_bucket)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    return client, settings.s3_bucket


@cache
def _signer() -> object:
    """A client that signs links for the address browsers use, which a proxy may change."""

    settings = load_settings()
    return _client(str(settings.s3_public_url or settings.s3_endpoint_url))


def _client(endpoint_url: str) -> object:
    settings = load_settings()
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
        region_name="us-east-1",
        # The bucket stays in the path, which is what the proxy routes on.
        config=Config(s3={"addressing_style": "path"}),
    )


__all__ = ["download_url", "exists", "get", "put"]
