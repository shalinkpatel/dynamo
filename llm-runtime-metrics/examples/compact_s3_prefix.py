#!/usr/bin/env python3
"""Compact many parquet request-log files from one S3 prefix into one local parquet file."""

from __future__ import annotations

import argparse

from llm_runtime_metrics import compact_s3_parquet_to_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and compact parquet files from an S3 prefix into one local parquet file."
    )
    parser.add_argument("--bucket", default="mp-hashed-requests", help="S3 bucket name")
    parser.add_argument("--prefix", default="gptoss", help="S3 prefix to compact")
    parser.add_argument(
        "--output-path", required=True, help="Local output parquet path"
    )
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument(
        "--endpoint", default=None, help="Optional S3 endpoint override"
    )
    parser.add_argument(
        "--access-key-id",
        default=None,
        help="AWS access key id or a filesystem path containing it",
    )
    parser.add_argument(
        "--secret-access-key",
        default=None,
        help="AWS secret access key or a filesystem path containing it",
    )
    parser.add_argument(
        "--max-concurrent-downloads",
        type=int,
        default=128,
        help="Maximum number of concurrent parquet downloads",
    )
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Whether to anonymize data by hashing token ids and session ids (not recommended for debugging)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        (
            f"Compacting parquet files from s3://{args.bucket}/{args.prefix} to {args.output_path}"
        )
    )
    result = compact_s3_parquet_to_local(
        s3_bucket=args.bucket,
        s3_region=args.region,
        s3_prefix=args.prefix,
        output_path=args.output_path,
        s3_endpoint=args.endpoint,
        s3_access_key_id=args.access_key_id,
        s3_secret_access_key=args.secret_access_key,
        max_concurrent_downloads=args.max_concurrent_downloads,
        anonymize=args.anonymize,
    )
    print(f"Compacted {result.source_objects} parquet objects")
    print(f"Wrote {result.rows_written} rows")
    print(f"Output: {result.output_path}")


if __name__ == "__main__":
    main()
