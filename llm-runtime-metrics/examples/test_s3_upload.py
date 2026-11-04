#!/usr/bin/env python3
"""Test S3 upload to mp-hashed-requests bucket"""

import time

from llm_runtime_metrics import REQUEST_FEATURE_NONE, RequestMetricsFactory

# Read credentials from files
with open("/secrets/aws-id", "r") as f:
    access_key_id = f.read().strip()

with open("/secrets/aws-key", "r") as f:
    secret_access_key = f.read().strip()

print("✓ Read credentials from /secrets/")
print(f"  Access Key ID: {access_key_id[:8]}...")
print()

# Create factory with S3 config
# Note: file_format can be "json" (default) or "parquet"
factory = RequestMetricsFactory(
    request_log_enabled=True,
    request_log_batch_size=1,  # Upload immediately for testing
    s3=True,
    s3_bucket="mp-hashed-requests",
    s3_region="us-east-1",
    s3_access_key_id=access_key_id,
    s3_secret_access_key=secret_access_key,
    s3_prefix="test-request",
    file_format="parquet",
    model_name="test-model-v1",
    block_size=32,  # for hashing.
)

print("✓ Factory created with S3 config")
print("  Bucket: mp-hashed-requests")
print("  Region: us-east-1")
print("  Prefix: test-request")
print("  File Format: parquet")
print()

# Create a test request
print("Creating test request...")
req = factory.new_request(
    input_token_ids=list(range(256)),
    features=REQUEST_FEATURE_NONE,
    session_id="test-session-123",
)

print("✓ Request created")
print()

# Record some tokens with cached_tokens
print("Recording tokens with cached_tokens...")
cached_tokens = 128
time.sleep(0.01)
req.record_tokens(
    token_ids=list(range(256, 257)), cached_tokens=cached_tokens, is_diff=True
)

print(f"✓ Prefill recorded (cached_tokens={cached_tokens}, added 1 token)")
for i in range(257, 288):
    time.sleep(0.005)
    req.record_tokens(token_ids=[i], cached_tokens=cached_tokens, is_diff=True)

# Mark success - this triggers S3 upload
print("Marking request as successful (triggers S3 upload)...")
req.success()

print("✓ Request completed - uploading to S3...")
print()

# Wait for async upload
print("Waiting 5 seconds for upload to complete...")
time.sleep(5)

print()
print("=" * 60)
print("DONE!")
print("=" * 60)
print()
print("Check your S3 bucket at:")
print("  s3://mp-hashed-requests/test-request/")
print()
print("JSON output format includes:")
print("  - timestamp_start_unix_ms: Start time")
print("  - duration_e2e_ms: End-to-end duration")
print("  - duration_prefill_ms: Time to first token")
print("  - cached_tokens_reference: Reference token count")
print()
print("To verify the upload, run:")
print("  aws s3 ls s3://mp-hashed-requests/test-request/ --profile production")
