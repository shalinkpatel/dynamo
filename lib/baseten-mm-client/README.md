# baseten-mm-client

Rust-native HTTP multimodal encoder client. Uses
`baseten_performance_client_core` 0.1.13-post.1 for batching, retries, timeouts, proxying,
and cancellation. No Python runtime is needed.

`MediaResolver::resolve_image` resolves an HTTP(S) URL (including S3 presigned
URLs), a base64 data URI, or bare base64 without invoking the encoder. It returns
a canonical base64 data URI, header-inspected dimensions, and full BLAKE3 identity
of the compressed bytes. Only send the data URI to prefill; decode needs the
expanded prompt and planning metadata. This content hash is not the encoder's
model-specific `mm_hash`.

```rust,no_run
use baseten_mm_client::{MediaResolver, MediaResolverConfig};

# async fn example() -> anyhow::Result<()> {
let resolver = MediaResolver::new(MediaResolverConfig::default())?;
let image = resolver.resolve_image("https://bucket.s3.amazonaws.com/image.png?X-Amz-Signature=...").await?;
// Move image.data_url into the prefill request's image_url.url field.
# Ok(())
# }
```

The crate pins `dynamo-multimodal` from frontend-crates revision
`17020c2662a490f4366d1710db1971776d4e63b5`, following the same pinned upstream
library plus Baseten adapter pattern as `baseten-parsers`. It reuses upstream
image inspection, Qwen geometry, token-layout validation/expansion and image-only
M-RoPE. Upstream fetching is currently a stub, so the Baseten resolver supplies
BDN-aware fetching and Dynamo's URL policy. `MULTIMODAL_UPSTREAM_REVISION` and
`NATIVE_IMAGE_FAMILIES` expose the pin and native coverage.

`MediaResolver::pre_resolve` (or `MultiModalClient::pre_resolve`) resolves media
with parallel fetching while retaining prompt order. Pass one
`PlanningRequest` per original media placeholder, in prompt order, and the
unexpanded token IDs. It returns `PreResolvedRequest` with:

- `token_plan`: expanded IDs, positions, lengths, model-sensitive hashes, feature
  ranges and geometry. Qwen image-only plans also include M-RoPE position IDs and
  delta. This contains no media and can go to both prefill and decode.
- `prefill_media`: canonical base64 data URIs and the corresponding plans. Send
  this only to prefill, which performs feature extraction at the supplied spans.

| Image family | Planner | Checkpoint settings |
| --- | --- | --- |
| Kimi K2.7 (also K2.5, K2.6) | Baseten NaViT adapter | `media_proc_cfg`: patch size, merger, patch and side budgets, optional fixed output length |
| Kimi K3 / K3 Flash | NaViT plus checkpoint-tokenized media framing | Same geometry config and the serving checkpoint tokenizer |
| DSV4.1 (`deepseek_v4`, `deepseek_v41`) | Baseten resize adapter | `vision_*` geometry and token budget |
| Qwen VL / 3.5 | Pinned upstream processor | HF model and preprocessor configs |
| Other models, audio and video | Configured encoder metadata | Encoder-reported exact hash and length |

Kimi K3/K3 Flash adds the tokenized prefix
`<|media_begin|>image WIDTHxHEIGHT<|media_content|>` and suffix `<|media_end|>`
around each feature run, matching the
[pinned SGLang processor](https://github.com/sgl-project/sglang/blob/ffac53d779c08dcdab2d07e5e2a41dba83f0e65c/python/sglang/srt/multimodal/processors/kimi_k3.py).
The dimensions are the original image dimensions after EXIF orientation, even when NaViT resizes the
image. These control/text IDs are part of the expanded context limit;
`mm_positions`, `mm_lengths` and `feature_ranges` cover only the feature tokens,
as SGLang image offsets do. Kimi K2.7 retains its existing template framing.

Pass the checkpoint tokenizer as Python `pre_resolve(..., tokenizer=tokenizer)`
for native K3 planning. The binding uses `encode(text, allowed_special="all")`,
with the pinned processor's `encode(text)` fallback for tokenizers without that
keyword. Rust callers attach an existing serving tokenizer with
`resolver.with_media_tokenizer(Arc::new(...))`; the callback returns `Vec<i32>`
and must preserve media special tokens without adding unrelated BOS/EOS tokens.
The tokenizer is never serialized to workers. Missing tokenizers fail before
fetching/encoding rather than producing an incomplete K3 prompt. The prefill
record retains the framing IDs in `kimi3_framing`.

The native Kimi and DSV4.1 geometry matches SGLang revision
`ffac53d779c08dcdab2d07e5e2a41dba83f0e65c`. Native image planning inspects compressed
image headers without decoding pixels or executing a model. Settings come from
checkpoint configs; no guessed model defaults are used. DSV4.1 spans include
boundaries and row separators: `n_llm_h * (n_llm_w + 1) + 2`; the recorded
feature ranges identify aligner cells. `image_token_types` distinguishes START,
IMAGE, NEW_LINE and END (0/1/2/3); SGLang repeats `image_token_id` throughout the
span and uses these types to attach learned boundary/row features. Placeholder counts/order and the expanded
context limit are validated before returning a plan.

Python native-only callers can use `MediaPreprocessor()` without configuring an
encoder URL or API key. Its `pre_resolve` has the same planning arguments shown
below, and `resolve_image(source)` returns canonical base64, content hash and
EXIF-oriented dimensions. Reuse one instance across requests: resolver clones
share the HTTP pool and resolution semaphore. BDN provides download caching;
this API does not add a separate content or URL cache. Native-only planning
errors raise `ValueError`; unsupported families need an encoder client.

DSV4.1 accepts flat `vision_*` fields or nested `vision_config` fields
(`patch_size`, `downsample_ratio`, `max_image_tokens`, `min_pixels`,
`max_wh_ratio`). Header inspection honors JPEG, PNG and WebP EXIF orientation
without changing compressed bytes, content hashes or canonical media.

Python also exposes planning through the existing encoder client:

```python
from dynamo._core import MultiModalEncoderClient

client = MultiModalEncoderClient(
    url="http://encoder:8000/encode", api_key="secret",
    max_media_bytes=64 * 1024 * 1024,
)
resolved = await client.pre_resolve(
    input_ids,
    [{"type": "image_url", "image_url": {"url": source},
      "model": "served-model", "placeholder_token_id": image_token_id}],
    max_tokens=context_limit,
    model_config=config_json, preprocessor_config=preprocessor_json,
)
# Route using resolved["token_plan"], including the expanded input_ids.
# Attach resolved["prefill_media"] only to the prefill request.
```

Rust callers select native planners with `TokenPlannerSpec::from_hf_configs`.
Python does this automatically for image requests when configs are supplied.
Requests can also specify a serialized `planner` explicitly. Without a native
planner, the encoder client is required: it calls the encoder and retains only
metadata. The existing encoder endpoint still returns its full embedding body;
this fallback does not eliminate that frontend transfer or initial encoding.
Omit model configs to deliberately select encoder planning.

For encoder-derived plans, prefill can call Python `client.replay(prefill_media)`
or Rust `MultiModalClient::replay` with each item's `replay_request()`. Replay
validates the routed hash and length before returning embeddings; Python keeps
binary embeddings in move-only `MediaEmbedding` handles. Native plans cannot use
encoder replay because their identities are defined by this planner. The native
worker adapter must validate geometry and consume supplied spans without
expanding again. Connecting this API to frontend routing and adapting native
engine processors are separate integrations; this crate does not modify them.
Audio/video containers remain opaque and use encoder-derived planning. Native
Qwen planning supports image-only prompts: every media item must have a Qwen
image planner. Mixing native Qwen images with encoder-planned media (including
video/audio) is rejected before fetching or encoding because encoder hash/length
metadata does not include the geometry required for complete Qwen M-RoPE.

Media fetching uses 64 shared resolution slots per resolver/client by default,
including across clones and concurrent requests. `max_concurrent_resolutions`
controls this independently of encoder `max_concurrent_requests`. Planning
returns canonical base64 data URIs in `prefill_media[*].request`, so callers need no separate fetch step.

HTTP downloads are GETs through `BDN_PROXY` when configured. Explicit
`MediaResolverConfig::proxy` overrides the environment, and
`DYNAMO_DISABLE_BDN_PROXY` disables the environment proxy. Proxy URLs use the
same credential normalization as encoder calls. Encoder bearer credentials
are never sent to media sources. Inline inputs require no HTTP call. The resolver
does not apply the encoder-only production proxy requirement.

`MediaFetcher` vendors Dynamo's domain/IP/DNS policy without NIXL dependencies;
it checks each redirect as well as the initial URL. The trusted configured proxy
may resolve to private addresses; media-source validation remains active. Its
defaults honor `DYN_MM_ALLOW_INTERNAL=1`. Downloads and decoded base64 are limited to 64 MiB
of compressed bytes per media item by default (`max_bytes`). Pre-resolution also
limits each request to 256 media items and 256 MiB of aggregate compressed data
(`max_media_items`, `max_total_bytes`). The aggregate budget is charged before
retaining HTTP chunks or allocating base64-decoded buffers; inline decoding
reserves a conservative upper bound (at most two padding bytes extra). Base64
representation and transport/Python copies consume additional memory, so these
are compressed-data budgets, not exact process-memory limits. Original images
are limited to 64 million pixels before planning (`max_image_pixels`), independent
of the model's resized/token budget. Python exposes the corresponding
`max_total_media_bytes`, `max_media_items`, `max_image_pixels` and
`max_concurrent_resolutions` constructor options. HTTP errors omit source
URLs so presigned query credentials do not appear in error messages. Dropping
the resolve future cancels an in-flight download. The resolver does not own a
local cache; BDN controls HTTP caching.

```rust,no_run
use baseten_mm_client::{EncodeRequest, HttpEncoderConfig, Media, MultiModalClient};

# async fn example() -> anyhow::Result<()> {
let mut config = HttpEncoderConfig::new("http://encoder:8000/encode");
config.api_key = "secret".into();
let client = MultiModalClient::new(config)?;
let request = EncodeRequest::new(
    Media::Image {
        image_url: [("url".into(), "https://example.com/image.jpg".into())].into(),
    },
    "served-model",
);
let responses = client.encode(vec![request]).await?;
let response = &responses[0];
let packed_embeddings = response.mm_kwargs.bytes()?;
# Ok(())
# }
```

`encode` validates success envelopes and returns hash, positive token count, and
opaque embedding payload. Encoder rejections retain their status in
`EncoderRejection`; HTTP status failures can be inspected with `http_error_status`.
`call_batch` returns raw envelopes and HTTP timing/cache metrics for integrations
that own their error mapping and metrics registration.

`cache_urls` routes through the first cache and forwards the remaining hops via
the transparent-route header. `proxy` enables cacheable PUT requests. When unset,
the client reads `BDN_PROXY` unless `DYNAMO_DISABLE_BDN_PROXY` is `1` or `true`
(case-insensitive). BDN URLs without credentials get the standard BDN credentials;
existing credentials are preserved. An explicit `proxy` overrides the environment.
If `ETCD_ENDPOINTS` contains `.svc.localnet`, client construction fails unless a
proxy is configured or `DYNAMO_DISABLE_BDN_PROXY` is set. Production multimodal
encoder calls must pass through BDN; the disable switch is an emergency bypass.
Response order matches input order. Dropping the batch future cancels pending work.

`ReplayRequest` serializes the original encoder input together with its expected
hash and length. `replay` re-executes those inputs through the same transport and
rejects metadata mismatches before returning full payloads. Credentials and cache
routes stay in the receiving worker's client configuration, not in the request.

The input format is not an RCE defense. Header inspection still invokes parsers
(including native libjpeg-turbo for JPEG), and workers later decode the original
compressed bytes. Enforce decoded-byte, image/frame/duration and CPU/time limits
at those decoders, keep codecs patched and isolate media processing from
credentials, network and host filesystem access. Frontend pixel/byte limits do
not replace worker limits or ingress request/admission limits.

Through BDN, frontend DNS validation is a preflight check only: the proxy resolves
the destination independently. This integration relies on BDN's independent blocking of internal destinations
at connection time; equivalent egress restrictions are required for other proxies.
The client cannot guarantee this proxy-side policy or prevent that DNS-rebinding
race by checking DNS locally. Neither base64, MIME sniffing nor a content hash
certifies that image bytes are safe.
