# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import pickle

import pytest

from dynamo._core import MediaEmbedding

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_b10_media_embedding_exposes_length_bytes_and_repr():
    embedding = MediaEmbedding(b"packed")

    assert len(embedding) == 6
    assert embedding.to_bytes() == b"packed"
    assert repr(embedding) == "MediaEmbedding(6 bytes)"


def test_b10_media_embedding_cannot_be_pickled_or_deep_copied():
    embedding = MediaEmbedding(b"packed")

    with pytest.raises(TypeError):
        pickle.dumps(embedding)
    with pytest.raises(TypeError):
        copy.deepcopy(embedding)


IMAGE_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAFQAAAA4CAIAAADsP0i3AAAAJElEQVR4nO3BAQEAAACCIP+v"
    "bkhAAQAAAAAAAAAAAAAAAMCPATdYAAFXCTwrAAAAAElFTkSuQmCC"
)


@pytest.fixture
def encoder_client(monkeypatch):
    from dynamo._core import MultiModalEncoderClient

    for variable in ("BDN_PROXY", "ETCD_ENDPOINTS", "DYNAMO_DISABLE_BDN_PROXY"):
        monkeypatch.delenv(variable, raising=False)

    def make(url="http://127.0.0.1:1/encode", **limits):
        return MultiModalEncoderClient(
            url=url,
            api_key="test-key",
            max_retries=0,
            max_concurrent_requests=2,
            **limits,
        )

    return make


def image_request():
    return {
        "type": "image_url",
        "image_url": {"url": IMAGE_BASE64},
        "model": "served-model",
        "placeholder_token_id": 100,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("family", ["kimi_k27", "kimi_k3", "deepseek_v4", "qwen3_vl"])
async def test_native_pre_resolution_expands_without_encoder(encoder_client, family):
    import json

    import numpy as np

    config = {"model_type": family, "image_token_id": 100}
    processor = {}
    length = 6
    if family.startswith("kimi"):
        processor = {
            "media_proc_cfg": {
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
                "in_patch_limit": 65536,
                "patch_limit_on_one_side": 512,
            }
        }
    elif family == "deepseek_v4":
        config.update(
            vision_patch_size=14,
            vision_downsample_ratio=3,
            vision_max_n_token=1024,
            vision_min_pixels=295936,
        )
        length = 189
    else:
        processor = {
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 2,
            "min_pixels": 3136,
            "max_pixels": 1003520,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        }
    client = encoder_client()  # Unreachable encoder: native planning must stay local.
    arguments = {"model_config": config, "preprocessor_config": processor}
    result = await client.pre_resolve(
        np.array([1, 100, 2, 100, 3], dtype=np.uint32),
        [image_request(), image_request()],
        max_tokens=2 * length + 3,
        **arguments,
    )
    plan = result["token_plan"]
    assert plan["mm_lengths"] == [length, length]
    assert plan["mm_positions"] == [1, length + 2]
    assert plan["input_ids"] == [1] + [100] * length + [2] + [100] * length + [3]
    assert "data:" not in json.dumps(plan)
    assert IMAGE_BASE64 not in json.dumps(plan)
    assert all(
        item["request"]["image_url"]["url"] == "data:image/png;base64," + IMAGE_BASE64
        for item in result["prefill_media"]
    )
    if family == "qwen3_vl":
        assert tuple(plan["qwen_geometry"][0]["grid_thw"]) == (1, 4, 6)
        assert len(plan["mrope"]["position_ids"]) == 3 * len(plan["input_ids"])
    if family == "deepseek_v4":
        assert result["prefill_media"][0]["feature_token_count"] == 176
        assert len(plan["feature_ranges"][0]) == 11
        types = plan["image_token_types"][0]
        assert types[0] == 0 and types[-1] == 3
        assert types.count(1) == 176 and types.count(2) == 11
    with pytest.raises(Exception, match="limit|maximum|tokens"):
        await client.pre_resolve(
            [1, 100, 2], [image_request()], max_tokens=length + 1, **arguments
        )
    with pytest.raises(Exception, match="Placeholder|placeholder"):
        await client.pre_resolve(
            [1, 100, 100, 2], [image_request()], max_tokens=1000, **arguments
        )
    with pytest.raises(Exception, match="encoder-derived|Encoder-derived"):
        client.replay(result["prefill_media"])


@pytest.mark.asyncio
async def test_encoder_fallback_replay_validates_before_returning_embeddings(
    encoder_client,
):
    import asyncio
    import json

    import msgpack

    requests = []
    failures = []
    packed = b"packed-embedding"

    async def handle(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            size = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            request = json.loads(await reader.readexactly(size))
            requests.append(request)
            body = msgpack.packb(
                {
                    "success": True,
                    "mm_response": {
                        "mm_hash": "a" * 64,
                        "length": 6,
                        "mm_kwargs": packed,
                    },
                },
                use_bin_type=True,
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/msgpack\r\n"
                + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            await writer.drain()
        except (OSError, ValueError, asyncio.IncompleteReadError) as error:
            failures.append(error)
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = encoder_client(f"http://127.0.0.1:{port}/encode")
        resolved = await asyncio.wait_for(
            client.pre_resolve(
                [1, 100, 2],
                [image_request()],
                max_tokens=8,
                model_config={"model_type": "future-model"},
            ),
            timeout=10,
        )
        assert resolved["token_plan"]["mm_hashes"] == ["a" * 64]
        assert "mm_kwargs" not in json.dumps(resolved)
        replayed = await client.replay(resolved["prefill_media"])
        embedding = replayed["data"][0]["mm_response"]["mm_kwargs"]
        assert isinstance(embedding, MediaEmbedding)
        assert embedding.to_bytes() == packed
        altered = copy.deepcopy(resolved["prefill_media"])
        altered[0]["length"] = 7
        with pytest.raises(ValueError, match="metadata differs"):
            await client.replay(altered)
        assert len(requests) == 3
        assert requests[0] == requests[1] == requests[2]
        assert requests[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "placeholder_token_id" not in requests[0]
        assert not failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits, message",
    [
        ({"max_media_items": 1}, "media item limit"),
        ({"max_total_media_bytes": 1}, "aggregate request byte limit"),
        ({"max_image_pixels": 1}, "original pixel limit"),
    ],
)
async def test_pre_resolution_resource_limits(encoder_client, limits, message):
    client = encoder_client(**limits)
    with pytest.raises(Exception, match=message):
        await client.pre_resolve(
            [100, 100],
            [image_request(), image_request()],
            max_tokens=100,
            model_config={"model_type": "qwen3_vl", "image_token_id": 100},
            preprocessor_config={
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "min_pixels": 3136,
                "max_pixels": 1003520,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["video_url", "audio_url"])
async def test_native_qwen_rejects_mixed_media_before_network(encoder_client, kind):
    client = encoder_client()
    with pytest.raises(Exception, match="image-only prompts"):
        await client.pre_resolve(
            [100, 200],
            [
                image_request(),
                {
                    "type": kind,
                    kind: {"url": "http://localhost:1/never-fetch"},
                    "model": "served-model",
                    "placeholder_token_id": 200,
                },
            ],
            max_tokens=100,
            model_config={"model_type": "qwen3_vl", "image_token_id": 100},
            preprocessor_config={
                "patch_size": 14,
                "merge_size": 2,
                "temporal_patch_size": 2,
                "min_pixels": 3136,
                "max_pixels": 1003520,
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
            },
        )
