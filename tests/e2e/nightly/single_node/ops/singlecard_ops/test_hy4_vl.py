# SPDX-License-Identifier: Apache-2.0
import socket

import pytest
import torch
import torch_npu  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.ops.rotary_embedding import gather_mrope_cache
from vllm_ascend.patch.hy4_vl.config import HYV4VLVisionConfig
from vllm_ascend.patch.hy4_vl.processor import vision_token_count
from vllm_ascend.patch.hy4_vl.random_weights import initialize_missing_vision_weights
from vllm_ascend.patch.hy4_vl.vision import HYV4VisionTransformer


@pytest.fixture(scope="module")
def distributed_vision():
    torch.npu.set_device(0)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0, "hccl")
        initialize_model_parallel(1, 1)
        yield
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize("projector", [None, "rmsnorm_mlp_adapter"])
@torch.inference_mode()
def test_real_vision_encoder_preserves_media_boundaries(distributed_vision, projector):
    config = HYV4VLVisionConfig(
        hidden_size=288,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        out_hidden_size=64,
        max_image_size=64,
        projector_type=projector,
        temporal_patch_size=2,
    )
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16):
        model = HYV4VisionTransformer(config).to("npu").eval()
        initialize_missing_vision_weights(model, set(), seed=123)
        pixels = torch.randn(64, 768, device="npu", dtype=torch.bfloat16)
        grids = [[1, 4, 4], [3, 4, 4]]
        together = model(pixels, grids)
        separate = [model(pixels[:16], grids[:1])[0], model(pixels[16:], grids[1:])[0]]
        changed = model(-pixels[:16], grids[:1])[0]
    for modality, grid, actual, expected in zip(("image", "video"), grids, together, separate):
        assert actual.shape == (vision_token_count(grid, config, modality), 64)
        assert torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    assert not torch.equal(changed, together[0])


@torch.inference_mode()
def test_multimodal_rope_cache_on_npu_and_graph_replay():
    cache = torch.randn(32, 64, device="npu", dtype=torch.bfloat16)
    positions = torch.tensor([[0, 1, 2, 3], [1, 3, 5, 7], [2, 4, 6, 8]], device="npu")
    expected = gather_mrope_cache(cache.cpu(), positions.cpu(), [12, 10, 10], True)
    torch.testing.assert_close(gather_mrope_cache(cache, positions, [12, 10, 10], True).cpu(), expected)
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        gather_mrope_cache(cache, positions, [12, 10, 10], True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = gather_mrope_cache(cache, positions, [12, 10, 10], True)
    positions.add_(1)
    graph.replay()
    torch.npu.synchronize()
    expected = gather_mrope_cache(cache.cpu(), positions.cpu(), [12, 10, 10], True)
    torch.testing.assert_close(actual.cpu(), expected)
