# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from vllm_ascend.ops.rotary_embedding import gather_mrope_cache, get_cos_and_sin_mla
from vllm_ascend.patch.hy4_vl.config import HYV4VLConfig
from vllm_ascend.patch.hy4_vl.model import HYV4VLForConditionalGeneration
from vllm_ascend.patch.hy4_vl.processor import (
    HYV4VLImageProcessor,
    HYV4VLProcessor,
    HYV4VLVideoProcessor,
    vision_token_count,
)
from vllm_ascend.patch.hy4_vl.random_weights import initialize_missing_vision_weights


@pytest.mark.parametrize("interleaved", [True, False])
def test_mrope_axis_selection_and_text_equivalence(interleaved):
    positions = torch.tensor([[0, 2, 9], [1, 3, 7], [2, 4, 8]])
    cache = torch.arange(16 * 64).reshape(16, 64).float()
    sections = [12, 10, 10]
    actual = gather_mrope_cache(cache, positions, sections, interleaved).squeeze(1).squeeze(1)
    expected = torch.empty(3, 32)
    for frequency in range(32):
        if interleaved:
            axis = 1 if frequency % 3 == 1 and frequency <= 30 else 2 if frequency % 3 == 2 and frequency <= 30 else 0
        else:
            axis = 0 if frequency < 12 else 1 if frequency < 22 else 2
        expected[:, frequency] = cache[positions[axis], frequency]
    torch.testing.assert_close(actual, expected.repeat(1, 2))
    same = positions[0].expand(3, -1)
    actual = gather_mrope_cache(cache, same, sections, interleaved).squeeze(1).squeeze(1)
    torch.testing.assert_close(actual[:, :32], cache[positions[0], :32])


def test_missing_random_vision_is_explicit_reproducible_and_preserves_loaded():
    module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8))
    module[0].bias.data.fill_(7)
    loaded = {"visual.0.bias"}
    with pytest.raises(ValueError, match="ascend_random_vision_seed"):
        initialize_missing_vision_weights(module, loaded, seed=None)
    names = initialize_missing_vision_weights(module, loaded, seed=13)
    first = {key: value.clone() for key, value in module.state_dict().items()}
    assert names == {"visual.0.weight", "visual.1.weight", "visual.1.bias"}
    assert torch.all(module[0].bias == 7)
    torch.testing.assert_close(module[1].weight, torch.ones(8))
    initialize_missing_vision_weights(module, loaded, seed=13)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, first[key], rtol=0, atol=0)
    initialize_missing_vision_weights(module, loaded, seed=14)
    assert not torch.equal(module[0].weight, first["0.weight"])
    assert (
        initialize_missing_vision_weights(module, {f"visual.{name}" for name, _ in module.named_parameters()}, None)
        == set()
    )


def test_mrope_metadata_needs_no_active_vllm_config():
    import vllm_ascend.ops.rotary_embedding as rotary

    cache = torch.randn(32, 64)
    positions = torch.tensor([[0, 1, 2], [1, 3, 5], [2, 4, 6]])
    rope = {"mrope_section": [12, 10, 10], "mrope_interleaved": True}
    buffer = torch.zeros(8, 1, 1, 64)
    with (
        patch.object(rotary, "_cos_cache", cache),
        patch.object(rotary, "_sin_cache", -cache),
        patch.object(rotary, "_cos_mla", buffer),
        patch.object(rotary, "_sin_mla", buffer.clone()),
    ):
        actual_cos, actual_sin = get_cos_and_sin_mla(positions, use_cache=True, rope_parameters=rope)
        expected = gather_mrope_cache(cache, positions, [12, 10, 10], True)
        torch.testing.assert_close(actual_cos, expected)
        torch.testing.assert_close(actual_sin, -expected)
        assert actual_cos.data_ptr() == buffer.data_ptr()
        with pytest.raises(ValueError, match="explicit rope_parameters"):
            get_cos_and_sin_mla(positions)


def make_processor():
    tokens = [
        "[UNK]",
        "text",
        "<image>",
        "<video>",
        "<im_start>",
        "<im_end>",
        "<im_pad>",
        "<newline>",
        "<vid_start>",
        "<vid_end>",
        "<vid_pad>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]")),
        unk_token="[UNK]",
        additional_special_tokens=tokens[2:],
    )
    config = HYV4VLConfig(
        image_start_token_id=4,
        image_end_token_id=5,
        image_token_id=6,
        image_newline_token_id=7,
        video_start_token_id=8,
        video_end_token_id=9,
        video_token_id=10,
        vision_config={"max_image_size": 64},
    )
    return HYV4VLProcessor(
        HYV4VLImageProcessor(min_pixels=4096, max_pixels=8192),
        tokenizer,
        HYV4VLVideoProcessor(),
        config=config,
    )


def test_image_processor_patch_layout_and_grid():
    processor = make_processor()
    result = processor.image_processor(
        images=[Image.new("RGB", (64, 64), "red"), Image.new("RGB", (128, 64), "blue")], return_tensors="pt"
    )
    assert result["pixel_values"].shape == (48, 768)
    assert result["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert torch.isfinite(result["pixel_values"]).all()
    assert not torch.equal(result["pixel_values"][:16], result["pixel_values"][16:32])
    assert vision_token_count([1, 4, 4], processor.config.vision_config, "image") == 6


@pytest.mark.parametrize("modality", ["image", "video"])
def test_visual_embedding_schema_boundaries_and_no_bypass(modality):
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = HYV4VLConfig(vision_config={"out_hidden_size": 16}, text_config={"hidden_size": 16})
    model.vision_diagnostics = False
    model.visual = Mock(return_value=[torch.randn(6, 16)])
    key = "pixel_values" if modality == "image" else "pixel_values_videos"
    inputs = {key: torch.randn(16, 768), f"{modality}_grid_thw": torch.tensor([[1, 4, 4]])}
    result = model.embed_multimodal(**inputs)
    assert result[0].shape == (6, 16)
    model.visual.assert_called_once()
    with pytest.raises(ValueError, match="precomputed"):
        model.embed_multimodal(image_embeds=torch.zeros(6, 16))


def test_image_mrope_uses_spatial_axes_and_preserves_text_continuation():
    processor = make_processor()
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = processor.config
    # A 64x128 image has 2x4 merged cells, plus one newline per row.
    tokens = [1, 4, *([6] * 10), 5, 1]
    grids = {"image_grid_thw": [torch.tensor([1, 4, 8])]}
    with patch("vllm_ascend.patch.hy4_vl.model.MultiModalFeatureSpec.gather_kwargs", return_value=grids):
        positions, delta = model.get_mrope_input_positions(tokens, [Mock()])
    torch.testing.assert_close(positions[0, 2:12], torch.full((10,), 2))
    torch.testing.assert_close(positions[1, 2:12], torch.tensor([2] * 5 + [3] * 5))
    torch.testing.assert_close(positions[2, 2:12], torch.tensor([2, 3, 4, 5, 6] * 2))
    torch.testing.assert_close(positions[:, 12:], torch.tensor([[7, 8]]).expand(3, -1))
    assert delta == -5
    with (
        patch("vllm_ascend.patch.hy4_vl.model.MultiModalFeatureSpec.gather_kwargs", return_value=grids),
        pytest.raises(ValueError, match="boundaries and grid"),
    ):
        model.get_mrope_input_positions([1, 4, 6, 5], [Mock()])


def test_video_mrope_keeps_timestamp_bridges_and_multiple_frames():
    processor = make_processor()
    model = HYV4VLForConditionalGeneration.__new__(HYV4VLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = processor.config
    tokens = [1, 8, 1, 4, *([10] * 6), 5, 1, 4, *([10] * 6), 5, 9, 1]
    grids = {"video_grid_thw": [torch.tensor([2, 4, 4])]}
    with patch("vllm_ascend.patch.hy4_vl.model.MultiModalFeatureSpec.gather_kwargs", return_value=grids):
        positions, delta = model.get_mrope_input_positions(tokens, [Mock()])
    torch.testing.assert_close(positions[0, 4:10], torch.full((6,), 4))
    torch.testing.assert_close(positions[:, 10:13], torch.tensor([[7, 8, 9]]).expand(3, -1))
    torch.testing.assert_close(positions[0, 13:19], torch.full((6,), 10))
    torch.testing.assert_close(positions[:, 19:], torch.tensor([[13, 14, 15]]).expand(3, -1))
    assert delta == -6


def test_multimodal_mtp_unwraps_text_and_keeps_mrope_and_quantization():
    # Platform patch registration must precede the upstream override call.
    from vllm.config.speculative import SpeculativeConfig

    from vllm_ascend.patch.platform import patch_speculative_config  # noqa: F401

    quantization = {"quant_method": "modelopt", "quantization": {"quant_algo": "MXFP8"}}
    config = HYV4VLConfig(quantization_config=quantization, num_nextn_predict_layers=1)
    draft = SpeculativeConfig.hf_config_override(config)
    assert draft.model_type == "hy_v4_mtp"
    assert draft.architectures == ["HYV4MTPModel"]
    assert draft.rope_parameters["mrope_section"] == [12, 10, 10]
    assert draft.quantization_config == quantization
