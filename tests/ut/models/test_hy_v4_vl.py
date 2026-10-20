# SPDX-License-Identifier: Apache-2.0
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast
from transformers.video_utils import VideoMetadata
from vllm.config import VllmConfig, get_current_vllm_config, set_current_vllm_config

from vllm_ascend.ops.layernorm import AscendRMSNorm
from vllm_ascend.ops.rotary_embedding import gather_mrope_cache, get_cos_and_sin_mla
from vllm_ascend.patch.hy4_vl.config import HYV4VLConfig
from vllm_ascend.patch.hy4_vl.model import HYV4VLForConditionalGeneration
from vllm_ascend.patch.hy4_vl.processing import HYV4VLProcessingInfo
from vllm_ascend.patch.hy4_vl.processor import (
    HYV4VLImageProcessor,
    HYV4VLProcessor,
    HYV4VLVideoProcessor,
    vision_token_count,
)
from vllm_ascend.patch.hy4_vl.random_weights import initialize_missing_vision_weights


def test_unquantized_vision_does_not_inherit_text_norm_bias_quantization(monkeypatch):
    """ModelSlim text norm biases must not add an unexported bias to the BF16 vision tower."""
    import vllm_ascend.patch.hy4_vl.model as model_module

    quant_config = SimpleNamespace(quant_description={"model.layers.0.input_layernorm.bias": "FLOAT"})
    config = VllmConfig()
    config.quant_config = quant_config
    config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(vision_config=SimpleNamespace(), text_config=SimpleNamespace()),
        get_multimodal_config=lambda: SimpleNamespace(get_limit_per_prompt=lambda modality: 1),
    )

    def make_vision(*args, **kwargs):
        visual = torch.nn.Module()
        visual.norm = AscendRMSNorm(8)
        return visual

    def make_language_model(**kwargs):
        assert get_current_vllm_config().quant_config is quant_config
        language_model = torch.nn.Module()
        language_model.make_empty_intermediate_tensors = lambda *args: None
        return language_model

    monkeypatch.setattr(HYV4VLForConditionalGeneration, "_mark_tower_model", lambda *args: nullcontext())
    monkeypatch.setattr(HYV4VLForConditionalGeneration, "_mark_language_model", lambda *args: nullcontext())
    monkeypatch.setattr(model_module, "HYV4VisionTransformer", make_vision)
    monkeypatch.setattr(model_module, "init_vllm_registered_model", make_language_model)
    with set_current_vllm_config(config):
        model = HYV4VLForConditionalGeneration(vllm_config=config)
        assert model.visual.norm.bias is None
        assert get_current_vllm_config().quant_config is quant_config
    assert config.quant_config is quant_config


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


@pytest.mark.parametrize("separate_video_config", [False, True])
def test_runtime_processor_preserves_checkpoint_normalization_and_overrides(separate_video_config):
    original = make_processor()
    context = Mock()
    context.get_hf_config.return_value = original.config
    context.get_tokenizer.return_value = original.tokenizer
    context.get_hf_image_processor_config.return_value = {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "min_pixels": 4096,
        "max_pixels": 8192,
    }
    context.get_merged_mm_kwargs.return_value = {
        "images_kwargs": {"max_pixels": 4096},
        "videos_kwargs": {"max_frames": 4},
    }
    video_config = {"image_mean": [0.25] * 3, "image_std": [0.75] * 3, "max_frames": 10}
    with patch(
        "vllm_ascend.patch.hy4_vl.processing.get_hf_file_to_dict",
        side_effect=lambda name, *args, **kwargs: (
            video_config if name == "video_preprocessor_config.json" and separate_video_config else None
        ),
    ):
        processor = HYV4VLProcessingInfo(context).get_hf_processor()
    pixels = processor.image_processor(Image.new("RGB", (64, 64), "red"), return_tensors="pt")["pixel_values"]
    channels = pixels.reshape(-1, 3, 16, 16).mean((0, 2, 3))
    torch.testing.assert_close(channels, torch.tensor([1.0, -1.0, -1.0]))
    assert processor.image_processor.size["longest_edge"] == 4096
    assert processor.video_processor.max_frames == 4
    assert list(processor.video_processor.image_mean) == ([0.25] * 3 if separate_video_config else [0.5] * 3)
    assert list(processor.video_processor.image_std) == ([0.75] * 3 if separate_video_config else [0.5] * 3)


def test_runtime_processor_reads_local_config_when_model_config_cache_is_empty(tmp_path):
    original = make_processor()
    context = Mock()
    context.get_hf_config.return_value = original.config
    context.get_tokenizer.return_value = original.tokenizer
    context.get_hf_image_processor_config.return_value = {}
    context.get_merged_mm_kwargs.return_value = {}
    context.model_config.model = str(tmp_path)
    context.model_config.revision = None
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.5] * 3, "image_std": [0.5] * 3, "min_pixels": 4096, "max_pixels": 4096})
    )
    processor = HYV4VLProcessingInfo(context).get_hf_processor()
    pixels = processor(images=Image.new("RGB", (64, 64), "red"), return_tensors="pt")["pixel_values"]
    assert pixels.shape == (16, 768)
    torch.testing.assert_close(pixels.reshape(-1, 3, 16, 16).mean((0, 2, 3)), torch.tensor([1.0, -1.0, -1.0]))


@pytest.mark.parametrize("total", [2, 4, 8])
def test_video_sampling_preserves_all_available_frames(total):
    processor = HYV4VLVideoProcessor()
    metadata = VideoMetadata(total_num_frames=total, fps=2.0)
    indices = processor.sample_frames(metadata, num_frames=total)
    assert indices.tolist() == list(range(total))


@pytest.mark.parametrize("minimum,maximum,side", [(4096, 524288, 64), (65536, 524288, 256), (65536, 4096, 32)])
def test_video_minimum_pixels_and_total_budget(minimum, maximum, side):
    processor = HYV4VLVideoProcessor()
    frames = torch.zeros(4, 3, 64, 64, dtype=torch.uint8)
    frames[0, 0] = 255
    frames[1, 2] = 255
    frames[2, 1] = 255
    frames[3] = 255
    result = processor(
        [frames],
        min_pixels=minimum,
        max_pixels=maximum,
        max_frames=4,
        video_max_image_size=256,
        timestamp_token_reserve=0,
        return_tensors="pt",
    )
    assert result["video_grid_thw"].tolist() == [[4, side // 16, side // 16]]
    assert result["pixel_values_videos"].shape == (4 * (side // 16) ** 2, 768)
    assert 4 * side**2 <= maximum
    means = result["pixel_values_videos"].reshape(4, -1, 3, 16, 16).mean((1, 3, 4))
    expected = torch.tensor([[1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, 1.0, 1.0]])
    torch.testing.assert_close(means, expected)


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
