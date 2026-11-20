from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.model_executor.offloader.base import create_offloader
from vllm_ascend.model_executor.offloader.prefetch import _is_using_nz_weight, _prefetch_device_loading_context


def test_prefetch_loading_preserves_unmarked_modules():
    module = torch.nn.Linear(2, 4)
    called = Mock()

    @contextmanager
    def fallback(actual_module, device):
        called(actual_module, device)
        yield actual_module

    device = torch.device("npu:0")
    with _prefetch_device_loading_context(module, device, fallback=fallback) as actual:
        assert actual is module
        assert actual.weight.device.type == "cpu"
    called.assert_called_once_with(module, device)


def test_create_offloader_without_config_returns_noop():
    offloader = create_offloader(None)

    assert isinstance(offloader, NoopOffloader)


def test_create_offloader_for_non_prefetch_backend_returns_noop():
    offload_config = SimpleNamespace(
        offload_backend=None,
        prefetch=None,
    )

    offloader = create_offloader(offload_config)

    assert isinstance(offloader, NoopOffloader)


def test_is_using_nz_weight_handles_invalid_npu_format(monkeypatch):
    param = SimpleNamespace(
        data=SimpleNamespace(device=SimpleNamespace(type="npu")),
    )

    monkeypatch.setattr(
        "vllm_ascend.model_executor.offloader.prefetch.torch_npu.get_npu_format",
        lambda _: object(),
        raising=False,
    )

    assert not _is_using_nz_weight(param)


def test_pageable_prefetch_graph_rejected_before_capture(monkeypatch):
    import pytest
    from vllm_ascend.model_executor.offloader.prefetch import (
        _raise_if_pageable_prefetch_incompatible_with_graph,
    )

    monkeypatch.setattr("vllm_ascend.model_executor.offloader.prefetch.should_pin_memory", lambda: False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False)))
    with pytest.raises(RuntimeError, match="Pageable H2D copies cannot be captured"):
        _raise_if_pageable_prefetch_incompatible_with_graph()


def test_pageable_prefetch_eager_allowed(monkeypatch):
    from vllm_ascend.model_executor.offloader.prefetch import (
        _raise_if_pageable_prefetch_incompatible_with_graph,
    )

    monkeypatch.setattr("vllm_ascend.model_executor.offloader.prefetch.should_pin_memory", lambda: False)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True)))
    _raise_if_pageable_prefetch_incompatible_with_graph()


def test_pinned_prefetch_graph_allowed(monkeypatch):
    from vllm_ascend.model_executor.offloader.prefetch import (
        _raise_if_pageable_prefetch_incompatible_with_graph,
    )

    monkeypatch.setattr("vllm_ascend.model_executor.offloader.prefetch.should_pin_memory", lambda: True)
    monkeypatch.setattr("vllm.config.get_current_vllm_config", lambda: SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False)))
    _raise_if_pageable_prefetch_incompatible_with_graph()
