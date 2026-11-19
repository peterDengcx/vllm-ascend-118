from contextlib import contextmanager


@contextmanager
def dfly_runner_bootstrap(vllm_config):
    """Allow only the disposable GPU-runner drafter to initialize HY4 DFly.

    NPUModelRunner replaces this GPU drafter after the parent constructor.
    Keep the real upstream M-RoPE guard intact for all other draft models.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    draft = getattr(spec, "draft_model_config", None)
    if not getattr(getattr(draft, "hf_config", None), "ascend_dfly", False):
        yield
        return

    from vllm.v1.worker import gpu_model_runner

    original = gpu_model_runner.DFlashProposer

    class DFlyBootstrapProposer(original):
        def _raise_if_mrope(self):
            if not getattr(self.draft_model_config.hf_config, "ascend_dfly", False):
                return super()._raise_if_mrope()

    gpu_model_runner.DFlashProposer = DFlyBootstrapProposer
    try:
        yield
    finally:
        gpu_model_runner.DFlashProposer = original
