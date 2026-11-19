# SPDX-License-Identifier: Apache-2.0
"""Configuration for a complete MTP drafter hosted on the final PP stage."""

from copy import copy


def local_mtp_parallel_config(target_parallel_config, draft_tensor_parallel_size=None):
    """Describe draft kernels without repartitioning the target PP process group.

    The runner owns stage placement and loads the drafter only on its final
    stage. The drafter itself is complete, uses the existing TP/DP groups, and
    must not inherit the target's PP layer partition or TP*PP world size.
    No process group is created or replaced here.
    """
    parallel = copy(target_parallel_config)
    parallel.pipeline_parallel_size = 1
    if draft_tensor_parallel_size is not None:
        parallel.tensor_parallel_size = draft_tensor_parallel_size
    parallel.world_size = parallel.tensor_parallel_size * getattr(parallel, "prefill_context_parallel_size", 1)
    if getattr(parallel, "distributed_executor_backend", None) == "external_launcher":
        parallel.world_size *= parallel.data_parallel_size
    return parallel
