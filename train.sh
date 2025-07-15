#!/bin/bash

    # os.environ["LIBTPU_INIT_ARGS"] = (
    #     os.getenv("LIBTPU_INIT_ARGS", "") + " "
    #     "--xla_tpu_enable_latency_hiding_scheduler=true "
    #     "--xla_enable_async_collective_permute=true "
    #     "--xla_tpu_enable_ag_backward_pipelining=true "
    #     "--xla_tpu_enable_data_parallel_all_reduce_opt=true "
    #     "--xla_tpu_data_parallel_opt_different_sized_ops=true "
    #     "--xla_tpu_enable_async_collective_fusion=true "
    #     "--xla_tpu_enable_async_collective_fusion_multiple_steps=true "
    #     "--xla_tpu_overlap_compute_collective_tc=true "
    #     "--xla_enable_async_all_gather=true "
    #     "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
    #     "--xla_tpu_megacore_fusion_allow_ags=true "
    #     "TPU_MEGACORE=MEGACORE_DENSE "
    # )
LIBTPU_INIT_ARGS="--xla_tpu_enable_latency_hiding_scheduler=true --xla_enable_async_collective_permute=true --xla_tpu_enable_ag_backward_pipelining=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_megacore_fusion_allow_ags=true TPU_MEGACORE=MEGACORE_DENSE" \
uv run -m qu1.trainer $@