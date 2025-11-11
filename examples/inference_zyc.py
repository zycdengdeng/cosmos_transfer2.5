#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# ZYC 自定义推理脚本
# 用于从点云投影的 color 和 depth 视频生成真实 RGB 视频

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter files."""
    setup: SetupArguments
    """Setup arguments."""
    overrides: InferenceOverrides
    """Inference parameter overrides."""


def main(
    args: Args,
):
    inference_samples, batch_hint_keys = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    if args.setup.benchmark:
        assert len(inference_samples) > 1, "Benchmarking must be run for more than 1 sample."
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.inference import Control2WorldInference

    inference = Control2WorldInference(args.setup, batch_hint_keys=batch_hint_keys)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)
    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
