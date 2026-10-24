# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Pro monokernels.

SGLang serves DeepSeek-V4-Pro and DeepSeek-V4.1 under one architecture,
DeepseekV4ForCausalLM; the config's model_type (deepseek_v4, deepseek_v41) tells
them apart. The spec, forward, and kernels land in the model PR. Nothing here is
registered until that PR lands.
"""
