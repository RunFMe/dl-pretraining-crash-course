# DL pretraining crash course

Short, runnable lessons on how large-scale training actually works. Each lesson is a notebook you can open in Colab.

## 1. How training starts on N GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RunFMe/dl-pretraining-crash-course/blob/main/01_how_training_starts/how_training_starts.ipynb)

A distributed training job is N separate, ordinary processes running the same script. They share nothing; each one knows its
place only because the launcher put a few environment variables into its environment. This notebook starts the processes, shows
exactly which variables they receive and what each one means, replaces `torchrun` with a `for` loop, watches the processes meet
at `MASTER_ADDR:MASTER_PORT` and run their first `all_reduce`, and ends with questions you check by breaking the environment on purpose.

Runs on a CPU-only machine (4 processes, `gloo`) or on a multi-GPU box (one process per GPU, NCCL). Needs only `torch`.

## 2. How open LLMs get long context

[Read the lesson](02_long_context/README.md)

How open-weight LLMs from 2022 to 2026 reach 128K–1M tokens of context. It starts with how RoPE assigns a frequency to every pair of
dimensions (illustrated on Qwen3, Qwen3.5 and Qwen VL's multidimensional M-RoPE), then walks through YaRN line by line in DeepSeek
V4-Flash's own modeling code. It follows with a year-by-year table of context lengths, RoPE bases and extension methods, and a
count of what the current flagship from each of 18 labs actually uses. This lesson is a reading with figures, not a notebook.

## 3. FlashAttention, from the idea to FlashAttention-4

[Read the lesson](03_flash_attention/README.md)

Why exact attention is limited by memory traffic, and how FlashAttention removes it with tiling, an online softmax and
recomputation (with a worked two-block example and a 25-line reference implementation). The lesson then follows each version to
the GPU unit that became the bottleneck: FA2's repartitioning on A100, FA3's overlap of exponentials and asynchronous matmuls
plus FP8 on H100, and FA4's polynomial exp2 and lazy rescaling on B200. It ends with the numerical-stability history, including the
2025–2026 BF16 backward precision problem, what fixes it, and which fixes are in the main repo. This lesson is a reading with figures, not a notebook.

## 4. Rebuilding MiMo-V2.6: an open frontier RL run, stage by stage

[Read the lesson](04_mimo_v26_rl/README.md) · [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RunFMe/dl-pretraining-crash-course/blob/main/04_mimo_v26_rl/rl_by_hand.ipynb)

What it would take to rebuild Xiaomi's MiMo-V2.6-Pro (1.02T parameters, September 2026), the most openly documented frontier
agentic-RL run so far. The lesson goes stage by stage (base model, SFT, a single mixed RL run, on-policy distillation) and says
for each one what is released, what is only described and what is missing. Most of it is about RL: how MiMo rebuilt GRPO for
100K-token, asynchronously generated agent trajectories (prompt-mean aggregation, importance ratios against the inference engine,
clipping as a mask, graders that rank passing solutions), the systems that keep 25K trajectories per step flowing, how the
environments and graders were built and where they leaked, and how on-policy distillation reuses the same loss with a teacher as
the advantage. It is a reading in five parts with figures, plus a NumPy notebook that recomputes every worked example on CPU.
