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
