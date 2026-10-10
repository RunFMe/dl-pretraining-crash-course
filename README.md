# DL pretraining crash course

Short, runnable lessons on how large-scale training actually works. Each lesson is a notebook you can open in Colab.

## 1. How training starts on N GPUs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RunFMe/dl-pretraining-crash-course/blob/main/01_how_training_starts/how_training_starts.ipynb)

A distributed training job is N separate, ordinary processes running the same script. They share nothing; each one knows its
place only because the launcher put a few environment variables into its environment. This notebook starts the processes, shows
exactly which variables they receive and what each one means, replaces `torchrun` with a `for` loop, watches the processes meet
at `MASTER_ADDR:MASTER_PORT` and run their first `all_reduce`, and ends with questions you check by breaking the environment on purpose.

Runs on a CPU-only machine (4 processes, `gloo`) or on a multi-GPU box (one process per GPU, NCCL). Needs only `torch`.
