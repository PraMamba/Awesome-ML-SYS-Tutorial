# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **documentation and tutorial repository** — not a software project. It contains ML Systems learning notes authored primarily by one researcher, covering RLHF infrastructure, inference systems (SGLang), distributed training (PyTorch), and transformer architectures. Content is bilingual (English and Chinese).

There is no build system, no test suite, no package manager, and no CI/CD pipeline. The repository consists almost entirely of Markdown articles and supporting images.

## Structure

- **`rlhf/`** — The largest section. Deep technical documentation on RL training frameworks:
  - `slime/` — slime RL framework: code walkthroughs, FP8/INT4 training, FSDP backend, speculative decoding, multi-turn VLM RL
  - `verl/` — verl framework: source code analysis (init, rollout, make experience), multi-turn interaction, server-based rollout, parameter reference
  - `areal/` — AReal framework code walkthrough
  - `OpenRLHF/` — OpenRLHF framework analysis and SGLang integration
  - `sys-design/` — System design deep dives: weight update mechanisms, FSDP training, Megatron, MoE expert parallelism
  - `GRPO/`, `partial-rollout/` — Algorithm-specific notes

- **`sglang/`** — SGLang inference framework documentation:
  - `code-walk-through/` — End-to-end request lifecycle, diffusion model support, multimodal requests
  - `scheduler/`, `zero-overhead-scheduler/` — Scheduler architecture and optimization
  - `kvcache-code-walk-through/`, `dp-attention/` — Memory and attention systems
  - `quantization/`, `speculative-decoding/`, `constraint-decoding/` — Optimization techniques
  - `online-update-weights/`, `sglang-verl-engine/` — RL integration points

- **`torch/`** — PyTorch fundamentals:
  - `torch-distributed/codes/` — **The only Python source files in the repo** (8 demo scripts for distributed communication primitives: all_reduce, broadcast, scatter, send/recv)
  - `nccl/` — NCCL and GPU topology
  - `cuda-graph/` — CUDA graph optimization
  - `tensor-parallelism/`, `mem-snapshot/` — Parallelism and memory analysis

- **`transformers/`** — Attention mechanisms, special tokens, chat templates

- **`engineer/`** — Developer guides for Docker and environment setup (uv)

## Content Conventions

- Articles often exist in both Chinese (`readme.md` or `*_CN.md`) and English (`readme-en.md`, `*_EN.md`, or `*_en.md`) versions. The naming is not fully consistent.
- Articles marked `[Pending Review]` in the README are drafts or incomplete.
- Supporting images live in `pics/`, `pic/`, `img/`, `imgs/`, `figs/`, `assets/`, `static/`, or `image/` subdirectories alongside their articles.
- The README.md serves as the primary table of contents with links to all articles and external Zhihu posts.

## Working with This Repo

- **Adding content**: Create a new subdirectory under the appropriate top-level section, add a `readme.md` (and optionally an English variant), and update `README.md` and `README-cn.md` with links.
- **Images**: Place in a subdirectory next to the markdown file. Use relative paths in markdown.
- **No linting or formatting tools** are configured. Markdown is freeform.
- **The only runnable code** is in `torch/torch-distributed/codes/` — these are PyTorch distributed examples meant to be run with `torchrun` or `python -m torch.distributed.launch`.
