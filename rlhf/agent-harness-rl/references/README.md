# Agent Harness → RL 参考资料索引

本文档是 `agent-harness-rl/readme.md`（Agent Harness 如何接入 RL 的架构版图/静态审计）写作时归档的全部论文与博客资料索引。所有资料按类别归档，保证文中事实可回溯到固定来源。

## 目录结构

```
references/
├── README.md                       ← 本索引
├── papers/                         ← 论文（MinerU 转换：md + images/）
│   ├── agentjet/                   ← AgentJet：分布式 swarm 训练框架
│   ├── agent-lightning/            ← Agent Lightning v1.0（harnessed agentic RL）
│   └── polar/                      ← Polar：任意 harness 上的 agentic RL
└── articles/                       ← 官方博客
    ├── vllm-agent-lightning/       ← vLLM Blog：token-ID 返回与 retokenization drift
    └── verifiers-v1/               ← primeintellect.ai：verifiers v1 分层设计
```

## 一、论文（papers/）

| 资料 | 论文 | 内容 | 对应 readme.md 章节 |
| --- | --- | --- | --- |
| `agentjet/` | arXiv `2606.04484v2`，ModelScope | 《AgentJet: A Distributed Swarm Training Framework for Agentic Reinforcement Learning》：episode API + swarm model server、多 agent 异构模型路由 | §4.8（AgentJet Swarm）、§Q2 |
| `agent-lightning/` | arXiv `2608.17528v1`，Microsoft | 《Agent Lightning v1.0: Towards Harnessed Agentic RL》：proxy/token-ID 捕获、per-rollout loss、异步训练 | §4.1、§7、§9 |
| `polar/` | arXiv `2605.24220v1`，NVIDIA NeMo | 《Polar: Agentic RL on Any Harness at Scale》：独立 rollout service、多协议代理、模拟 streaming | §4.7（Polar + slime）、§12.1 |

每个目录内含：`<slug>.md`（原文转换）、`images/`（转换图片，md 引用完整）。原始 PDF 原件与 MinerU 中间产物已删除（可在线获取，来源见上表），md 为本地唯一副本。

## 二、博客文章（articles/）

| 资料 | 来源 | 内容 | 对应 readme.md 章节 |
| --- | --- | --- | --- |
| `vllm-agent-lightning/` | `https://vllm.ai/blog/agent-lightning` | vLLM《No More Retokenization Drift: Returning Token IDs via the OpenAI Compatible API Matters in Agent RL》：生成时返回 token IDs 的由来与 retokenization drift 问题 | §7.2（重新分词漂移）、§13 时间线（2025-10-22） |
| `verifiers-v1/` | `https://www.primeintellect.ai/blog/verifiers-v1` | primeintellect.ai《verifiers v1: Decomposing Tasksets and Harnesses for Agentic RL & Evaluations》：taskset/harness/runtime 分层与 interception server | §4.9（PRIME-RL + verifiers v1） |

每个目录内含：`<slug>.md`（全文，图片引用已修复为本地 `images/`）、`<slug>.html`（原始快照）。原始抓取目录中的 Paperpile `_files/` 图片缺失，已从在线站点补下载至本地 `images/` 并修复 md 链接；html 快照内部的相对引用随 `_files/` 删除而失效，正文以修复后的 md 为准。

## 归档与清理记录

- 归档日期：2026-09-09（与 `readme.md` 检索口径一致），来源 URL 与 arXiv 编号取自 `readme.md` 引用列表
- 已删除：MinerU/抓取中间产物（`*_content_list.json`、`*_content_list_v2.json`、`*_model.json`、`layout.json`、`content_list.json`）与论文原始 PDF
- 已修复：两篇博客共 11 张图片（vLLM 4 张 png/svg、verifiers 7 张 png）下载为有效图片文件，md 图片引用缺失 0
- 三篇论文 md 图片引用缺失 0
