---
title: 推测解码
description: A3S Power 如何分离建议性 proposal 生成与目标模型的精确验证、回滚和质量门槛。
---

# 模型中立的推测解码

推测解码是一项运行时能力，而不是 Qwen 专用分支。Proposal 生成只提供建议；每一个最终输出 token 都由目标模型裁决。

Power 提供共享的事务、调度、回滚与证据协议。后端适配器提供特定架构的 draft graph、状态快照、logits 访问与兼容性元数据。

## 精确性事务

每一轮模型推测严格遵循同一顺序：

1. 为目标模型、草稿模型、采样器和解码器状态保留检查点。
2. 请求适配器生成不超过调度上限的 proposal token。
3. 在一次目标模型前向中计算 anchor 与 proposal block。
4. 只采样到第一个不匹配位置为止的目标行。
5. 提交已接受前缀，并丢弃所有被拒绝的状态行。
6. 若生成继续，只输出一个 correction 或 bonus token。
7. 取消或失败时，恢复最后一次已提交事务。

在 greedy 解码下，推测模式与自回归模式的 token ID 必须完全一致。随机采样下，目标采样器状态对每个实际输出的目标样本只推进一次，绝不为不可见的被拒后缀推进。

## 能力必须失败关闭

| 后端／模型能力 | 可用策略 |
| --- | --- |
| mistral.rs 或 proxy | `off`；`auto` 解析为 `off` |
| picolm | `off`、`prompt-lookup`、`ngram-context` |
| 不含原生 prediction tensor 的 llama.cpp | `off` |
| 带 `*.nextn_predict_layers > 0` 的 llama.cpp | `off`、`mtp` |

共享策略词表还包含 `draft-model`、`dflash` 与 `dspark`，但在具备兼容适配器制品和计算图之前保持不可用。显式请求不受支持的模式会返回错误；Power 不会静默替换成廉价算法，也不会把 n-gram lookup 伪装成 MTP。

## Draft 宽度不等于回滚宽度

```text
spec_mode = "mtp"
spec_draft_max = 7
spec_mtp_recurrent_snapshots = 7

# 实验性紧凑 draft 投影；全词表模式请省略。
# spec_mtp_fr_vocab_size = 8192
```

`spec_draft_max` 限制 proposal 宽度，`spec_mtp_recurrent_snapshots` 限制目标模型常驻回滚状态。K7/S7 为每一个 proposal token 保留回滚点。K7/S6 在高接受率提示词上可能更快，但被拒后缀可能超出驻留窗口。

Power 的防护型 K7/S6 路径只允许一次精确重放，之后把该请求的 proposal 上限收紧到 6。它在不改变高接受率路径的前提下限制重放。K7/S7 完全避开该条件，因此是平衡型默认配置。

## TBQ4 与 FR 解决不同问题

TBQ4 是制品构建策略：它降低部分张量的带宽需求，并不是通用运行时开关。当前混合制品让 MTP block 保持 Q6_K，主 FFN 张量使用 Q4_0，独立 draft head 使用 Q4_K。

FR 只减少 MTP draft head 投影的行数。它可以提高窄词表分布上的峰值，但接受率对语言和领域高度敏感。历史前缀 FR 矩阵的稳态速度很高，在代表性负载上却比 TBQ4 自回归更慢。因此当前配置保留全部 248,320 个 draft-head 行。

## 当前验收配置

平衡型 Qwen3.8-27B K7/S7 记录组合了：

- 从 Q6_K 衍生的 TBQ4 混合制品；
- 原生全词表 MTP；
- 7 个 proposal 与 7 个循环状态快照；
- 目标与草稿模型的批量 greedy CUDA 采样；
- Flash Attention 与完整 CUDA 层 offload；
- 精确目标验证与确定性输出摘要。

在固定 100 题负载上，它达到 175.2089 token/s 稳态解码中位数与 83.228 token/s 请求全程均值。质量矩阵没有观察到相对 TBQ4 自回归对照的回归，但该样本不能证明通用智力提升。

完整解读参阅[性能证据](/performance)，适配器 API、基准命令与验收规则参阅[规范推测解码设计](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)。
