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
| 带已验证外部 DFlash GGUF 的 llama.cpp | `off`、`dflash` |
| 带已验证外部 DSpark GGUF 的 llama.cpp | `off`、`dspark` |

`draft-model` 仍是尚无生产 llama.cpp 适配器的保留策略。DFlash 与 DSpark 使用不同的外部制品契约；Power 会校验制品类型、目标绑定、tokenizer、来源与摘要，不会把其中一种静默当作另一种。显式请求不受支持的模式会返回错误。

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

FR 只减少 MTP draft head 投影的行数。它可以提高窄词表分布上的峰值，但接受率对语言和领域高度敏感。历史前缀 FR 矩阵的稳态速度很高，在代表性负载上却比 TBQ4 自回归更慢。因此混合制品发布配置保留全部 248,320 个 draft-head 行。当前原始 Q6_K 峰值有意启用 8,192 个 token-ID 前缀，但平衡型工作负载仍使用全词表 MTP。

## 当前实测配置

原始 Q6_K 峰值组合了：

- 原始 22,884,408,288 字节 Q6_K 制品；
- 带 8,192 行 draft-only token-ID 前缀的原生 MTP；
- 7 个 proposal 与 6 个循环状态快照；
- 目标与草稿模型的批量 greedy CUDA 采样；
- Flash Attention 与完整 CUDA 层 offload；
- 精确目标验证与确定性输出摘要。

它的稳态解码中位数为 176.6109 token/s；同一制品的全词表 K7/S7 对照为 147.0207 token/s。但在单轮 12 题校准中，全词表 K7/S6 的请求全程吞吐为 47.032 token/s，前缀 FR 只有 37.290 token/s。该峰值配置尚未完成重复 100 题矩阵。

此前的混合制品 K7/S7 仍是代表性质量记录：稳态解码中位数 175.2089 token/s，请求全程吞吐 83.228 token/s，相对 TBQ4 自回归对照没有观察到回归。该样本同样不能证明通用智力提升。

外部 DSpark 是独立路径，不与 DFlash 或原生 MTP 叠加。固定 K10/S6 的 context-512 峰值达到 169.324 token/s 中位数、167.102 最低值，相对配对目标对照提升 5.250 倍；三次 256-token 输出逐字一致，接受率 90.873%，回放为零。

context-1024 的跨领域诊断给出了更接近真实工作负载的边界：600 个请求全部成功，DSpark 请求全程吞吐为 32.678 token/s，目标对照为 22.618 token/s，提升 1.445 倍，并未观察到分数下降。但跨模式完整输出一致率只有 54/100，而且每个 DSpark 请求都进入过精确回放。因此该配置可用于显式实验，尚不能成为无损生产默认值。

完整解读参阅[性能证据](/performance)，适配器 API、基准命令与验收规则参阅[规范推测解码设计](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)。
