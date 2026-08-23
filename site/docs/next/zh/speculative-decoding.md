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
| 带已验证 DFlash2 GGUF 的 Power 清单 | 只完成准入；等待绑定更新期间执行保持失败关闭 |
| 带已验证外部 DSpark GGUF 的 llama.cpp | `off`、`dspark` |

`draft-model` 仍是尚无生产 llama.cpp 适配器的保留策略。DFlash v1、DFlash2
与 DSpark 使用不同的外部制品契约；Power 会校验制品类型、目标绑定、
tokenizer、来源与摘要，不会把其中一种静默当作另一种。显式请求不受支持的
模式会返回错误。

DFlash2 有独立的类型策略与 selector/convolution 张量契约。当前固定的
`llama-cpp-rs` revision 尚未暴露上游 DFlash2 执行器，所以显式或 `auto`
选择都会返回需要审查绑定更新的错误，不会被重命名为 DFlash v1。

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

按请求工作的自适应控制器不会把重放当成第一条反馈。它从 `min(K, S)` 开始；首轮 proposal 全部接受才开启更宽的 K 形状，首轮只要部分接受，就在该请求中关闭宽路径。健康的部分接受轮保持图形状；连续低收益则单向切回目标模型。原生 MTP、DFlash 与 DSpark 后端共享这套调度器；为兼容现有配置，ACL 键仍叫 `spec_mtp_adaptive`。DFlash2 只有在后端暴露等价事务状态后才能接入同一调度器。

## Q6_K 验收与 FR

当前验收目标完整保留原始 Q6_K 主模型权重，不使用混合量化目标模型。

FR 只减少 MTP draft head 投影的行数，不改写 Q6_K 权重。它可以提高窄词表分布上的峰值，但接受率对语言和领域高度敏感。当前原始 Q6_K 峰值启用 8,192 个 token-ID 前缀，平衡型工作负载仍使用全词表 MTP。

## 当前实测配置

原始 Q6_K 峰值配置组合了：

- 原始 22,884,408,288 字节 Q6_K 制品；
- 带 8,192 行 draft-only token-ID 前缀的原生 MTP；
- 7 个 proposal 与 6 个循环状态快照；
- 固定 B11 目标验证容量与普通 CUDA Graph；
- 目标与草稿模型的批量 greedy CUDA 采样；
- 短批量 Flash Attention 关闭与完整 CUDA 层 offload；
- 高优先级 CUDA stream、物理核亲和性和单模型／单请求调度；
- 精确目标验证与确定性输出摘要。

当前干净九次采集的稳态中位数为 172.835 token/s，最低 171.298，最高 175.533；采集前共享 Windows 显示 GPU 已有 5–8% 利用率。较安静主机的历史高水位为 176.6109 token/s；同一制品全词表 K7/S7 对照为 147.0207 token/s。

通用短任务配置使用固定 K6/S6/B8，在当前 12 题、256-token 配对校准中达到 46.923 token/s；关闭推测执行为 28.713 token/s，提升 63.42%。两种模式的 12 个最终答案和 9/12 分数相同，接受率为 26.81%，每次目标前向提交 2.591 个已验证 token，回放为零。

固定形状比更高的名义接受率更重要。较早的无约束自适应 K 实验把同一代表性负载的接受率提高到 50.07%，却因验证形状变化减少 CUDA Graph 复用而降至 35.178 token/s。关闭 CUDA Graph 也把峰值工作负载降至 133.876 token/s。K、S、目标 batch 与图形状必须作为一个整体调优。

外部 DSpark 是独立路径，不与 DFlash 或原生 MTP 叠加。固定 K10/S6 的 context-512 峰值达到 169.324 token/s 中位数、167.102 最低值，相对配对目标对照提升 5.250 倍；三次 256-token 输出逐字一致，接受率 90.873%，回放为零。

单独的 DFlash2 原型始终使用不变的 Q6_K 目标，1.14 GB Q4 文件只作为
proposal 模型。12 题交叉校准中，两种模式都是 9/12，提取答案 12/12 一致，
完整输出 7/12 一致；请求全程吞吐均值从 29.702 提高到 45.143 token/s。
高接受率重复提示词达到 108.429 token/s 中位数与 98.230% 接受率。这是精确
上游 llama.cpp 的实验记录，不是 Power 原生执行，也没有达到 175 token/s。

context-1024 的跨领域诊断给出了更接近真实工作负载的边界：600 个请求全部成功，DSpark 请求全程吞吐为 32.678 token/s，目标对照为 22.618 token/s，提升 1.445 倍，并未观察到分数下降。但跨模式完整输出一致率只有 54/100，而且每个 DSpark 请求都进入过精确回放。因此该配置可用于显式实验，尚不能成为无损生产默认值。

当前自适应版本先在 S6 回滚窗口内探测。干净受控峰值的三次采集中位数为 164.756 token/s、最低 160.881；输出与回执摘要一致，接受率 92.713%，每次目标前向提交 9.8077 个 token，回放为零。100 题配对采集为 31.052 token/s，对照为 22.872 token/s，提升 1.358 倍；同样没有回放，24 个请求单向切回目标模型。

该采集的宽松／严格分数从 67/58 变为 69/56，包含 5 个宽松收益、3 个宽松损失、1 个严格收益和 3 个严格损失。完整输出一致 55/100；双方都未截断的 57 题答案一致。控制器消除了已知重放开销。

随后用干净的 5 题集合复测了全部答案损失，并加入 1 道正向对照。512-token 配置重复 3 轮，每轮都是 5/5 配对答案一致、0 损失；把预算提高到 1,024 token 后，5 题全部未截断且答案 5/5 一致。完整输出仍为 0/5 一致。控制器继续保持显式启用，因为目标模型裁决每个提交 token，并不承诺串行与批量 CUDA 轨迹逐字一致；这组损失样本结果也不能替代代表性质量矩阵。

完整执行路径参阅[优化手册](./optimization)，数字解读参阅[性能证据](./performance)，适配器 API、基准命令与验收规则参阅[规范推测解码设计](https://github.com/A3S-Lab/Power/blob/main/docs/speculative-decoding.md)，DFlash2 的原始边界与失败关闭状态参阅[专用证据包](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)。
