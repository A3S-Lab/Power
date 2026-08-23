---
title: 独立验证
description: 由客户端掌控 A3S Power 的模型身份、运行策略、TEE 报告、加速器证据与响应回执验证。
---

# 独立验证

只有真正依赖答案的一方掌控接受策略时，远程证明才有价值。因此 Power 把证据生成与证据接受彻底分离。

```text
模型字节 + 运行策略
          |
          v
规范化声明摘要 + 新鲜 nonce
          |
          v
CPU TEE 报告 + 可选 GPU 证据
          |
          v
请求、有效提示词与输出回执
          |
          v
独立客户端验证器接受或拒绝
```

## 回执绑定什么

响应回执可提交以下内容：

- 精确模型制品身份；
- 运行时与实际生效策略；
- 输入、有效提示词、解码、工具、输出与响应摘要；
- 选定加速器或精确回退路径；
- 融合批处理或异构设备网格证据；
- CPU TEE 报告与可选 GPU 机密计算声明。

无法可靠推导的字段保持缺失。例如，不透明的多模态 renderer 路径不会伪造有效提示词摘要。

## 构建严格验证器

```bash
cargo build --release --bin a3s-power-verify --features hw-verify
```

未启用 `hw-verify` 时，严格签名验证会失败关闭。显式 `--allow-offline` 绕过只用于 fixture 与离线检查，不是生产接受策略。

机密发布提升还需要模型中立的嵌入式契约类型：

```bash
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
```

## 验证运行中的服务

```bash
a3s-power-verify \
  --url https://power.example.com \
  --model your-model \
  --nonce <fresh-client-nonce-hex> \
  --model-hash <64-character-artifact-sha256> \
  --inference-execution-digest <resolved-power-policy-sha256> \
  --auxiliary-artifacts-digest <portable-auxiliary-set-sha256> \
  --expected-measurement <96-character-launch-measurement-hex>
```

先从已审查的 ACL 计算服务端策略摘要。模型还使用外部 draft、LoRA adapter 或
多模态 projector 时，再从部署 manifest 计算与本机路径无关的期望值：

```bash
a3s-power-verify --print-inference-execution-digest power.acl
a3s-power-verify --print-auxiliary-artifacts-digest model-manifest.json
```

新的本地模型报告始终声明推理策略，因此严格客户端必须固定其摘要。证明中声明
辅助制品时，还必须固定辅助制品摘要。该摘要覆盖制品角色、解码
契约、字节长度、制品哈希以及外部 draft 的目标绑定，不包含本机路径。

推理执行摘要覆盖完全解析后的 ACL 中的投机解码、MTP/FR、前缀缓存边界、
模型驻留、mmap/mlock、线程数、Flash Attention 和并行请求槽位；环境变量覆盖会
改变摘要。验收必须证明某一种解码器时，应使用显式 `spec_mode`；`auto` 证明的是
后端选择策略，不会伪装成已经预先选定 MTP、DFlash 或 DSpark。

由验证器，而不是服务运营方，选择可接受的启动度量、制品哈希、运行策略、GPU 证据与回执字段。

## 硬件证据

| TEE | 严格验证 |
| --- | --- |
| AMD SEV-SNP | 原始报告解析、nonce 与度量绑定、VCEK 获取、ECDSA P-384 签名验证 |
| Intel TDX | 失败即关闭：当前本地 TDREPORT 不是可远程验证的 DCAP Quote，仍需实现 Quote 生成与 QVL 验证 |
| NVIDIA 机密 GPU | 新鲜设备声明、固件与拓扑策略、固定 NRAS verdict、GPU 放置摘要与推理执行摘要 |

Power 默认在内存中缓存获取到的 AMD KDS 证书材料一小时。运营方可调整该缓存，但网络或证书失败在生产中仍然是阻断错误，除非存在显式审查过的离线证书设计。任何缓存设置都不能启用 TDX 验证。

## 配置严格策略

```text
tee_mode = true
tee_policy_mode = "strict"
redact_logs = true

expected_measurement "sev-snp" {
  digest = "<96-character measurement hex>"
}

model_hash "your-model" {
  digest = "sha256:<64-character artifact digest>"
}
```

严格策略拒绝模拟报告。CPU TEE 放置也不会自动让 GPU offload 具备机密性；GPU 路径需要经过验证的 `gpu-confidential` 声明。

## 模型中立的运行时发布契约

发布门禁与具体模型的质量评测相互独立。它把同一个 Power revision、精确权重和已审查图绑定到各平台自己的 shape profile 与 TEE 策略，并验证标量／批处理一致性、有界峰值内存、活动任务取消清理、队列超时、replica 恢复和显式精确回退。证据结构中没有 Qwen、GGUF、分词器、解码器或模型族分发字段。

受信任的构造路径由类型系统约束。严格的机密 GPU 验证会返回一个绑定到精确报告的不可构造证明；只有 `ReleaseCapture::promote_confidential_gpu` 能用它把有效的本地 CUDA 捕获提升为机密 GPU 捕获。原始报告、反序列化标签或调用方写入的布尔值都不能铸造这类发布证据。提升后的捕获会显式保留已接受的 48 字节启动度量、原始签名报告精确字节的 SHA-256、推理执行摘要与可选辅助制品摘要，最终 bundle 回放会直接校验全部字段。生成的 bundle 仍须由发布信任根认证。

组装前，验证每份跨主机传输的捕获：

```bash
a3s-power-tensor-batch-bench verify-release-capture \
  --capture <文件> \
  --platform <cpu|cuda|metal|confidential-gpu> \
  --power-version <版本> \
  --power-commit <revision>
```

命令会检查有界 JSON、规范摘要、平台与源码身份；回执明确标记为单捕获范围，只有严格的四平台 bundle 才能授权生产发布。

当前[干净 revision 的 CPU/CUDA 完整捕获](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/release-contract-windows-20260821/README.md)可回放为一个经过验证的两平台部分 bundle。在 Metal 与机密 GPU 捕获补齐前，严格的四平台 v1 策略仍不会通过。

## 离线验证基准证据

模型专项性能证据不是发布证明，但仍应失败关闭并可由第三方独立验证。DSpark 质量包固定了干净源码与服务端二进制、目标与 draft 制品、题目与 ACL 输入、6 份原始报告摘要、GPU 准入窗口、聚合指标和配对任务向量：

```bash
python3 tools/qwen38_quality_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/evidence.json \
  --json
```

该证据能通过完整性校验，同时明确标记为不具备生产默认资格。加入 `--require-production-default` 会因目标与 DSpark 完整输出一致率仅为 54/100 而失败。证据完整性、质量观察与部署准入是三个独立判断。

当前自适应证据把受控峰值与 100 题配对采集合并成一份路径无关文档，其中包含主机控制证明、原始采集摘要、运行时遥测与全部配对任务向量：

```bash
python3 tools/dspark_adaptive_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/adaptive/evidence.json \
  --json
```

它验证峰值中位数 164.756 token/s、最低 160.881、零回放，以及质量负载 1.358 倍的请求全程加速。由于矩阵包含 3 个配对宽松损失，且完整输出一致率仅为 55/100，生产默认校验仍会拒绝该配置。

损失样本复测证据独立固定了哈希锁定的 5 题集合、512/1,024-token 请求身份、基准工具哈希、主机控制、原始报告摘要和紧凑任务向量：

```bash
python3 tools/dspark_quality_followup_evidence.py verify \
  --evidence docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/followup-evidence.json \
  --json
```

它验证配对答案 0 损失，并在 1,024 token 下实现 5/5 未截断答案一致。完整输出一致仍为 0/5，所以损失样本诊断通过后，逐字输出生产门槛仍保持关闭。

## 生产发布信任链

从 v1 开始，硬件捕获绑定一个冻结的源码提交。它的直接子提交只能新增
`release/v<version>/release-evidence.json` 与对应的 SHA-256 pin。这个证据子提交必须已经位于 `main`，并由一个 GitHub 标记为 Verified 的附注标签直接指向。发布 CI 会验证两提交结构和四平台 bundle，随后从冻结的父提交构建二进制与 crate。轻量标签、未验证标签、脱离 `main` 的标签，或夹带额外改动的标签都会在发布前失败。

这种拆分消除了提交哈希的自引用：bundle 认证源码父提交，签名子提交再认证 bundle。仅有源码或历史基准文件，不能证明某个版本已经通过生产发布门禁。

创建标签前，先在本地运行与发布 CI 相同的失败关闭预检：

```bash
git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main
bash tools/verify-release-candidate.sh \
  --evidence-ref HEAD \
  --main-ref refs/remotes/origin/main
```

它只接受工作树干净、版本不低于 v1、变更日志已经收口、证据子提交位于远程
`main`，且严格四平台 bundle 能完整回放的候选版本。原生 Metal 与经
SEV-SNP/NVIDIA 证明提升的机密 GPU 证据仍然是硬要求；预检不会用模拟或部分
捕获替代它们。

## 复现外部硬件捕获

仓库中的流程会先在真实 Metal 设备上运行完整契约，再用同一个新鲜 nonce 绑定保留原字节的 NVIDIA evidence、远程 NRAS verdict、CPU TEE 报告、规范化 GPU 与推理执行策略、可选辅助制品集合，以及模型自有加速器声明。`a3s-power-verify --promote-capture` 在同一进程内消费严格验证证明，并以禁止覆盖的方式创建机密捕获。

把每台主机的完整原始制品放进独立的只读目录，清单必须位于该目录之外。传输前生成精确文件清单，评审主机收到文件后再完整回放：

```bash
cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  build-release-handoff --root ./metal-handoff --platform metal \
  --power-version "$power_version" --power-commit "$power_commit" \
  --output ./metal-handoff.manifest.json

cargo run --locked --release --no-default-features \
  --features embedded-inference \
  --bin a3s-power-tensor-batch-bench -- \
  verify-release-handoff --root ./metal-handoff --platform metal \
  --manifest ./metal-handoff.manifest.json \
  --power-version "$power_version" --power-commit "$power_commit"
```

文件被修改、缺失、增加，路径不安全，出现符号链接或 Windows reparse point，或者平台与源码版本被重标记，校验都会失败。无绝对路径的清单仍须由发布信任根认证，不能替代单项捕获验证和四平台 bundle。

精确命令、ACL、设备 pin、失败条件与制品清单见[外部 Metal 与机密 GPU 发布捕获](https://github.com/A3S-Lab/Power/blob/main/docs/external-release-capture.md)。每个生产标签都必须携带同一源码父提交的 Metal 与机密 GPU 证明；硬件证据缺失或验证失败都会阻止发布。

## 生产阻断条件

- 严格验证器未包含硬件验证能力；
- 缺少或错误的启动度量与制品 pin；
- 运行时声明 draft、adapter 或 projector 时，缺少或不匹配的辅助制品 pin；
- 保存的证据缺少原始报告字节；
- 厂商证书获取、解析或签名验证失败；
- nonce 过期，或模型、策略、输入、输出、设备摘要不匹配；
- 严格路径收到模拟报告或 `tee_type=none`。

证书服务、缓存行为、证据保存与失败策略参阅[硬件验证器运维文档](https://github.com/A3S-Lab/Power/blob/main/docs/hardware-verifier-operations.md)。
