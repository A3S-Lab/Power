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
  --expected-measurement <96-character-launch-measurement-hex>
```

由验证器，而不是服务运营方，选择可接受的启动度量、制品哈希、运行策略、GPU 证据与回执字段。

## 硬件证据

| TEE | 严格验证 |
| --- | --- |
| AMD SEV-SNP | 原始报告解析、nonce 与度量绑定、VCEK 获取、ECDSA P-384 签名验证 |
| Intel TDX | 失败即关闭：当前本地 TDREPORT 不是可远程验证的 DCAP Quote，仍需实现 Quote 生成与 QVL 验证 |
| NVIDIA 机密 GPU | 新鲜设备声明、固件与拓扑策略、固定 NRAS verdict、GPU 执行摘要 |

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

受信任的构造路径由类型系统约束。严格的机密 GPU 验证会返回一个绑定到精确报告的不可构造证明；只有 `ReleaseCapture::promote_confidential_gpu` 能用它把有效的本地 CUDA 捕获提升为机密 GPU 捕获。原始报告、反序列化标签或调用方写入的布尔值都不能铸造这类发布证据。生成的 bundle 仍须由发布信任根认证。

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

## 生产发布信任链

从 v1 开始，硬件捕获绑定一个冻结的源码提交。它的直接子提交只能新增
`release/v<version>/release-evidence.json` 与对应的 SHA-256 pin。这个证据子提交必须已经位于 `main`，并由一个 GitHub 标记为 Verified 的附注标签直接指向。发布 CI 会验证两提交结构和四平台 bundle，随后从冻结的父提交构建二进制与 crate。轻量标签、未验证标签、脱离 `main` 的标签，或夹带额外改动的标签都会在发布前失败。

这种拆分消除了提交哈希的自引用：bundle 认证源码父提交，签名子提交再认证 bundle。仅有源码或历史基准文件，不能证明某个版本已经通过生产发布门禁。

## 复现外部硬件捕获

仓库中的流程会先在真实 Metal 设备上运行完整契约，再用同一个新鲜 nonce 绑定保留原字节的 NVIDIA evidence、远程 NRAS verdict、CPU TEE 报告、规范化 GPU 执行策略与模型自有加速器声明。`a3s-power-verify --promote-capture` 在同一进程内消费严格验证证明，并以禁止覆盖的方式创建机密捕获。

精确命令、ACL、设备 pin、失败条件与制品清单见[外部 Metal 与机密 GPU 发布捕获](https://github.com/A3S-Lab/Power/blob/main/docs/external-release-capture.md)。每个生产标签都必须携带同一源码父提交的 Metal 与机密 GPU 证明；硬件证据缺失或验证失败都会阻止发布。

## 生产阻断条件

- 严格验证器未包含硬件验证能力；
- 缺少或错误的启动度量与制品 pin；
- 保存的证据缺少原始报告字节；
- 厂商证书获取、解析或签名验证失败；
- nonce 过期，或模型、策略、输入、输出、设备摘要不匹配；
- 严格路径收到模拟报告或 `tee_type=none`。

证书服务、缓存行为、证据保存与失败策略参阅[硬件验证器运维文档](https://github.com/A3S-Lab/Power/blob/main/docs/hardware-verifier-operations.md)。
