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

当前[干净 revision 的 CPU/CUDA 完整捕获](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/release-contract-windows-20260821/README.md)可回放为一个经过验证的两平台部分 bundle。在 Metal 与机密 GPU 捕获补齐前，严格的四平台 v1 策略仍不会通过。

## 生产阻断条件

- 严格验证器未包含硬件验证能力；
- 缺少或错误的启动度量与制品 pin；
- 保存的证据缺少原始报告字节；
- 厂商证书获取、解析或签名验证失败；
- nonce 过期，或模型、策略、输入、输出、设备摘要不匹配；
- 严格路径收到模拟报告或 `tee_type=none`。

证书服务、缓存行为、证据保存与失败策略参阅[硬件验证器运维文档](https://github.com/A3S-Lab/Power/blob/main/docs/hardware-verifier-operations.md)。
