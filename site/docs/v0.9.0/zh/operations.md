---
title: 部署运维
description: A3S Power 后端能力、构建配置、服务端点、制品存储与生产边界。
---

# 部署运维

后端是 Power 共享执行与证据契约背后的能力提供方。应根据模型格式、平台和信任边界选择后端，而不是把后端名称当作系统架构。

## 构建配置

| Feature | 作用 | 原生依赖 |
| --- | --- | --- |
| `mistralrs` | 默认 Candle 后端，支持 GGUF、SafeTensors、视觉与嵌入 | 无 C++ 推理引擎 |
| `llamacpp` | 成熟 GGUF 后端，支持原生 MTP | CMake、C++ 编译器、libclang |
| `llamacpp-cuda` | llama.cpp 的 CUDA 执行 | CUDA toolkit |
| `llamacpp-mtp-fr` | 实验性缩减词表 MTP draft 投影 | 对固定 llama.cpp 源码应用已审查补丁 |
| `picolm` | 面向受限 TEE 内存的纯 Rust 层流式 GGUF 后端 | 无 C/C++ 推理引擎 |
| `embedded-cuda` / `embedded-metal` | 模型自有嵌入式图的加速器支持 | 平台工具链 |
| `tls` / `vsock` | RA-TLS 与 A3S Box guest-host 传输 | 平台相关 |
| `hw-verify` | AMD SEV-SNP 验证；Intel TDX 在 v0.9.0 中不支持生产使用 | 平台密码学依赖与 AMD KDS 网络访问 |

```bash
# 默认托管服务
cargo build --release

# 无监听器嵌入式运行时
cargo build --release --no-default-features --features embedded-inference

# 纯 Rust 层流式 TEE 服务
cargo build --release --no-default-features --features tee-minimal

# 带 CUDA 的 llama.cpp
cargo build --release --no-default-features --features llamacpp-cuda
```

`llamacpp-mtp-fr` 独立成 profile，是因为它会修改固定版本的源码。普通 `llamacpp` 构建不需要该实验补丁。

## 服务端点

| 方法 | 端点 | 作用 |
| --- | --- | --- |
| `GET` | `/health` | 就绪状态、已加载模型、后端能力与 TEE 状态 |
| `POST` | `/v1/chat/completions` | 对话、工具、结构化输出、视觉与 SSE 流式响应 |
| `POST` | `/v1/completions` | 文本补全与 SSE 流式响应 |
| `POST` | `/v1/embeddings` | 嵌入推理 |
| `GET` | `/v1/models` | 已注册模型 |
| `POST` | `/v1/models/pull` | 可续传 ModelScope 或 Hugging Face 下载 |
| `GET` | `/v1/attestation` | 绑定 nonce 与模型的 TEE 证据 |
| `GET` | `/metrics` | Prometheus 指标 |

健康检查与模型检查端点公开实际生效的非敏感配置，使基准和部署自动化能够拒绝配置漂移。

## 制品安装

制品安装器要求提供预期文件名、最大字节数与 SHA-256 摘要。它将数据流式写入私有暂存文件，验证精确字节，再在跨进程锁下原子提交。离线策略找不到已验证制品时会失败关闭。

托管模型仓库默认位于 `~/.a3s/power`，采用内容寻址。模型别名指向 manifest，而不会削弱 blob 身份。

## 生产边界

- 除非经过审查的传输策略另有要求，开发服务只绑定回环地址。
- 显式选择 RA-TLS 或 vsock；构造嵌入式运行时不会替调用方选择传输。
- 模拟 TEE 模式只能用于开发。
- 不得因为 CPU TEE 放置就声称 GPU 推理具备机密性。
- 保存证明证据时保留原始报告字段。
- 混合量化与词表缩减 draft 都是需要质量门槛、与负载相关的技术。
- 每次性能验收都同时保存模型字节、ACL、二进制哈希、驱动与主机控制信息。

## 供应链与存储

纯 Rust `tee-minimal` 路径减少原生推理依赖；llama.cpp 路径则用更大的原生工具链换取成熟 GGUF 与 CUDA 能力。审计时必须以实际发布的 feature profile 为准。

- [供应链审计](https://github.com/A3S-Lab/Power/blob/main/docs/supply-chain.md)
- [可验证存储基准协议](https://github.com/A3S-Lab/Power/blob/main/docs/storage-benchmark.md)
- [项目路线图](https://github.com/A3S-Lab/Power/blob/main/ROADMAP.md)
- [版本变更记录](https://github.com/A3S-Lab/Power/blob/main/CHANGELOG.md)

Rust API 类型与 feature flag 参阅 [docs.rs/a3s-power](https://docs.rs/a3s-power)。
