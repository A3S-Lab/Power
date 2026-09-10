# A3S Power

<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <a href="https://a3s-lab.github.io/Power/"><img src="./site/docs/public/a3s-os-logo.png" width="72" alt="A3S OS"></a>
</p>

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power 将模型自有图与托管 API 请求经有界准入、加速器执行、规范回执与调用方自有验证进行路由">
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI 状态"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/pages.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/pages.yml?branch=main&amp;style=flat-square&amp;label=docs" alt="文档部署状态"></a>
  <a href="https://a3s-lab.github.io/Power/"><img src="https://img.shields.io/badge/docs-ZH-2864e8?style=flat-square" alt="A3S Power 中文文档"></a>
  <a href="https://a3s-lab.github.io/Power/en/"><img src="https://img.shields.io/badge/docs-EN-2864e8?style=flat-square" alt="A3S Power 英文文档"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=2864e8" alt="crates.io 上的 a3s-power"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power API 文档"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT 许可证"></a>
</p>

<p align="center">
  <a href="#实测而非承诺">证据</a> &middot;
  <a href="#选择边界">表面</a> &middot;
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#统一运行时契约">架构</a> &middot;
  <a href="#无隐藏捷径的优化">优化</a> &middot;
  <a href="#验证与发布门">验证</a> &middot;
  <a href="https://a3s-lab.github.io/Power/">文档</a>
</p>

A3S Power 是面向推理的模型无关 Rust 执行层。模型 crate 保留其拓扑、
分词器、预处理、可变状态与质量策略。Power 在其周围提供共享边界：
制品身份、设备放置、准入、取消、有界状态、执行回执与独立验证。

同一契约支持无监听器的嵌入式库、OpenAI 兼容服务、已验证制品供给，
以及最小的层流式 TEE 配置。运行时核心不对 Qwen 或任何其他模型族做分发。

> [!IMPORTANT]
> `main` 包含 v1.0.0 源码候选。最新已发布的 crate 与 API 文档仍为
> [v0.9.0](https://crates.io/crates/a3s-power/0.9.0)；请使用下方基于源码的命令
> 以使用当前 v1 API。在严格的四平台证据包与已验证的附注标签均通过之前，
> 不会发布 v1 标签。

## 实测而非承诺

Power 通过其真实流式 API 记录性能，并发布输入、原始样本、环境回执、
输出身份与离线校验器。下方 Qwen3.8-27B 结果是 Windows 11、RTX 4090 与
Intel Xeon w5-2445 上的一次固定 llama.cpp/CUDA 集成。它们是这些精确工作负载的证据，
而非引擎范围或服务 SLA。

每个活动行的目标都是同一未改动的 22,884,408,288 字节 Q6_K 制品。
Q4 文件仅在说明处作为辅助 proposer 出现。

| 路径 | 实测结果 | 验收边界 |
| --- | ---: | --- |
| Q6_K 自回归对照 | 23.642 请求级 token/s | 3 × 100 固定任务；宽松 67/100，严格 60/100 |
| Q6_K 全词表 MTP | **41.035 请求级 token/s；1.736×** | 相同 3 × 100 任务与相同目标字节；宽松 67/100，严格 58/100；需显式启用 |
| Q6_K MTP/FR 峰值形态 | **中位 174.413 token/s；最低 172.723** | 九次 1,024-token 运行、同一输出摘要；不是稳定的 175 token/s 下限 |
| 仅目标前缀复用 | **后端 prefill 23.5299×；TTFT 13.1593×** | 五对冷/热，复用 9,740 个 prompt token；仅重复上下文延迟 |
| Q6_K + DFlash2 proposer | **decode 144.453；端到端 63.182 token/s** | 五对精确合成；更广的 12 任务运行保持 12/12 答案但仅 7/12 完整输出 |

这些行使用不同的工作负载形态，不得当作同一基准比较。固定任务分数是质量代理，
而非智能度量。精确目标验证证明已提交 token 的目标权威性；
它不承诺字节级相同的散文。

[基准索引](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md) ·
[仅 Q6_K 离线证据](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json) ·
[精确复现](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) ·
[性能文档](https://a3s-lab.github.io/Power/en/performance)

## 选择边界

选择适合产品的最窄表面。

| 表面 | 适用场景 | 网络行为 |
| --- | --- | --- |
| **嵌入式运行时** | Rust 模型 crate 拥有经审阅的图，需要共享设备、调度、状态与回执。 | 无监听器、模型中心、下载或子进程。 |
| **托管服务** | 现有客户端需要 chat、completions、embeddings、模型生命周期、指标或证明。 | 显式 HTTP、RA-TLS 或 vsock 传输。 |
| **制品供给器** | 产品安装精确修订锁定的模型包。 | 策略门控下载；已验证 blob 可离线复用。 |
| **最小 TEE 服务** | 受限 enclave 需要纯 Rust、层流式 GGUF 路径。 | 传输仍为显式特性选择。 |

## 快速开始

### 运行当前托管服务

在 v1 仍为发布候选期间，直接从 `main` 安装：

~~~bash
cargo install --git https://github.com/A3S-Lab/Power.git --locked a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
~~~

在另一终端：

~~~bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
~~~

若使用已发布的 v0.9.0 CLI：

~~~bash
cargo install a3s-power --version 0.9.0 --locked
~~~

模型与内容寻址 blob 默认位于 `~/.a3s/power`。设置 `A3S_POWER_HOME` 可移动存储。

### 嵌入当前运行时

~~~toml
[dependencies]
a3s-power = { git = "https://github.com/A3S-Lab/Power.git", branch = "main", default-features = false, features = ["embedded-inference"] }
~~~

~~~rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;

    println!("execution device: {}", runtime.device().name());
    Ok(())
}
~~~

构造 `EmbeddedRuntime` 从不打开监听器或下载模型。调用方提供经审阅的图并保留语义状态；
Power 提供有界执行与证据契约。

### 发送 OpenAI 兼容请求

~~~bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Explain capability-based security."}],
    "prompt_cache_key": "shared-agent-prefix-v1",
    "stream": true
  }'
~~~

流式响应在 `[DONE]` 之前发出执行回执。

## 统一运行时契约

~~~text
model-owned graph                   OpenAI-compatible client
        |                                      |
   embedded API                           hosted service
        \                                      /
         +--------- shared Power core --------+
                            |
           artifact identity + bounded admission
                            |
              placement + reviewed execution
                            |
                   CPU / CUDA / Metal
                            |
                  canonical receipt
                            |
               independent verification
~~~

| Power 拥有 | 模型拥有 crate 拥有 |
| --- | --- |
| 类型化设备、有界图执行、不透明驻留张量与聚合内存预算 | 架构、拓扑、层、内核与算术 |
| 准入、微批、会话副本、取消、截止期限与放置 | 分词、预处理、后处理与生成策略 |
| 制品身份、镜像、驻留计划与执行回执 | 资产修订、转换、张量契约与质量门 |
| TEE 隐私、证明绑定、密封状态与校验器输入 | KV 或循环布局与语义可变状态 |

这一拆分是刻意的。形状配置文件名、模型族字符串与会话身份对核心不透明。
语言、视觉、OCR、嵌入、音频、多模态、科学与调用方自有图可共享运行时，
而无需向 Power 添加架构分发。

### 三条第一性原理

| 约束 | 运行时后果 |
| --- | --- |
| 内存、计算、队列与传输是有限的。 | 请求经显式限制进入，共享一个物理设备门，保持可取消，且不能静默超额承诺已声明状态。 |
| 模型名不能标识一次执行。 | 回执以规范摘要绑定制品、策略、设备路径、输入、输出与回退身份。 |
| 服务器不能成为自身声明的信任根。 | 客户端选择可接受的度量、哈希、运行时策略、GPU 证据与回执字段。 |

完整契约模型见 [Embedded Inference Architecture](docs/embedded-inference-architecture.md)。

## 无隐藏捷径的优化

Power 不暴露含糊的「快速模式」。每一层都有各自的所有者、度量、回退与回执身份。

| 层 | 机制 | 规则 |
| --- | --- | --- |
| 图与内核 | 有限形状配置、CUDA Graph 复用、配置特定 Flash Attention，以及经审阅的融合路径 | 模型定义形状含义；缺失或过大的形状失败，或使用显式摘要绑定的回退。 |
| 张量移动 | 确定性微批、执行批、设备驻留图链，以及一次最终物化 | 不兼容设备从不触发隐藏的跨设备拷贝。 |
| 推测解码 | Prompt lookup、n-gram、草稿模型、MTP、DFlash、DFlash2 或 DSpark，带精确目标验证 | 仅已接受 token 提交；不支持的显式策略失败关闭。 |
| 前缀复用 | 租户、端点与模型作用域的 KV/循环上下文，带有界 LRU 与 TTL | `prompt_cache_key` 是显式的；不支持的后端返回错误而非忽略它。 |
| 分布式请求生命周期 | 有界运行时在同一执行 ID 下组合类型化阶段 prepare/execute/abort 与状态 prepare/publish/consume/abort | 默认服务器不宣称 P/D 能力；经审阅的注入适配器必须拥有模型执行、已注册内存、传输完整性与清理。 |
| 调度 | 共享设备准入、有界队列、取消、截止期限、会话副本与主机控制 | 副本声明在加载前预留其完整驻留预算。 |
| 权重与存储 | 内容寻址制品、mmap/mlock 策略、已验证镜像、预取与有界驻留 | 回退返回原始制品，不改变张量身份。 |
| 滚动发布 | 两阶 A/B 运行、输出哈希、代表性质量门、硬件回执与离线重放 | 更快的配置在其验收策略通过之前不会成为默认。 |

原生 MTP 与带键的 llama.cpp prompt-cache 会话尚不能组合：
显式 MTP 对带键请求失败关闭，而 `auto` 选择仅目标模式。
DFlash、DFlash2 与 DSpark 是替代的外部草稿契约，而非可堆叠模式。

[优化手册](docs/optimization-playbook.md) ·
[推测解码](docs/speculative-decoding.md) ·
[Prompt 前缀缓存](docs/prompt-prefix-cache.md) ·
[形状配置](docs/shape-profiles.md) ·
[会话副本](docs/session-replicas.md) ·
[设备驻留图](docs/device-resident-graphs.md)

## 后端即能力

后端在同一资源与证据契约背后实现模型特定工作。它们不定义 Power 核心的架构。

| 特性 | 能力 | 原生要求 |
| --- | --- | --- |
| `mistralrs` | 默认 GGUF、SafeTensors、视觉与嵌入后端 | 无 C++ 推理引擎 |
| `llamacpp` | 成熟 GGUF 后端，支持原生 MTP | CMake、C++ 编译器与 libclang |
| `llamacpp-cuda` | llama.cpp 的 CUDA 执行 | CUDA 工具包 |
| `llamacpp-external-draft` | 类型化 DFlash、DFlash2 与 DSpark 制品契约 | 针对固定 llama.cpp 源的经审阅补丁 |
| `llamacpp-mtp-fr` | 实验性缩减词表草稿投影 | 针对固定 llama.cpp 源的经审阅补丁 |
| `picolm` | 面向受限 TEE 内存的纯 Rust 层流式 GGUF | 无 C 或 C++ 推理引擎 |
| `embedded-cuda` / `embedded-metal` | 模型自有嵌入式图的加速器 | 平台工具包 |
| `tls` / `vsock` / `hw-verify` | RA-TLS、宾主传输与 AMD SEV-SNP 验证 | 平台特定依赖与信任根 |

<details>
<summary>构建配置</summary>

~~~bash
# Default hosted service
cargo build --locked --release

# Listener-free embedded runtime
cargo build --locked --release --no-default-features --features embedded-inference

# Pure-Rust layer-streaming TEE service
cargo build --locked --release --no-default-features --features tee-minimal

# llama.cpp with CUDA
cargo build --locked --release --no-default-features --features llamacpp-cuda

# Strict verifier and release-promotion path
cargo build --locked --release --no-default-features \
  --features server,embedded-inference,hw-verify \
  --bin a3s-power-verify
~~~

在 Windows 上，首次检出固定 llama.cpp 之前启用 Git 长路径：

~~~powershell
git config --global core.longpaths true
~~~

</details>

## 策略是显式的

服务从 `~/.a3s/power/config.acl` 或传给 `a3s-power serve --config` 的路径读取 A3S ACL。

~~~acl
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
prompt_cache_max_entries = 1
prompt_cache_ttl_seconds = 300
worker_observation_ttl_seconds = 15
keep_alive = "5m"

serving_execution {
  profile = "aggregated"
}

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0
}
~~~

无效 ACL、范围、哈希、策略、执行配置或不支持的显式后端在推理前失败。
`prefill-decode` 仅作为闭合的 `serving_execution` 块被接受，且不能被环境变量覆盖。
其精确配置摘要是证明所用规范推理策略的一部分。TEE 部署添加校验器拥有的模型哈希、
度量与严格策略；模拟证明从不通过严格验证。

内置组合仍为聚合式。下游解聚构建必须通过 `PowerServerBuilder` 注入精确配置绑定的
`StateTransferService` 与 `ServingPhaseExecutor`；单独任一服务都是启动错误。
Empty/Unavailable 占位（`EmptyStateTransferService` / `EmptyServingPhaseExecutor`）
在注入具体适配器前同样失败关闭。注入端口声明
`ProductionAdapterContract::REQUIRED`（适配器拥有的内存注册、适配器拥有的传输完整性、
确认回收）；该合约是软件所有权边界，而非高速网络或生产就绪证据。
对钉住 `BufferedHostMemoryPullV1` 与 `AuthenticatedEncryptedTransport` 的配置，
Power 提供可注入的产品面对：`BufferedHostLoopbackStateTransfer` 与
`BufferedHostLoopbackPhaseExecutor`（`paired_for_profile`），经认证回环 TCP 移动
不透明主机缓冲，并在 Ready decode 前校验不透明一致性字节。仅传输回执从不算
decode 成功；不完整配对与 Empty 占位仍失败关闭。这不是 HSN、llama.cpp P/D 或
模型后端证据。
组合根将该对包装并组装为单一 `DistributedServingRuntime`，它是唯一的请求级生命周期与就绪源。
仅当运行时匹配不可变配置且可接受工作时，Power 才发布已配置的 P/D 角色。仅传输完成
从不计为成功 decode。每个注入的传输适配器由 `BoundedStateTransferService` 包装，
后者将宣称能力收窄到不可变本地角色，并通过共享 `AdmissionController` 强制执行
进程 epoch、失败快速 inflight 准入（`waiting_limit == 0`，ACL
`max_inflight_transfers`）、幂等租约、单调截止期限、过期回收、有界 abort 与
失败关闭的清理健康度。被包装的适配器仍拥有已注册内存与真实数据路径。
运行时在传输前准备 decode 目标，仅在阶段执行后发布 prefill 状态，在开始 decode
前消费已验证状态，并在终止前保留流取消所有权。阶段与传输租约仍为独立域，但在
组合时拒绝第二种准入策略形态。当运行时匹配不可变配置时，worker 观察投影该共享
失败快速 inflight 准入快照，而非 HTTP 请求限流器，并在取消或清理污染后保持
observation generation 单调。经认证的内部请求流 API 将这些操作暴露给 Gateway，
将每次调用绑定到当前 worker epoch 与执行配置摘要。跨进程一致性套件启动独立的
prefill 与 decode Power 进程，并证明经认证的 HTTP 流、加密不透明状态交接、
对等丢失失败、重启 epoch 失效、陈旧 Cloud deployment generation / 外源
peer-set 拒绝与优雅清理。其后端与回环传输是测试 fixture，
而非导出适配器。产品表面回环对仅覆盖不透明一致性组合，不是端到端 llm-d
或真实后端 P/D 声明。

## API 表面

| 方法 | 端点 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 就绪、已加载模型、TEE 状态与带版本的 worker 观察 |
| `POST` | `/v1/chat/completions` | Chat、工具、结构化输出、视觉与 SSE |
| `POST` | `/v1/completions` | 文本补全与 SSE |
| `POST` | `/v1/embeddings` | 嵌入推理 |
| `POST` | `/internal/v1/distributed-serving/decode/prepare` | 准备配置绑定的 decode 目标 |
| `POST` | `/internal/v1/distributed-serving/prefill/execute` | 执行 prefill 并发布不透明模型状态 |
| `POST` | `/internal/v1/distributed-serving/decode/execute` | 消费状态并返回带版本 NDJSON decode 输出 |
| `POST` | `/internal/v1/distributed-serving/abort` | 幂等回收一次分布式执行 |
| `GET` | `/v1/models` | 已注册模型 |
| `POST` | `/v1/models` | 注册权重与类型化辅助制品 |
| `POST` | `/v1/models/pull` | 可恢复的 ModelScope 或 Hugging Face 拉取 |
| `GET` | `/v1/attestation` | 与 nonce 及模型绑定的 TEE 证据 |
| `GET` | `/metrics` | Prometheus 指标 |

GGUF 注册接受类型化 adapter、projector 与 external-draft 对象。
Power 自行度量文件并记录精确长度与 SHA-256 身份；严格 TEE 启动拒绝仅路径的遗留辅助制品。

## 验证与发布门

~~~text
model bytes + resolved runtime policy
                  |
        canonical claims + fresh nonce
                  |
       CPU TEE report + GPU evidence
                  |
       request + prompt + output receipt
                  |
       independent client accepts or rejects
~~~

~~~bash
a3s-power-verify \
  --url http://127.0.0.1:11434 \
  --nonce <client-nonce-hex> \
  --model-hash <artifact-sha256-hex> \
  --inference-execution-digest <resolved-policy-sha256> \
  --auxiliary-artifacts-digest <portable-auxiliary-set-sha256> \
  --expected-measurement <launch-measurement-hex>
~~~

Power 验证 AMD SEV-SNP 签名，将策略可见字段绑定到精确签名报告，
检查 nonce 新鲜度与 RA-TLS 绑定，并可验证 NVIDIA GPU/NVSwitch 证据与 NRAS 裁决。
Intel TDX 当前发出本地 TDREPORT，但在经审阅的 DCAP Quote/QVL 路径存在之前，
严格验证失败。

严格 v1 发布策略要求四次不同捕获：CPU、CUDA、原生 Apple Silicon Metal，
以及经证明提升的 SEV-SNP/NVIDIA 机密 GPU。标签必须指向冻结源的仅证据子提交，
且必须是 GitHub 验证的附注签名。托管或虚拟 Metal、将本地 CUDA 重新标注为机密、
混合源/证据提交，以及轻量或未验证标签均失败关闭。

[生产发布门](docs/release-evidence-gate.md) ·
[v1 支持矩阵](docs/v1-support-matrix.md) ·
[外部硬件捕获](docs/external-release-capture.md) ·
[硬件校验器运维](docs/hardware-verifier-operations.md)

### 安全边界

- 模拟 TEE 模式仅用于开发。
- CPU TEE 放置不会使普通 GPU 卸载变为机密。
- 有效 prompt 摘要仅存在于确定性文本路径。
- 缩减词表 FR 与辅助 proposer 是性能技术，而非通用质量保证。
- 来自某一模型、主机或工作负载的性能证据，在无新捕获时不能迁移到另一场景。

## 文档

| 从这里开始 | 它回答什么 |
| --- | --- |
| [文档首页 - 中文](https://a3s-lab.github.io/Power/) | 默认 `next` 文档 |
| [文档首页 - 英文](https://a3s-lab.github.io/Power/en/) | 英文 `next` 文档 |
| [入门](https://a3s-lab.github.io/Power/en/getting-started) | 安装、嵌入式使用、服务使用与首次请求 |
| [架构](https://a3s-lab.github.io/Power/en/architecture) | 所有权、执行、状态与回执边界 |
| [优化](https://a3s-lab.github.io/Power/en/optimization) | 图、张量、推测、调度、存储与证据方法 |
| [性能](https://a3s-lab.github.io/Power/en/performance) | 当前测量、质量边界与限制 |
| [复现](https://a3s-lab.github.io/Power/en/reproduction) | 离线验证与完整硬件重放 |
| [验证](https://a3s-lab.github.io/Power/en/verification) | 证明声明、客户端策略与失败行为 |
| [运维](https://a3s-lab.github.io/Power/en/operations) | 特性配置与部署控制 |

仓库还将详细契约保持在代码附近：

- [模型自有形状配置](docs/shape-profiles.md)
- [模型无关会话副本](docs/session-replicas.md)
- [设备驻留经审阅图链](docs/device-resident-graphs.md)
- [带键 prompt 前缀缓存](docs/prompt-prefix-cache.md)
- [分布式服务 worker 观察](docs/distributed-serving-observation.md)
- [模型无关推测解码](docs/speculative-decoding.md)
- [供应链审计](docs/supply-chain.md)
- [存储基准](docs/storage-benchmark.md)
- [张量批成本基准](docs/tensor-batch-benchmark.md)
- [路线图](ROADMAP.md) 与 [变更日志](CHANGELOG.md)

检入的 `site/docs/v1.0.0` 树是候选文档快照，并不证明 v1 标签已通过生产门。

## 开发

在本 crate 中运行检查，而非从 monorepo 根目录：

~~~bash
cargo fmt --all -- --check
cargo test --locked --lib
cargo test --locked --no-default-features --features embedded-inference --lib
cargo test --locked --no-default-features --features picolm --lib
cargo clippy --locked --all-targets -- -D warnings

npm ci --prefix site
npm run typecheck --prefix site
npm run build --prefix site
npm run check:site --prefix site
~~~

CI 检查格式化、Clippy 特性矩阵、聚焦与集成测试、无监听器嵌入式构建、
发布契约、基准证据与 GitHub Pages 产物。

## A3S 生态

| 项目 | 关系 |
| --- | --- |
| [A3S Box](https://github.com/A3S-Lab/Box) | 在 SEV-SNP 或 TDX MicroVM 中托管 Power |
| [A3S Gateway](https://github.com/A3S-Lab/Gateway) | 路由推理流量 |
| [A3S Runtime](https://github.com/A3S-Lab/Runtime) | 提供部署单元契约 |
| [A3S Code](https://github.com/A3S-Lab/Code) | 消费本地推理与已验证包 |
| [A3S Event](https://github.com/A3S-Lab/Event) | 分发平台事件 |

问题与设计讨论欢迎在 [Discord](https://discord.gg/XVg6Hu6H) 进行。

## 许可证

[MIT](LICENSE)
