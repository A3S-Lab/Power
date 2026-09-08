# A3S Power

<p align="center">
  <a href="https://a3s-lab.github.io/Power/"><img src="./site/docs/public/a3s-os-logo.png" width="72" alt="A3S OS"></a>
</p>


<p align="center">
  <strong>Language / 语言:</strong>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

<p align="center">
  <img src="./assets/readme/hero.svg" width="100%" alt="A3S Power routes model-owned graphs and hosted API requests through bounded admission, accelerator execution, canonical receipts, and caller-owned verification">
</p>

<p align="center">
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/ci.yml?branch=main&amp;style=flat-square&amp;label=CI" alt="CI status"></a>
  <a href="https://github.com/A3S-Lab/Power/actions/workflows/pages.yml"><img src="https://img.shields.io/github/actions/workflow/status/A3S-Lab/Power/pages.yml?branch=main&amp;style=flat-square&amp;label=docs" alt="Documentation deployment status"></a>
  <a href="https://a3s-lab.github.io/Power/"><img src="https://img.shields.io/badge/docs-ZH-2864e8?style=flat-square" alt="A3S Power Chinese documentation"></a>
  <a href="https://a3s-lab.github.io/Power/en/"><img src="https://img.shields.io/badge/docs-EN-2864e8?style=flat-square" alt="A3S Power English documentation"></a>
  <a href="https://crates.io/crates/a3s-power"><img src="https://img.shields.io/crates/v/a3s-power?style=flat-square&amp;color=2864e8" alt="a3s-power on crates.io"></a>
  <a href="https://docs.rs/a3s-power"><img src="https://img.shields.io/docsrs/a3s-power?style=flat-square" alt="a3s-power API documentation"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-17181a?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <a href="#measured-not-promised">证据</a>·
  <a href="#choose-the-boundary">曲面</a>·
  <a href="#quick-start">快速入门</a>·
  <a href="#one-runtime-contract">架构</a>·
  <a href="#optimization-without-hidden-shortcuts">优化</a>·
  <a href="#verification-and-release-gates">验证</a>·
  <a href="https://a3s-lab.github.io/Power/">文档</a>
</p>

A3S Power 是一个模型中立的 Rust 执行层，用于推理。模型crate
保持其拓扑、分词器、预处理、可变状态和质量
政策。电源提供了它们周围的共享边界：工件身份，
设备放置、准入、取消、有界状态、执行收据、
和独立验证。

相同的合约支持无侦听器的嵌入式库，
OpenAI 兼容服务、经过验证的工件配置以及最小的
层流 TEE 配置文件。运行时核心不会在 Qwen 或任何
其他模范家庭。

> [!IMPORTANT]
> `main` 包含 v1.0.0 候选源。最新发布的箱子和
> API文档仍在
> [v0.9.0](https://crates.io/crates/a3s-power/0.9.0)；使用基于源的
> 以下命令适用于当前 v1 API。 v1 标签不会发布，直到
> 严格的四平台证据捆绑和经过验证的注释标签均通过。

## 衡量，而非承诺

Power 通过其真实的流 API 记录性能并发布
输入、原始样本、环境收据、输出身份和离线
验证者。下面的 Qwen3.8-27B 结果是一个固定的 llama.cpp/CUDA
Windows 11、RTX 4090 和 Intel Xeon w5-2445 上的集成。他们是
这些确切工作负载的证据，而不是引擎或服务的范围
SLA。

每个活动行中的目标都是相同的未触及的 22,884,408,288 字节 Q6_K
神器。 Q4 文件仅作为辅助提议者出现（如有说明）。

|路径|测量结果|接受边界|
| ---| ---: | ---|
| Q6_K 自回归控制 | 23.642 请求范围令牌 | 3 x 100 项固定任务； 67/100 宽松和 60/100 严格 |
| Q6_K 全词汇MTP | **41.035 个请求范围的令牌； 1.736x** |相同的 3 x 100 任务和相同的目标字节； 67/100 宽松，58/100 严格；选择加入 |
| Q6_K MTP/FR 峰形 | **174.413 代币/秒中位数；最低 172.723** |九次 1,024 个令牌运行，一次输出摘要；不稳定的 175 代币/秒下限 |
|仅目标前缀重用 | **23.5299x 后端预填充； 13.1593x TTFT** |五个冷/暖对重复使用 9,740 个提示令牌；仅重复上下文延迟 |
| Q6_K + DFlash2提议者| **144.453解码； 63.182 个端到端令牌** |五个精确的合成对；更广泛的 12 任务运行保留了 12/12 个答案，但只有 7/12 个完整输出 |

这些行使用不同的工作负载形状，不得将它们视为不同的行进行比较
是一个基准。固定任务分数是质量指标，而不是智力
测量。精确的目标验证证明承诺的目标权威
代币；它不保证字节相同的散文。

[Benchmark index](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/README.md) ·
[Q6_K-only offline evidence](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/pure-q6-rtx4090-3x.evidence.json) ·
[Exact reproduction](docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) ·
[Performance documentation](https://a3s-lab.github.io/Power/en/performance)

## 选择边界

选择适合产品的最窄表面。

|表面|当 | 时使用它网络行为 |
| ---| ---| ---|
| **嵌入式运行时** | Rust 模型箱拥有经过审查的图表，并且需要共享设备、调度、状态和收据。 |没有侦听器、模型中心、下载或子进程。 |
| **托管服务** |现有客户需要聊天、完成、嵌入、模型生命周期、指标或证明。 |显式 HTTP、RA-TLS 或 vsock 传输。 |
| **工件供应者** |产品安装精确的版本锁定模型捆绑包。 |策略门控下载；经过验证的 blob 仍可离线重用。 |
| **最少的 TEE 服务** |受约束的 enclave 需要纯 Rust、层流式 GGUF 路径。 |传输仍然是一个明确的功能选择。 |

## 快速开始

### 运行当前托管服务

直接从 `main` 安装，而 v1 仍然是候选版本：

~~~bash
cargo install --git https://github.com/A3S-Lab/Power.git --locked a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
~~~

在另一个终端中：

~~~bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
~~~

对于已发布的 v0.9.0 CLI：

~~~bash
cargo install a3s-power --version 0.9.0 --locked
~~~

默认情况下，模型和内容寻址的 blob 位于 `~/.a3s/power` 下。套装
`A3S_POWER_HOME` 搬家。

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

构造 `EmbeddedRuntime` 永远不会打开监听器或下载模型。
调用者提供经过审查的图表并保留语义状态；电源
有限制的执行和证据合同。

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

流响应在 `[DONE]` 之前发出执行收据。

## 一份运行时合约

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

|权力拥有|模型拥有的crate拥有|
| ---| ---|
|类型化设备、有界图执行、不透明驻留张量和聚合内存预算 |架构、拓扑、层、内核和算术 |
|入场、微批处理、会话副本、取消、截止日期和安置 |标记化、预处理、后处理和生成策略 |
|工件身份、镜子、驻留计划和执行收据 |资产修改、转换、张量合约和质量门 |
| TEE 隐私、证明绑定、密封状态和验证者输入 | KV 或循环布局和语义可变状态 |

这种分裂是故意的。形状轮廓名称、模型系列字符串以及
会话身份对于核心来说是不透明的。语言、视觉、OCR、嵌入、
音频、多模式、科学和调用者拥有的图表可以共享运行时
无需向 Power 添加架构调度。

### 三个第一原则

|约束|运行时结果 |
| --- | --- |
|内存、计算、队列和传输都是有限的。 |请求通过明确的限制进入，共享一个物理设备门，保持可取消状态，并且不能默默地过度使用声明的状态。 |
|模型名称并不标识执行。 |收据将工件、策略、设备路径、输入、输出和后备身份与规范摘要绑定在一起。 |
|服务器不能成为其自身声明的信任根。 |客户端选择可接受的测量、哈希值、运行时策略、GPU 证据和收据字段。 |

阅读[Embedded Inference Architecture](docs/embedded-inference-architecture.md)
以获得完整的合同模型。

## 无隐藏快捷方式的优化

电源不会暴露一种模糊的“快速模式”。每一层都有自己的所有者，
测量、后备和收据身份。

|层 |机制|规则|
| ---| ---| ---|
|图和核 |有限形状配置文件、CUDA Graph 重用、特定于配置文件的 Flash Attention 以及审查的融合路径 |模型定义了形状含义；丢失或过大的形状失败或使用显式摘要绑定后备。 |
|张量运动 |确定性微批处理、执行批处理、设备驻留图链和最终实现 |不兼容的设备永远不会触发隐藏的跨设备复制。 |
|推测解码 |快速查找、n-gram、草稿模型、MTP、DFlash、DFlash2 或 DSpark，并进行精确目标验证 |仅接受接受的代币提交；不受支持的显式策略无法关闭。 |
|前缀重用 |具有有限 LRU 和 TTL 的租户、端点和模型范围的 KV/循环上下文 | `prompt_cache_key` 是显式的；不受支持的后端会返回错误而不是忽略它。 |
|分布式请求生命周期 |有界运行时在一个执行 ID 下将类型化阶段准备/执行/中止与状态准备/发布/消费/中止组成 |默认服务器不通告P/D能力；经过审查的注入适配器必须拥有模型执行、注册内存、传输完整性和清理功能。 |
|日程安排 |共享设备准入、有界队列、取消、截止日期、会话副本和主机控制 |副本声明在加载前保留其全部居民预算。 |
|重量和存储|内容寻址工件、mmap/mlock 策略、经过验证的镜像、预取和有限驻留 |回退返回到原始工件而不更改张量身份。 |
|推出 |二阶 A/B 运行、输出哈希、代表性质量门、硬件收据和离线重播 |在接受政策通过之前，更快的配置文件不会成为默认配置文件。 |

本机 MTP 和键控 llama.cpp 提示缓存会话尚未组成：
显式 MTP 对于键控请求无法关闭，而 `auto` 选择仅目标。
DFlash、DFlash2 和 DSpark 是替代外部草案合约，而不是
可堆叠模式。

[Optimization playbook](docs/optimization-playbook.md) ·
[Speculative decoding](docs/speculative-decoding.md) ·
[Prompt-prefix cache](docs/prompt-prefix-cache.md) ·
[Shape profiles](docs/shape-profiles.md) ·
[Session replicas](docs/session-replicas.md) ·
[Device-resident graphs](docs/device-resident-graphs.md)

## 后端是能力

后端在相同的资源和证据背后实施特定于模型的工作
合同。他们没有定义 Power 核心的架构。

|特色|能力|原生要求|
| ---| ---| ---|
| `mistralrs` |默认 GGUF、SafeTensors、视觉和嵌入后端 |没有 C++ 推理引擎 |
| `llamacpp` |成熟的 GGUF 后端，具有原生 MTP 支持 | CMake、C++ 编译器和 libclang |
| `llamacpp-cuda` | llama.cpp 的 CUDA 执行 | CUDA工具包|
| `llamacpp-external-draft` |类型化 DFlash、DFlash2 和 DSpark 工件合约 |已审核固定 llama.cpp 源代码的补丁 |
| `llamacpp-mtp-fr` |实验性缩减词汇草稿投影 |已审核固定 llama.cpp 源代码的补丁 |
| `picolm` |用于受限 TEE 内存的 Pure-Rust 层流 GGUF |没有 C 或 C++ 推理引擎 |
| `embedded-cuda` / `embedded-metal` |模型拥有的嵌入图的加速器 |平台工具包|
| `tls` / `vsock` / `hw-verify` | RA-TLS、来宾-主机传输和 AMD SEV-SNP 验证 |特定于平台的依赖关系和信任根 |

<details>
<summary>建立档案</summary>

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

在 Windows 上，在第一次固定 llama.cpp 签出之前启用 Git 长路径：

~~~powershell
git config --global core.longpaths true
~~~

</details>

## 政策明确

该服务从 `~/.a3s/power/config.acl` 或从路径读取 A3S ACL
传递到`a3s-power serve --config`。

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

无效的 ACL、范围、哈希、策略、执行配置文件或不受支持
显式后端在推理之前失败。 `prefill-decode` 仅被接受为
一个封闭的`serving_execution`块，不能被环境覆盖
变量。其确切的配置文件摘要是规范推理策略的一部分
通过认证使用。 TEE 部署添加了验证者拥有的模型哈希值，
测量和严格的政策；模拟证明从未通过严格要求
验证。

内置组合保持聚合状态。下游分解构建
必须注入精确的配置文件绑定 `StateTransferService` 并且
`ServingPhaseExecutor` 至 `PowerServerBuilder`；单独的任一服务都是
启动错误。组合根将这一对包装并组装成一个
`DistributedServingRuntime`，这是单个请求级别的生命周期，
准备来源。仅当运行时时，Power 才会发布配置的 P/D 角色
与不可变的配置文件匹配并可以接受工作。单独运输完成
永远不会被视为成功解码。
每个注入的转接头都被`BoundedStateTransferService`包裹，
它将广告的功能缩小到不可变的本地角色并强制执行
进程纪元、快速失败传输容量、幂等租赁、单调
最后期限、到期收割、有界中止和故障关闭清理运行状况。的
包装的适配器仍然拥有注册的内存和真实的数据路径。
运行时在传输前准备解码目标，发布预填充
仅在阶段执行后才状态，在开始之前消耗验证状态
解码，并保留流取消所有权直到终止。的
经过身份验证的内部请求流 API 将这些操作公开给网关，
将每个调用绑定到当前工作进程和执行配置文件摘要。
跨进程一致性套件启动单独的预填充和解码 Power
处理并证明经过身份验证的 HTTP 流、加密的不透明状态
切换、对等丢失故障、重新启动纪元失效以及优雅的清理。
它的后端和环回传输是测试装置，而不是导出的适配器。
该存储库仍然没有提供具体的分布式后端/传输对，因此
这不是端到端的 llm-d 部署声明。

## API 接口

|方法|端点|目的|
| ---| ---| ---|
| `GET` | `/health` |准备情况、加载模型、TEE 状态和版本化工作人员观察 |
| `POST` | `/v1/chat/completions` |聊天、工具、结构化输出、愿景和 SSE |
| `POST` | `/v1/completions` |文本补全和 SSE |
| `POST` | `/v1/embeddings` |嵌入推理 |
| `POST` | `/internal/v1/distributed-serving/decode/prepare` |准备配置文件绑定的解码目标 |
| `POST` | `/internal/v1/distributed-serving/prefill/execute` |执行预填充并发布不透明模型状态 |
| `POST` | `/internal/v1/distributed-serving/decode/execute` |使用状态并返回版本化 NDJSON 解码输出 |
| `POST` | `/internal/v1/distributed-serving/abort` |幂等地回收一个分布式执行 |
| `GET` | `/v1/models` |注册型号 |
| `POST` | `/v1/models` |注册权重和键入辅助工件 |
| `POST` | `/v1/models/pull` |可恢复的 ModelScope 或 Hugging Face 拉力 |
| `GET` | `/v1/attestation` |随机数和模型绑定的 TEE 证据 |
| `GET` | `/metrics` |普罗米修斯指标 |

GGUF 注册接受类型化适配器、投影仪和外部草稿对象。
Power 测量文件本身并记录准确的长度和 SHA-256
身份；严格的 TEE 启动拒绝遗留的仅路径辅助工件。

## 验证和发布门

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

Power 验证 AMD SEV-SNP 签名，将策略可见字段绑定到确切的
签名的报告，检查随机数新鲜度和 RA-TLS 绑定，并可以验证
NVIDIA GPU/NVSwitch 证据和 NRAS 裁决。英特尔 TDX 目前发出
本地 TDREPORT，但在审核 DCAP 报价/QVL 之前未能严格验证
路径存在。

严格的 v1 发布策略需要四个不同的捕获：CPU、CUDA、
原生 Apple 硅金属，以及经过验证的 SEV-SNP/NVIDIA 机密
图形处理器。该标签必须指向冻结源的仅证据子项，并且它
必须是 GitHub 验证的带注释的签名。托管或虚拟金属，
本地 CUDA 重新标记为机密、混合源/证据提交，以及
轻量级或未经验证的标签都无法关闭。

[Production release gate](docs/release-evidence-gate.md) ·
[v1 support matrix](docs/v1-support-matrix.md) ·
[External hardware capture](docs/external-release-capture.md) ·
[Hardware verifier operations](docs/hardware-verifier-operations.md)

### 安全边界

- 模拟 TEE 模式仅供开发。
- CPU TEE 放置不会使普通 GPU 卸载变得保密。
- 有效提示摘要仅适用于确定性文本路径。
- 减少词汇量的 FR 和辅助提议者是表演技术，
  不是普遍的质量保证。
- 来自一种模型、主机或工作负载的性能证据不会转移到
  另一个没有新捕获的。

## 文档

|从这里开始 |它回答什么 |
| ---| ---|
| [Documentation home - Chinese](https://a3s-lab.github.io/Power/) |默认 `next` 文档 |
| [Documentation home - English](https://a3s-lab.github.io/Power/en/) |英文`next`文档 |
| [Getting started](https://a3s-lab.github.io/Power/en/getting-started) |安装、嵌入使用、服务使用和首次请求 |
| [Architecture](https://a3s-lab.github.io/Power/en/architecture) |所有权、执行、状态和接收边界 |
| [Optimization](https://a3s-lab.github.io/Power/en/optimization) |图、张量、推测、调度、存储和证据方法 |
| [Performance](https://a3s-lab.github.io/Power/en/performance) |当前测量、质量边界和限制 |
| [Reproduction](https://a3s-lab.github.io/Power/en/reproduction) |离线验证和全硬件回放 |
| [Verification](https://a3s-lab.github.io/Power/en/verification) |证明声明、客户端策略和失败行为 |
| [Operations](https://a3s-lab.github.io/Power/en/operations) |功能配置文件和部署控制|

该存储库还将详细的合约保留在代码附近：

- [Model-owned shape profiles](docs/shape-profiles.md)
- [Model-neutral session replicas](docs/session-replicas.md)
- [Device-resident reviewed graph chains](docs/device-resident-graphs.md)
- [Keyed prompt-prefix cache](docs/prompt-prefix-cache.md)
- [Distributed-serving worker observation](docs/distributed-serving-observation.md)
- [Model-neutral speculative decoding](docs/speculative-decoding.md)
- [Supply-chain audit](docs/supply-chain.md)
- [Storage benchmark](docs/storage-benchmark.md)
- [Tensor-batch cost benchmark](docs/tensor-batch-benchmark.md)
- [Roadmap](ROADMAP.md) 和 [changelog](CHANGELOG.md)

签入的`site/docs/v1.0.0`树是候选文档快照，
没有证据表明 v1 标签已经通过了生产大门。

## 开发

从此crate运行检查，而不是从 monorepo 根运行检查：

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

CI 检查格式、Clippy 特征矩阵、重点测试和集成测试，
无侦听器的嵌入式构建、发布合同、基准证据以及
GitHub Pages 神器。

## A3S 生态系统

|项目|关系 |
| ---| ---|
| [A3S Box](https://github.com/A3S-Lab/Box) | SEV-SNP 或 TDX MicroVM 内的主机电源 |
| [A3S Gateway](https://github.com/A3S-Lab/Gateway) |路线推断流量 |
| [A3S Runtime](https://github.com/A3S-Lab/Runtime) |供应部署单位合同|
| [A3S Code](https://github.com/A3S-Lab/Code) |使用本地推理和验证包 |
| [A3S Event](https://github.com/A3S-Lab/Event) |分发平台事件 |

欢迎提出问题和设计讨论
[Discord](https://discord.gg/XVg6Hu6H)。

## 许可证

[MIT](LICENSE)
