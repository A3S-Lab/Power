---
title: 快速开始
description: 选择 A3S Power 的接入面，嵌入有界运行时，或启动 OpenAI 兼容服务。
---

# 快速开始

Power 提供两个推理入口与一个独立的制品安装入口。请选择产品真正需要的最窄边界。

| 产品需求 | 从这里开始 |
| --- | --- |
| 在 Rust crate 内拥有经过审查的模型图 | `embedded-inference` |
| 需要托管的对话、补全或嵌入 API | 默认 `server,mistralrs` 配置 |
| 安装锁定到精确修订版的制品包 | `artifact-provisioning` |
| 在受限 enclave 内运行层流式 GGUF 服务 | `tee-minimal` |

## 1. 嵌入运行时

只添加不启动网络监听器的推理能力：

```toml
[dependencies]
a3s-power = { version = "1.0.0", default-features = false, features = ["embedded-inference"] }
```

用明确的设备偏好和资源限制构造运行时：

```rust
use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};

fn main() -> Result<(), a3s_power::error::PowerError> {
    let runtime = EmbeddedRuntime::new(
        DevicePreference::Auto,
        InferenceLimits::default(),
    )?;

    println!("execution device: {}", runtime.device().name());
    Ok(())
}
```

创建 `EmbeddedRuntime` 不会绑定端口、启动监听器、下载模型或调用外部进程。模型 crate 提供经过审查的计算图并保留语义状态；Power 提供执行边界。

继续阅读[架构设计](/architecture)，了解完整职责划分。

## 2. 运行托管服务

安装默认服务配置，并只绑定本机回环地址：

```bash
cargo install a3s-power
a3s-power serve --host 127.0.0.1 --port 11434
```

在另一个终端拉取并打开一个小型 GGUF 模型：

```bash
a3s-power models pull Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
a3s-power chat Qwen/Qwen2.5-0.5B-Instruct-GGUF:q4_k_m
```

模型清单与内容寻址的 blob 默认保存在 `~/.a3s/power`。可通过 `A3S_POWER_HOME` 指定其他存储目录。

## 3. 发送 OpenAI 兼容请求

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "解释基于能力的安全模型。"}],
    "stream": true
  }'
```

对话与补全响应包含 `attestation_receipt` 及其 SHA-256 摘要。流式响应会在 `[DONE]` 前发出回执。

## 4. 显式配置策略

服务默认读取 `~/.a3s/power/config.acl` 中的 A3S ACL，也可通过 `a3s-power serve --config` 指定路径。

```text
host = "127.0.0.1"
port = 11434
max_loaded_models = 1
keep_alive = "5m"

flash_attention = true
num_parallel = 1

gpu {
  gpu_layers = -1
  main_gpu = 0
}
```

无效的 ACL、范围、策略、哈希或不受支持的显式后端会在推理开始前失败。生产 TEE 策略还需要由验证方掌控的度量值与模型哈希；参阅[独立验证](/verification)。

## 下一步

- [理解执行与职责边界](/architecture)
- [按图、张量、调度、权重和证据选择优化](/optimization)
- [检查 Qwen3.8 的实测性能与质量证据](/performance)
- [了解推测解码如何保持目标模型的绝对裁决权](/speculative-decoding)
- [选择后端与生产构建配置](/operations)
