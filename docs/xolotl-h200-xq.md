# 8× H200 上的 DeepSeek-V4.1-Flash-1017 XQ

状态：准入与非流式 HTTP 转发已实现。本说明不把通过的单测当成 8 卡 H200 实测，也不把本机 4090 的单测当成一次推理。

单卡 RTX 4090 是另一条放置，见文末。它是 Xolotl 的单卡专家缓存（`--experts gpu` 加 `--rank`），不是这条 8 卡层切分。

Power 不链接 Xolotl 内核，也不启动 `ds41_generate`。它只接受唯一一种放置，把启动参数交给外部监督进程，再把一次请求转到已经在跑的 `serve_ds41.py`。

相关代码：`src/backend/xolotl/`、`src/config.rs` 的 `XolotlConfig`、`src/config/acl.rs`。

## 权重

准入的是社区 XQ 包（原 GSQ），例如 `IndrasMirror/DeepSeek-V4.1-Flash-1017-XQ-Uncensored`。文件名须包含 `V41-XQ`、`XQ-1017` 或 `1017-XQ`。`family = xolotl` 也会被认领。文件名含 `-UD-` 或 `-UD_` 的 Unsloth 包不会被认领。

XQ 专家混合约 2.093 bpw（PTQ1_0、Q2_0、Q3_K）。这不是官方 1017 的 FP8 / FP4 检查点，也不是 Unsloth UD-IQ1，也不是 llama.cpp 的 TQ1_0。该社区包在量化前对 attention 输出和 shared-expert down 做了 rank-2 拒绝编辑；路由专家未改。官方 FP8 / FP4 仍走各自的服务路径，不进入这个后端。

Engram 分片留在 NVMe。`--engram` 至少给出两个目录。Linux 用 `:` 分隔。Windows 用 `;`，这样盘符里的冒号不会被拆开，例如 `D:\engram-a;E:\engram-b`。

## 唯一接受的启动

`ds41_generate` 遇到未知参数会以状态 2 退出。本计划发出的参数是：

```text
ds41_generate --serve -m <xq.gguf> --engram <dir1:dir2> --rank <rank-1017.bin> --backend cuda --experts gpu --layer-split 5,10,15,20,25,30,35 --split-device 0,1,2,3,4,5,6,7
```

环境变量只有：

```text
CUDA_VISIBLE_DEVICES=<八个 nvidia-smi 序号，按阶段顺序>
```

`--split-device` 用的是 `CUDA_VISIBLE_DEVICES` 重映射之后的可见序号 `0,1,2,3,4,5,6,7`，不是原始 nvidia-smi 序号。

40 层被切成 8 段，每段 5 层。7 个边界对应 8 张卡。省略 `--split-device` 时引擎最多保留两张可见卡。`layer-split = auto` 是在层数一半处切一刀，本后端拒绝它。

计划不发出 `--gpus`、`--split-skip-if-fits`、`-ngl`、`--ngl`、`--batch`。`STRATA_DSPARK` 和 `STRATA_SERVE_DSPARK` 保持未设置。

二进制文件名必须是 `ds41_generate` 或 `ds41_generate.exe`。Rank 文件名必须包含 `rank-1017` 且以 `.bin` 结尾。`xolotl.model` 若填写，必须与清单路径相同，且路径本身要通过 XQ 文件名检查。

## 就绪

监督进程可以启动上述命令。进程就绪的条件是日志里有这一整行，且没有 `not indexer-safe`：

```text
ds41: 7-way layer split at 5 10 15 20 25 30 35 across GPUs 0 1 2 3 4 5 6 7
```

这里是 7-way，因为 7 个边界加最后一段才是 8 张卡。引擎可能按 GGUF 元数据改写不满足 indexer-safe 的边界。改写后的日志不能通过就绪检查。相等的 5 层切分是否会被改写，要有权重和实机日志才能知道。本仓库没有做这次证明。

层切分同时只跑一个请求。`max_concurrent_requests` 和 `num_parallel` 都必须是 1。8 张卡用来放下专家权重。每一段多一次交接。本文不声称 8 段流水会更快。

## 硬件

配置的 8 个序号必须互不相同，并且每一张都在 `nvidia-smi` 清单里。多出来的卡可以忽略。每张卡按这个顺序检查：

1. 产品名里有完整词 `H200`。`H20`、`H100`、`H2000` 不通过。
2. 计算能力正好是 9.0。
3. 显存至少 120 GiB（128849018880 字节）。`nvidia-smi` 的 `memory.total` nounits 是 MiB。

`xolotl.cuda_architecture` 必须是字符串 `9.0`。CMake 用的整数 `90` 会被拒绝。编译 Xolotl 时使用 `-DCMAKE_CUDA_ARCHITECTURES=90`。H200 是 sm_90。Xolotl 已发布的架构说明里没有把 sm_90 列为已测量目标。

## ACL

默认配置省略整个 `xolotl` 块。`enabled = false` 时清单仍会被认领，加载失败关闭，不会落到 llama.cpp 或 mistral.rs。加载成功还要 `enabled = true`，并且二进制、模型、Engram 目录和 rank 文件都存在。

```acl
max_concurrent_requests = 1
num_parallel = 1

xolotl {
    enabled = true
    bin = "/opt/xolotl/ds41_generate"
    model = "/models/DeepSeek-V4.1-Flash-1017-V41-XQ.gguf"
    engram = "/models/engram-a:/models/engram-b"
    rank = "/models/v2/rank-1017-085.bin"
    upstream = "http://127.0.0.1:8150"
}
```

`gpus`、`layer_count`、`cuda_architecture`、`layer_split` 的默认值已经是 8 卡计划，可以不写。`upstream` 是 `serve_ds41.py` 的源站，例如 `http://127.0.0.1:8150`，不要带 `/v1`。`--serve` 本身是引擎的行协议（GEN / STOP / QUIT），不是 HTTP。HTTP 由 `serve_ds41.py` 提供，默认端口 8150。

## 请求

聊天和补全各向 `{upstream}/v1/chat/completions` 或 `{upstream}/v1/completions` 发送一次 JSON，并且 `stream` 固定为 false。客户端即使要求流式，这里也只转发这一次非流式响应，再把它拆成一个内容块和一个结束块。采样字段按客户端原样转发，缺省时不补写。

带图片或工具定义的聊天会被拒绝。补全拒绝图片。嵌入不可用。`upstream` 未设置、上游错误或进程未就绪时，错误信息里会带上完整命令和就绪日志行，并说明 Power 不启动 `ds41_generate`。

## 验证边界

在 `crates/power` 内，用本仓库的空 `[workspace]` 表避免加入 Aster 父工作区：

```text
C:\Users\admin\.cargo\bin\cargo.exe test --no-default-features --features server --lib -- xolotl default_backends_route test_config_roundtrip test_config_serialize_acl xolotl_block
```

这些测试注入假的 GPU 清单，不调用 `nvidia-smi`，不启动 `ds41_generate`，也不下载权重。绿色结果只说明准入、ACL 和一次 JSON 转发与上面的合同一致。

仍留在本仓库之外的部分：监督并拉起 `ds41_generate`、sm_90 二进制、XQ 权重、indexer-safe 边界的实机确认、工具调用，以及真正的 SSE 流。

## 单卡 RTX 4090

这是 Xolotl 在一张 24 GiB 卡上的专家缓存，不是上面的 8 卡层切分。热专家留在显存，`--experts gpu` 配合 `--rank` 决定谁常驻，Engram 仍在 NVMe。命令里没有 `--layer-split`，没有 `--split-device`，没有 `--gpus`，也没有 `STRATA_DSPARK`。

三条配置要一起写。少一条就失败关闭，不会落到 8 卡计划，也不会落到 llama.cpp：

- `cuda_architecture = "8.9"`。CMake 的整数 `89` 不行。
- `gpus` 只有一个 nvidia-smi 序号。`CUDA_VISIBLE_DEVICES` 就是这个序号。
- `layer_split = "none"`。

显卡产品名要以 `GeForce RTX 4090` 结尾。后面再跟 `Ti`、`D` 或 `Laptop` 就不认。计算能力必须是 8.9。`nvidia-smi` 的 `memory.total` 要在 23000 到 26000 MiB。本机这张卡是 24564 MiB、计算能力 8.9。

8 卡计划的检查不放宽。把这张 4090、一张 H20、一张 H100、一张 H2000，或一张计算能力 8.9 的卡填进 8 个 H200 槽位，仍然拒绝。默认配置仍是 8 卡计划；不改 `xolotl` 时 ACL 继续省略整个块。

```acl
xolotl {
    enabled = true
    bin = "D:\xolotl\ds41_generate.exe"
    model = "D:\xolotl\V41-XQ-1017.gguf"
    engram = "D:\engram-a;E:\engram-b"
    rank = "D:\xolotl\rank-1017-085.bin"
    upstream = "http://127.0.0.1:8150"
    gpus = [0]
    cuda_architecture = "8.9"
    layer_split = "none"
}
```

这条放置没有 7 段切分日志。日志里如果出现 layer split 或 `not indexer-safe`，单卡计划拒绝。Power 仍然不启动 `ds41_generate`。
