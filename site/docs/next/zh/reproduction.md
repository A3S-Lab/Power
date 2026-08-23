---
title: 复现实验
description: 从固定离线证据到 RTX 4090 完整复跑，复现 A3S Power 当前 Q6_K 执行路径案例。
---

# 复现实验

这套流程把“复现”拆成两个可验收层级：任何机器都能完成**离线证据验真**；满足固定软硬件条件的机器可以继续完成**性能复跑**并生成一份新的环境回执。两者使用同一组模型、提示词、ACL 与输出身份。

:::warning 边界说明
当前原始 Q6_K 采集记录了干净源码 revision 与精确二进制身份。从更新 revision 复跑仍属于新实验，必须保留自己的 Git revision、二进制哈希和环境回执。此前披露 dirty worktree 的混合制品记录继续作为历史证据保留。
:::

## 验收基线

| 项目 | 固定值 |
| --- | --- |
| 模型制品 | 原始 Q6_K GGUF，22,884,408,288 字节 |
| 模型 SHA-256 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| 峰值模式 | 前缀 FR8192 MTP K7/S6；目标模型精确校验 |
| 工作形状 | 1 次预热 + 9 次实测；每次生成 1,024 token；batch 11；greedy；短批量 Flash Attention 关闭 |
| 最新精确构建采集 | **174.4133 token/s 中位数**；最低 172.7230；最高 177.1497；4 / 9 个样本不低于 175 |
| 较安静主机历史高水位 | **176.6109 token/s 中位数**；最低 173.2630；7 / 9 个样本不低于 175 |
| 全词表对照 | 中位数 147.0207 token/s，最低 146.0917 |
| 历史 12 题配对校准 | 关闭 MTP 28.713；固定 K6/S6/B8 为 46.923 token/s；提升 63.42% |
| 输出 SHA-256 | `a54538eaaf6cc0b8b43cbafd489c7779f0f5206c93d5034fd3a16f4366a90523` |

## 1. 获取源码并冻结实验身份

在 Windows PowerShell 中执行：

```powershell
git clone https://github.com/A3S-Lab/Power.git
Set-Location Power

$powerCommit = (git rev-parse HEAD).Trim()
$dirtyFiles = @(git status --porcelain)
if ($dirtyFiles.Count -ne 0) {
  throw 'A clean worktree is required'
}
$powerCommit
```

最新峰值采集的干净源码 revision 为 `da2c1dd5a2c6a573ef8be7789de4a67fdb2a0eb0`，当前质量矩阵为 `64aef15ddff7232c6261385700c8a912d1ed0963`。从更新的干净 revision 复跑是允许的，但必须保留独立证据，不能覆盖已有记录。

## 2. 先做无需模型的离线验真

当前纯 Q6_K 校验器不加载 22.88 GB 模型，也不需要 NVIDIA GPU。它固定整个
紧凑证据载荷，并重新计算 600 请求质量矩阵：

```powershell
py -3.13 .\tools\test_qwen38_q6_quality_evidence.py
py -3.13 .\tools\qwen38_q6_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\quality\pure-q6-rtx4090-3x.evidence.json `
  --json
```

它会重新计算 Q6_K 自回归的 23.642 token/s，以及同一 Q6_K 使用全词表 MTP
后的 41.035 token/s。可选的 `--require-lossless` 会按设计失败，因为完整输出
一致率是 50/100，严格评分有 2 个损失。

### 验证历史峰值与校准证据

下面的历史校验器验证 23 个文件 SHA-256，并重新计算较早的峰值、混合制品
质量和配对校准记录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

成功时进程退出码为 `0`，输出包含：

```json
{
  "status": "passed",
  "verified_file_hashes": 23,
  "quality": {
    "completed_requests": 900,
    "request_wide_tokens_per_second": 83.22814601950864
  },
  "pure_q6": {
    "full_vocabulary_k7_s7_median": 147.020656574707,
    "prefix_fr8192_k7_s6_median": 176.6108685085471
  },
  "deep_optimization": {
    "peak": {
      "median_decode_tokens_per_second": 172.8353133057359,
      "minimum_decode_tokens_per_second": 171.29810355919784
    },
    "general": {
      "target_only_tokens_per_second": 28.71272184998198,
      "mtp_tokens_per_second": 46.92338764288924,
      "speedup_percent": 63.4236833695329,
      "paired_final_answers": 12,
      "fallback_replays": 0
    }
  }
}
```

任一文件字节或统计值变化，脚本都会返回非零退出码并指出不匹配字段。

## 3. 对齐验收主机

这些 token/s 数值是下面这台机器上的边界，不是跨硬件承诺：

| 层级 | 验收环境 |
| --- | --- |
| OS | Windows 11 build 22631 |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB，compute capability 8.9 |
| CPU | Intel Xeon w5-2445，10 核 / 20 线程 |
| 驱动与 CUDA | NVIDIA 610.74；CUDA UMD 13.3；构建工具链按环境回执固定 |
| 工具链 | Rust 1.97.1、受支持的 MSVC、CMake、Ninja、libclang |
| 主机控制 | High Performance 电源计划；进程优先级 High；GPU 2745 MHz |
| CPU 亲和性 | `0x55555`（十进制 `349525`，只适用于该 CPU 拓扑） |

更换 GPU、驱动、时钟、显示负载或 CPU 拓扑后仍可运行同一协议，但必须把结果标为新平台；不要为了通过门槛而复制不适用的亲和性掩码。

## 4. 构建固定 CUDA 配置

```powershell
cargo fetch
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\apply-llamacpp-power-patches.ps1

$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_CUDA_ARCHITECTURES = '89'
cargo build --release --bins `
  --target-dir target-native-sm89-ninja `
  --no-default-features `
  --features llamacpp-cuda,llamacpp-mtp-fr
```

补丁工具必须确认绑定层、MTP/FR 与高优先级 CUDA stream 三组补丁都已应用。runner 会拒绝任何不是独占 `llama.cpp` 的后端。

## 5. 校验模型与输入

先把原始 Q6_K 制品注册为 `qwen3.8-27b-q6-k`，再设置 Power 数据目录。下面三个仓库输入的哈希必须完全一致：

| 输入 | SHA-256 |
| --- | --- |
| `prompt.txt` | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| 原始 Q6_K 全词表 K7/S7 ACL | `eb445101c1e33a035c9b1d120fec12d9b21e6ce1b2fe5486ad46bee52878a588` |
| 当前原始 Q6_K 前缀 FR8192 K7/S6/B11 ACL | `674d3a36e0f0019c9e39e60994ea40eee0477615827464edee1fb9627a74cdec` |
| 当前原始 Q6_K 前缀 FR8192 K6/S6/B8 ACL | `b4f3db4229bfad05371bbed0ce1fec165aa2b05279405078aa8f7721721abb37` |

```powershell
$powerHome = 'D:\models\a3s-power\qwen38\power-home'
$manifestPath = Join-Path $powerHome `
  'models\manifests\qwen3.8-27b-q6-k.json'
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$model = Get-Item -LiteralPath $manifest.path

if ($model.Length -ne 22884408288) { throw 'Unexpected model size' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $model.FullName).Hash -ne
    '562FBF760503008F118E5DF38DE5B3E97992D1F693F475815631198547486727') {
  throw 'Unexpected model hash'
}
```

## 6. 完整复跑前缀 FR8192 峰值

关闭不必要的 GPU 程序，并在允许锁定 GPU 时钟的终端中执行。这里使用零性能门槛采集真实测量；只有安静主机的独立服务门槛才应要求每个样本都不低于 175：

```powershell
$benchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark'
$powerHome = 'D:\models\a3s-power\qwen38\power-home'

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-fr8192-k7s6-b11-cudahigh `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap6-fr8192-rtx4090-throughput.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot $benchmarkRoot `
  -PowerHome $powerHome `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 11 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 0 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -CudaHighPriority `
  -MaximumIdleGpuUtilizationPercent 8 -IdleGpuSampleCount 3 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

runner 在 `finally` 中恢复 GPU 时钟，并把失败报告也保留下来，便于区分真实性能回归与环境争用。

## 7. 验收新结果

在 `$benchmarkRoot` 中保留以下文件：

- `pure-q6-fr8192-k7s6-b11-cudahigh.json`：9 个原始样本、统计值和输出摘要；
- `pure-q6-fr8192-k7s6-b11-cudahigh.environment.json`：Git、二进制、模型、ACL、提示词、GPU、进程与电源状态；
- `pure-q6-fr8192-k7s6-b11-cudahigh.preflight.json`：启动前身份与主机控制检查；
- 对应 stdout / stderr 日志：后端初始化与失败诊断。

只有同时满足以下条件才是有效采集：9 个请求都生成 1,024 token；所有输出 SHA-256 一致；模型身份精确匹配；后端独占；工作树干净；高优先级 stream、亲和性、时钟与电源控制真实生效。是否达到部署门槛由调用方根据该主机的独立 SLO 决定，不能把历史 176.61 直接当成当前共享桌面的失败阈值。

## 8. 验证只使用 Q6_K 目标的 DFlash2 原型

两种配对模式都使用同一个 22.88 GB Q6_K 目标。1.14 GB Q4 DFlash2 制品只
负责 proposal，不是目标模型结果。无需模型或 GPU 即可验证峰值与三轮交叉
质量证据：

```powershell
py -3.13 .\tools\dflash2_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dflash2\evidence.json `
  --json
```

校验器会重新计算重复提示词下 35.380 对 108.429 token/s、固定 12 题负载下
29.702 对 45.143 token/s、答案一致 12/12 与完整输出一致 7/12。加入
`--require-production-default` 必须失败，因为逐字输出门槛未通过，而且 Power
当前绑定尚不执行 DFlash2。[完整 DFlash2 指南](https://github.com/A3S-Lab/Power/tree/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dflash2)
固定了上游 PR 27342、模型与运行时哈希、CUDA 构建参数、主机控制和配对命令。

## 9. 复现原生 DSpark 门槛

外部 DSpark 使用保持不变的 22,884,408,288 字节 Q6_K 目标和固定摘要的 1.10 GB DSpark Q4 制品。先运行四个无需模型与 GPU 的校验器：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-dspark-evidence.ps1 -Json

py -3.13 .\tools\qwen38_quality_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\evidence.json `
  --json

py -3.13 .\tools\dspark_adaptive_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\adaptive\evidence.json `
  --json

py -3.13 .\tools\dspark_quality_followup_evidence.py verify `
  --evidence .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\dspark\quality\followup-evidence.json `
  --json
```

第一个验证 context-512 峰值：目标对照 32.249 token/s，DSpark K10/S6 中位数 169.324、最低 167.102 token/s，三次输出和回执完全一致。第二个验证 context-1024 的 600 请求质量采集：22.618 对 32.678 token/s、1.445 倍提升，以及分数、回放与全部配对任务向量。加入 `--require-production-default` 应当失败，因为完整输出一致率只有 54/100；该 K10/S6 矩阵是诊断证据，不是无损默认值。

第三个校验器绑定当前干净的按请求自适应采集：峰值中位数 164.756 token/s、最低 160.881，峰值输出与回执摘要一致，回放为零；100 题负载为 22.872 对 31.052 token/s，提升 1.358 倍，并重新计算全部配对任务向量。候选有 5 个宽松收益和 3 个损失，完整输出一致 55/100，因此加入 `--require-production-default` 同样必须失败。

第四个校验器绑定干净提交 `7bdeb960f5a38ea7515c67a12636a29198fd95f6` 上的损失样本复测：512 token 交叉运行 3 轮，另做 1 轮 1,024-token 配对。每轮都是 5/5 答案一致、0 收益、0 损失；1,024-token 配对中 5 题全部正常结束且答案 5/5 一致。512-token 吞吐为 30.521 对 24.967 token/s。完整输出一致仍为 0/5，因此加入 `--require-production-default` 应当失败。

复跑时使用哈希锁定的 `dspark/quality/divergence-v1.selection.json` 与 `run-qwen38-quality-matrix.ps1`。性能复测参数为 `-MaxTokensOverride 512 -NumCtx 1024 -Repetitions 3`；无截断质量检查参数为 `-MaxTokensOverride 1024 -NumCtx 2048 -Repetitions 1`。完整主机控制与可复制命令见[复测协议](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/dspark/quality/README.md#adaptive-truncation-follow-up)。

证据包中的性能命令会记录 2745 MHz GPU 时钟锁定请求、高优先级 CUDA stream、High 进程优先级、`0x55555` 亲和性、干净工作树、GPU 空闲门槛与至少 23,000 MiB 可用显存。这些是本机采集条件，不是其他 CPU 或 GPU 拓扑的默认配置。

完整的 [Windows/CUDA 长版流程](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) 还包含 DSpark 质量矩阵命令、配对全词表对照、原始 Q6_K 12 题校准与此前的混合制品门槛；[质量矩阵协议](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) 说明如何重复现有 100 题 × 3 轮测试。
