---
title: 复现实验
description: 从离线证据验真到 RTX 4090 完整复跑，复现 A3S Power 的 Qwen3.8-27B 性能边界。
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
| 工作形状 | 1 次预热 + 9 次实测；每次生成 1,024 token；batch 14；greedy |
| 稳态门槛 | 中位数 175 token/s；已提交结果 **176.6109**，最低 173.2630；7 / 9 个样本不低于 175 |
| 全词表对照 | 中位数 147.0207 token/s，最低 146.0917 |
| 12 题请求全程校准 | 关闭 MTP 29.713；全词表 47.032；前缀 FR 37.290 token/s |
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

当前原始 Q6_K 门槛记录的干净源码 revision 为 `eb6aeda59561eff3e4e7592704cab6fc863b72c7`。从更新的干净 revision 复跑是允许的，但结果应作为新实验保存，不能覆盖已提交 JSON。

## 2. 先做无需模型的离线验真

这个命令不加载 22.88 GB 模型，也不需要 NVIDIA GPU。它验证 14 个文件 SHA-256，并重新计算样本数、中位数、最低值、质量分数、请求全程吞吐、接受率、重放次数与确定性输出身份：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\tools\verify-qwen38-q6k-evidence.ps1 -Json
```

成功时进程退出码为 `0`，输出包含：

```json
{
  "status": "passed",
  "verified_file_hashes": 14,
  "quality": {
    "completed_requests": 900,
    "request_wide_tokens_per_second": 83.22814601950864
  },
  "pure_q6": {
    "full_vocabulary_k7_s7_median": 147.020656574707,
    "prefix_fr8192_k7_s6_median": 176.6108685085471
  }
}
```

任一文件字节或统计值变化，脚本都会返回非零退出码并指出不匹配字段。

## 3. 对齐验收主机

176.61 token/s 是下面这台机器上的边界，不是跨硬件承诺：

| 层级 | 验收环境 |
| --- | --- |
| OS | Windows 11 build 22631 |
| GPU | NVIDIA GeForce RTX 4090，24,564 MiB，compute capability 8.9 |
| CPU | Intel Xeon w5-2445，10 核 / 20 线程 |
| 驱动与 CUDA | NVIDIA 610.74；CUDA 12.6 |
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

补丁工具必须确认绑定层与内嵌 llama.cpp 补丁都已应用。runner 会拒绝任何不是独占 `llama.cpp` 的后端。

## 5. 校验模型与输入

先把原始 Q6_K 制品注册为 `qwen3.8-27b-q6-k`，再设置 Power 数据目录。下面三个仓库输入的哈希必须完全一致：

| 输入 | SHA-256 |
| --- | --- |
| `prompt.txt` | `d95a5e4dad822ba9c84138f7a120017318bcb3a6a90e77246a8ec4ede0e65d89` |
| 原始 Q6_K 全词表 K7/S7 ACL | `eb445101c1e33a035c9b1d120fec12d9b21e6ce1b2fe5486ad46bee52878a588` |
| 原始 Q6_K 前缀 FR8192 K7/S6 ACL | `9b1213df972ea3731010a1fa72b0d553ba73da42f31e92eaa4fecd3156cbf2ef` |

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

关闭占用 GPU 的程序，并在允许锁定 GPU 时钟的终端中执行：

```powershell
$benchmarkRoot = 'D:\models\a3s-power\qwen38\benchmark'
$powerHome = 'D:\models\a3s-power\qwen38\power-home'

.\tools\run-qwen38-q6k-benchmark.ps1 `
  -Label pure-q6-fr8192-k7s6-replay `
  -Config .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\pure-q6-mtp7-snap6-fr8192-host-staged.acl `
  -PromptFile .\docs\benchmarks\qwen3.8-27b-q6k-rtx4090\prompt.txt `
  -BenchmarkRoot $benchmarkRoot `
  -PowerHome $powerHome `
  -ModelHash 562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727 `
  -MaxTokens 1024 -NumBatch 14 -WarmupRuns 1 -Samples 9 `
  -MinimumTokensPerSecond 175 -ProcessPriority High `
  -ProcessorAffinityMask 349525 -LockGpuClockMHz 2745 `
  -TargetDirectory target-native-sm89-ninja `
  -RequireHighPerformancePowerPlan -RequireCleanTree
```

runner 在 `finally` 中恢复 GPU 时钟，并把失败报告也保留下来，便于区分真实性能回归与环境争用。

## 7. 验收新结果

在 `$benchmarkRoot` 中保留以下文件：

- `pure-q6-fr8192-k7s6-replay.json`：9 个原始样本、统计值和输出摘要；
- `pure-q6-fr8192-k7s6-replay.environment.json`：Git、二进制、模型、ACL、提示词、GPU、进程与电源状态；
- 对应 stdout / stderr 日志：后端初始化与失败诊断。

只有同时满足以下条件才算通过：9 个请求都生成 1,024 token；稳态中位数不低于 175 token/s；所有输出 SHA-256 一致；模型身份精确匹配；后端独占；工作树干净；要求的主机控制真实生效。

完整的 [Windows/CUDA 长版流程](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/REPRODUCE.md) 还包含配对全词表对照、原始 Q6_K 12 题校准与此前的混合制品门槛；[质量矩阵协议](https://github.com/A3S-Lab/Power/blob/main/docs/benchmarks/qwen3.8-27b-q6k-rtx4090/quality/README.md#reproduce) 说明如何重复现有 100 题 × 3 轮测试。
