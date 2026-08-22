import "./OptimizationAtlas.css";

type Locale = "zh" | "en";

const atlasCopy = {
  zh: {
    title: "优化发生在整个执行路径。",
    intro:
      "Power 把可移植的运行时机制、模型 crate 的选择、后端内核和主机调度分开。每项优化都有明确边界、精确回退和可复验证据。",
    action: "查看完整优化手册",
    ownership: "职责",
    layers: [
      {
        title: "图与内核",
        owner: "运行时 + 后端",
        body: "模型声明有限形状；运行时复用稳定执行路径并在最后一个消费者后释放中间张量，后端按工作形状选择图捕获、注意力与已审查融合算子。",
        techniques: ["有限形状", "CUDA Graph", "Flash Attention", "算子融合"],
      },
      {
        title: "张量数据路径",
        owner: "通用运行时",
        body: "确定性微批处理合并调用，标准张量批次保持顺序；相邻已审查图通过设备驻留句柄传递结果，只在最终边界实体化。",
        techniques: ["微批处理", "张量批次", "设备驻留图链", "单次实体化"],
      },
      {
        title: "推测事务",
        owner: "运行时 + 适配器",
        body: "Prompt lookup、n-gram 与原生 MTP 共用精确提交协议。Proposal 宽度、状态快照、词表投影、接受率和回放由配置与证据约束。",
        techniques: ["精确目标校验", "K / S 形状", "MTP / FR", "回滚防护"],
      },
      {
        title: "调度与副本",
        owner: "运行时 + 主机",
        body: "有界准入、连续批处理和独占会话副本共享同一设备容量门。主机配置可以叠加高优先级 stream、物理核亲和性与单服务调度。",
        techniques: ["设备准入", "连续批处理", "会话副本", "GPU 优先级"],
      },
      {
        title: "权重与驻留",
        owner: "通用运行时",
        body: "已验证的 mmap、位置读取与直接读取进入同一权重层级；LFRU、预取、分组暂存、局部镜像和热集替换都保留原始张量身份。",
        techniques: ["LFRU", "异步预取", "驻留计划", "局部镜像"],
      },
      {
        title: "测量与证明",
        owner: "运行时 + 客户端",
        body: "候选配置交替执行并要求输出一致；吞吐、尾延迟、质量、设备路径和回退写入可固定哈希的证据，由客户端决定是否接受。",
        techniques: ["双顺序 A/B", "输出等价", "质量门槛", "硬件证据"],
      },
    ],
  },
  en: {
    title: "Optimization spans the whole execution path.",
    intro:
      "Power separates portable runtime mechanisms, model-crate decisions, backend kernels, and host scheduling. Every optimization has a boundary, an exact fallback, and reproducible evidence.",
    action: "Read the optimization playbook",
    ownership: "Owner",
    layers: [
      {
        title: "Graphs and kernels",
        owner: "Runtime + backend",
        body: "Models declare finite shapes. The runtime reuses stable execution paths and releases intermediates after their last consumer; backends select graph capture, attention, and reviewed fusion for each workload shape.",
        techniques: ["Finite shapes", "CUDA Graph", "Flash Attention", "Kernel fusion"],
      },
      {
        title: "Tensor data path",
        owner: "Shared runtime",
        body: "Deterministic microbatching coalesces calls while canonical tensor batches preserve order. Adjacent reviewed graphs pass device-resident handles and materialize only at the final boundary.",
        techniques: ["Microbatching", "Tensor batches", "Resident graph chains", "One materialization"],
      },
      {
        title: "Speculation transaction",
        owner: "Runtime + adapter",
        body: "Prompt lookup, n-gram, and native MTP share one exact commit protocol. Proposal width, state snapshots, vocabulary projection, acceptance, and replay remain configuration- and evidence-bound.",
        techniques: ["Exact target check", "K / S shapes", "MTP / FR", "Rollback guard"],
      },
      {
        title: "Scheduling and replicas",
        owner: "Runtime + host",
        body: "Bounded admission, continuous batches, and exclusive session replicas share one physical-device gate. Host profiles may add priority streams, physical-core affinity, and single-service scheduling.",
        techniques: ["Device admission", "Continuous batching", "Session replicas", "GPU priority"],
      },
      {
        title: "Weights and residency",
        owner: "Shared runtime",
        body: "Verified mmap, positional, and direct reads enter one weight hierarchy. LFRU, prefetch, grouped staging, partial mirrors, and hot-set replacement retain canonical tensor identity.",
        techniques: ["LFRU", "Async prefetch", "Residency plans", "Partial mirrors"],
      },
      {
        title: "Measurement and proof",
        owner: "Runtime + client",
        body: "Candidate settings run in both orders and must preserve output. Throughput, tail latency, quality, device path, and fallback become hash-pinned evidence that the client decides whether to accept.",
        techniques: ["Two-order A/B", "Output parity", "Quality gates", "Hardware evidence"],
      },
    ],
  },
} as const;

export function OptimizationAtlas({
  href,
  locale,
}: {
  href: string;
  locale: Locale;
}) {
  const copy = atlasCopy[locale];

  return (
    <section
      className="power-section power-optimization"
      aria-labelledby="optimization-title"
    >
      <header className="power-optimization__header">
        <div>
          <h2 id="optimization-title">{copy.title}</h2>
          <p>{copy.intro}</p>
        </div>
        <a href={href}>
          {copy.action} <span className="power-arrow" aria-hidden="true">→</span>
        </a>
      </header>

      <div className="power-optimization__ledger">
        {copy.layers.map((layer) => (
          <article key={layer.title}>
            <div className="power-optimization__name">
              <h3>{layer.title}</h3>
              <small><span>{copy.ownership}</span>{layer.owner}</small>
            </div>
            <p>{layer.body}</p>
            <ul>
              {layer.techniques.map((technique) => (
                <li key={technique}>{technique}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  );
}
