import "./OptimizationAtlas.css";

type Locale = "zh" | "en";

const atlasCopy = {
  zh: {
    title: "速度来自整条执行路径。",
    intro:
      "Power 不提供一个含糊的“极速开关”。图、张量、推测解码、调度和权重分别优化；任何快路径都保留精确回退。",
    action: "查看全部优化方法",
    ownership: "负责",
    layers: [
      {
        title: "图与内核",
        owner: "模型 + 后端",
        body: "固定常用形状，复用 CUDA Graph；再按实际批量决定是否启用 Flash Attention 和融合算子。",
        techniques: ["有限形状", "CUDA Graph", "Flash Attention", "算子融合"],
      },
      {
        title: "张量路径",
        owner: "Power",
        body: "合并小请求，让相邻计算图直接传递设备内结果，只在最终输出时复制一次。",
        techniques: ["微批处理", "张量批次", "设备驻留", "单次复制"],
      },
      {
        title: "推测解码",
        owner: "Power + 后端",
        body: "草稿负责猜，目标模型负责验。只提交通过的 token；遇到不匹配就回到最后一个正确位置。",
        techniques: ["目标模型校验", "K / S 形状", "MTP / FR", "精确回滚"],
      },
      {
        title: "调度与副本",
        owner: "Power + 主机",
        body: "队列、连续批处理和会话副本共用同一设备预算；主机再配置 GPU 优先级和 CPU 亲和性。",
        techniques: ["设备准入", "连续批处理", "会话副本", "主机调度"],
      },
      {
        title: "权重与驻留",
        owner: "Power",
        body: "按需读取、预取和缓存热点权重；缓存不足时回到原始制品，张量身份始终不变。",
        techniques: ["LFRU", "异步预取", "驻留计划", "局部镜像"],
      },
      {
        title: "上线前验证",
        owner: "Power + 客户端",
        body: "同一输入交替测试新旧配置。速度更快、输出一致、质量过线，才允许替换原路径。",
        techniques: ["双顺序 A/B", "输出一致", "质量门槛", "硬件记录"],
      },
    ],
  },
  en: {
    title: "Speed comes from the whole execution path.",
    intro:
      "Power does not hide performance behind a vague fast-mode switch. Graphs, tensors, speculation, scheduling, and weights are tuned separately, with an exact fallback for every fast path.",
    action: "See every optimization",
    ownership: "Owned by",
    layers: [
      {
        title: "Graphs and kernels",
        owner: "Model + backend",
        body: "Fix the common shapes, reuse CUDA Graphs, then choose Flash Attention and fused kernels for the batch that actually runs.",
        techniques: ["Finite shapes", "CUDA Graph", "Flash Attention", "Kernel fusion"],
      },
      {
        title: "Tensor path",
        owner: "Power",
        body: "Merge small calls, pass device-resident results between adjacent graphs, and copy once at the final output.",
        techniques: ["Microbatching", "Tensor batches", "Device residency", "One copy"],
      },
      {
        title: "Speculative decoding",
        owner: "Power + backend",
        body: "The draft guesses and the target model checks. Only matching tokens are committed; a mismatch returns to the last correct position.",
        techniques: ["Target verification", "K / S shapes", "MTP / FR", "Exact rollback"],
      },
      {
        title: "Scheduling and replicas",
        owner: "Power + host",
        body: "Queues, continuous batches, and session replicas share one device budget. The host adds GPU priority and CPU affinity.",
        techniques: ["Device admission", "Continuous batching", "Session replicas", "Host scheduling"],
      },
      {
        title: "Weights and residency",
        owner: "Power",
        body: "Read on demand, prefetch, and keep hot weights resident. When cache space runs out, fall back to the original artifact without changing tensor identity.",
        techniques: ["LFRU", "Async prefetch", "Residency plans", "Partial mirrors"],
      },
      {
        title: "Validation before rollout",
        owner: "Power + client",
        body: "Run old and new settings in both orders on the same input. Replace the old path only when speed improves, output matches, and quality passes.",
        techniques: ["Two-order A/B", "Output parity", "Quality gates", "Hardware record"],
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
