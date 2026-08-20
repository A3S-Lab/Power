import {
  useLang,
  useSite,
  useVersion,
  withBase,
} from "@rspress/core/runtime";

import { CodeExecutionDemo } from "./CodeExecutionDemo";
import "./PerformanceProof.css";

type Locale = "zh" | "en";

const modeData = [
  {
    mode: "Q6_K / AR",
    quality: "67 / 60",
    request: "30.883",
    steady: "35.5793*",
  },
  {
    mode: "TBQ4 / AR",
    quality: "70 / 64",
    request: "38.724",
    steady: "-",
  },
  {
    mode: "TBQ4 + MTP / K7/S7",
    quality: "76 / 66",
    request: "83.228",
    steady: "175.2089",
    current: true,
  },
];

const benchmarkMetricValues = ["175.2089", "83.228", "76 / 66", "51.33%"];

const homeCopy = {
  zh: {
    kicker: "A3S 可验证推理运行时",
    title: "运行模型。",
    titleAccent: "证明边界。",
    summary:
      "面向有界执行、规范化回执与验证方自主信任的模型中立 Rust 运行时，可嵌入，也兼容 OpenAI API。",
    primaryAction: "从运行时开始",
    evidenceAction: "查看实测证据",
    cargoLabel: "Cargo 依赖",
    facts: [
      "K7/S7 稳态解码 token/s",
      "K7/S7 请求全程 token/s",
      "100 题 × 3 轮完成请求",
    ],
    principlesTitle: "当推理中的隐含假设成为显式契约，系统才值得信任。",
    principles: [
      {
        title: "约束每一项资源",
        body: "有限的内存、算力与队列容量，被转化为明确的准入、放置、微批处理与取消契约。",
      },
      {
        title: "绑定执行身份",
        body: "模型字节、运行策略、设备路径、输入与输出，共同写入一份规范化回执。",
      },
      {
        title: "把信任交给验证方",
        body: "由客户端选择接受哪些度量、哈希、证据与回执字段，服务端无法降低标准。",
      },
    ],
    measuredTitle: "性能边界必须附带可复验回执，而不是脱离语境的数字。",
    reproduce: "复现基准测试",
    benchmarkContext:
      "RTX 4090 · Qwen3.8-27B · Q6_K 衍生 TBQ4 + 全词表 MTP K7/S7 · 1 次预热 + 9 × 1,024 token",
    metrics: [
      "稳态解码中位数 token/s",
      "请求全程平均 token/s",
      "宽松 / 严格质量分",
      "proposal 加权接受率",
    ],
    columns: ["模型制品 / 模式", "宽松 / 严格", "请求全程 t/s", "稳态 t/s"],
    current: "当前",
    note:
      "* 较早的稳态记录。175+ 是 Q6_K 衍生混合制品的稳态解码边界，不是原始 6-bit 模型的服务下限；质量值是固定任务代理指标，而非通用智力分数。",
    surfacesTitle: "选择推理从哪里进入，但始终保留同一执行契约。",
    surfaces: [
      {
        label: "库",
        title: "嵌入已审查的模型图",
        body: "使用无监听器运行时共享设备、准入、放置、状态与证据；模型 crate 继续拥有语义。",
        meta: "embedded-inference",
      },
      {
        label: "服务",
        title: "暴露 OpenAI 兼容 API",
        body: "通过显式 HTTP、RA-TLS 或 vsock 传输承载对话、补全、嵌入、模型生命周期、指标与证明。",
        meta: "server + backend",
      },
      {
        label: "制品安装器",
        title: "安装精确的制品包",
        body: "流式写入私有暂存区，校验字节上限与 SHA-256 身份，再在跨进程锁下原子提交。",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power 负责执行，模型 crate 负责语义。",
    boundaryBody:
      "Power 不复制拓扑、分词、预处理或质量策略；它围绕已审查的模型代码，提供共享设备、资源、完整性、状态、隐私与证据机制。",
    architectureAction: "阅读架构设计",
    trace: [
      ["准入", "约束队列与内存"],
      ["执行", "选择精确设备路径"],
      ["提交", "绑定制品、策略与输入输出"],
      ["验证", "按客户端策略接受"],
    ],
    ctaTitle: "优化执行路径，而不移动信任边界。",
    ctaBody: "原生 MTP、完整回滚验证、可复现实验矩阵与诚实的适用边界。",
    speculationAction: "阅读推测解码",
    sourceAction: "查看源码",
  },
  en: {
    kicker: "A3S VERIFIABLE INFERENCE RUNTIME",
    title: "Run the model.",
    titleAccent: "Prove the boundary.",
    summary:
      "A model-neutral Rust runtime for bounded execution, canonical receipts, and verifier-owned trust, embedded or OpenAI-compatible.",
    primaryAction: "Start with the runtime",
    evidenceAction: "Inspect the evidence",
    cargoLabel: "Cargo dependency",
    facts: [
      "K7/S7 steady-decode token/s",
      "K7/S7 request-wide token/s",
      "100 tasks × 3 runs completed",
    ],
    principlesTitle:
      "Inference becomes trustworthy when its hidden assumptions become contracts.",
    principles: [
      {
        title: "Bound every resource",
        body: "Finite memory, compute, and queue capacity become explicit admission, placement, microbatch, and cancellation contracts.",
      },
      {
        title: "Bind execution identity",
        body: "Artifact bytes, runtime policy, device path, input, and output are committed into one canonical receipt.",
      },
      {
        title: "Move trust to the verifier",
        body: "The client selects accepted measurements, hashes, evidence, and receipt fields; the server cannot weaken them.",
      },
    ],
    measuredTitle:
      "A performance boundary with receipts, not a headline without context.",
    reproduce: "Reproduce the benchmark",
    benchmarkContext:
      "RTX 4090 · Qwen3.8-27B · Q6_K-derived TBQ4 + full-vocabulary MTP K7/S7 · 1 warm-up + 9 × 1,024 tokens",
    metrics: [
      "median steady-decode token/s",
      "mean request-wide token/s",
      "lenient / strict quality",
      "weighted proposal acceptance",
    ],
    columns: [
      "Artifact / mode",
      "Lenient / strict",
      "Request-wide t/s",
      "Steady t/s",
    ],
    current: "CURRENT",
    note:
      "* Earlier steady capture. The 175+ result is a steady-decode boundary for a Q6_K-derived mixed artifact, not an untouched 6-bit service floor. Quality values are fixed-task proxies, not general intelligence scores.",
    surfacesTitle: "Choose where inference enters. Keep the execution contract.",
    surfaces: [
      {
        label: "LIBRARY",
        title: "Embed a reviewed model graph",
        body: "Use the listener-free runtime for shared devices, admission, placement, state, and evidence while the model crate retains semantics.",
        meta: "embedded-inference",
      },
      {
        label: "SERVICE",
        title: "Expose an OpenAI-compatible API",
        body: "Host chat, completions, embeddings, model lifecycle, metrics, and attestation over explicit HTTP, RA-TLS, or vsock transport.",
        meta: "server + backend",
      },
      {
        label: "PROVISIONER",
        title: "Install an exact artifact bundle",
        body: "Stream into private staging, enforce byte limits and SHA-256 identity, then commit atomically under a cross-process lock.",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power owns execution. Model crates own meaning.",
    boundaryBody:
      "Power does not duplicate topology, tokenization, preprocessing, or quality policy. It supplies the shared device, resource, integrity, state, privacy, and evidence mechanisms around reviewed model code.",
    architectureAction: "Read the architecture",
    trace: [
      ["ADMIT", "Bound queue and memory"],
      ["EXECUTE", "Select an exact device path"],
      ["COMMIT", "Bind artifacts, policy, and I/O"],
      ["VERIFY", "Accept against client policy"],
    ],
    ctaTitle: "Optimize the path without moving the trust boundary.",
    ctaBody:
      "Native MTP, rollback-complete verification, reproducible matrices, and honest limits.",
    speculationAction: "Read speculative decoding",
    sourceAction: "View source",
  },
} as const;

function ArrowIcon() {
  return <span className="power-arrow" aria-hidden="true">→</span>;
}

function CheckIcon() {
  return <span className="power-check" aria-hidden="true">✓</span>;
}

export function HomeLayout() {
  const rawLang = useLang();
  const locale: Locale = rawLang === "zh" ? "zh" : "en";
  const copy = homeCopy[locale];
  const version = useVersion();
  const { site } = useSite();
  const defaultVersion = site.multiVersion.default;
  const routePrefix = [
    version && version !== defaultVersion ? version : "",
    locale !== site.lang ? locale : "",
  ]
    .filter(Boolean)
    .join("/");
  const route = (pathname: string) => {
    const normalizedPath = pathname.replace(/^\/+/, "");
    const parts = [routePrefix, normalizedPath].filter(Boolean).join("/");
    return withBase(`/${parts}`);
  };

  const architectureHref = route("/architecture");
  const benchmarkHref = route("/performance");
  const reproduceHref = route("/reproduction");
  const speculationHref = route("/speculative-decoding");

  return (
    <main className="power-home">
      <section className="power-hero" aria-labelledby="power-title">
        <div className="power-hero__copy">
          <p className="power-kicker">
            <img
              alt=""
              aria-hidden="true"
              className="power-brand-mark"
              height="30"
              src={withBase("/a3s-os-logo.png")}
              width="30"
            />
            {copy.kicker}
          </p>
          <h1 id="power-title">
            {copy.title}
            <span>{copy.titleAccent}</span>
          </h1>
          <p className="power-hero__summary">{copy.summary}</p>
          <div className="power-hero__actions">
            <a className="power-action power-action--primary" href={architectureHref}>
              {copy.primaryAction} <ArrowIcon />
            </a>
            <a className="power-action power-action--secondary" href={benchmarkHref}>
              {copy.evidenceAction}
            </a>
          </div>
          <div className="power-install" aria-label={copy.cargoLabel}>
            <span>$</span>
            <code>cargo add a3s-power --no-default-features -F embedded-inference</code>
          </div>
          <dl className="power-hero__facts">
            <div><dt>175.2089</dt><dd>{copy.facts[0]}</dd></div>
            <div><dt>83.228</dt><dd>{copy.facts[1]}</dd></div>
            <div><dt className="power-is-verified">900/900</dt><dd>{copy.facts[2]}</dd></div>
          </dl>
        </div>

        <div className="power-hero__specimen">
          <CodeExecutionDemo locale={locale} />
        </div>
      </section>

      <section className="power-section power-principles" aria-labelledby="principles-title">
        <header className="power-section__header">
          <h2 id="principles-title">{copy.principlesTitle}</h2>
        </header>
        <div className="power-principles__grid">
          {copy.principles.map((principle) => (
            <article key={principle.title}>
              <h3>{principle.title}</h3><p>{principle.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="power-section power-proof" aria-labelledby="proof-title">
        <header className="power-section__header power-section__header--split">
          <div>
            <h2 id="proof-title">{copy.measuredTitle}</h2>
            <p className="power-proof__context">{copy.benchmarkContext}</p>
          </div>
          <a href={reproduceHref}>{copy.reproduce} <ArrowIcon /></a>
        </header>
        <dl className="power-proof__metrics">
          {benchmarkMetricValues.map((value, index) => (
            <div key={value}>
              <dt>{value}</dt>
              <dd>{copy.metrics[index]}</dd>
            </div>
          ))}
        </dl>
        <div className="power-proof__table-wrap">
          <table>
            <thead><tr>{copy.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
            <tbody>
              {modeData.map((mode) => (
                <tr className={mode.current ? "power-current-row" : undefined} key={mode.mode}>
                  <th>{mode.mode}{mode.current && <span>{copy.current}</span>}</th>
                  <td>{mode.quality}</td><td>{mode.request}</td><td>{mode.steady}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="power-proof__note">{copy.note}</p>
      </section>

      <section className="power-section power-surfaces" aria-labelledby="surfaces-title">
        <header className="power-section__header">
          <h2 id="surfaces-title">{copy.surfacesTitle}</h2>
        </header>
        <div className="power-surfaces__grid">
          {copy.surfaces.map((surface) => (
            <article key={surface.label}>
              <small>{surface.label}</small><h3>{surface.title}</h3><p>{surface.body}</p><code>{surface.meta}</code>
            </article>
          ))}
        </div>
      </section>

      <section className="power-section power-contract" aria-labelledby="contract-title">
        <div className="power-contract__copy">
          <h2 id="contract-title">{copy.boundaryTitle}</h2><p>{copy.boundaryBody}</p>
          <a href={architectureHref}>{copy.architectureAction} <ArrowIcon /></a>
        </div>
        <div className="power-contract__trace">
          <ol>
            {copy.trace.map((item, index) => (
              <li className={index === 3 ? "is-verified" : undefined} key={item[0]}>
                <span>{index === 3 ? <CheckIcon /> : "→"}</span>
                <div><small>{item[0]}</small><strong>{item[1]}</strong></div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="power-cta">
        <div><h2>{copy.ctaTitle}</h2><span>{copy.ctaBody}</span></div>
        <div>
          <a className="power-action power-action--primary" href={speculationHref}>
            {copy.speculationAction} <ArrowIcon />
          </a>
          <a className="power-action power-action--secondary" href="https://github.com/A3S-Lab/Power">
            {copy.sourceAction}
          </a>
        </div>
      </section>
    </main>
  );
}
