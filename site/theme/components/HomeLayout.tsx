import {
  useLang,
  useSite,
  useVersion,
  withBase,
} from "@rspress/core/runtime";

import { SpeculativeDecodingDemo } from "./SpeculativeDecodingDemo";
import { OptimizationAtlas } from "./OptimizationAtlas";
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
    mode: "Q6_K + MTP / FR · K6/S6/B8",
    quality: "9 / 9†",
    request: "46.923",
    steady: "—",
  },
  {
    mode: "Q6_K + MTP / FR · K7/S6/B11",
    quality: "same digest",
    request: "—",
    steady: "172.835",
    current: true,
  },
  {
    mode: "TBQ4 + MTP / full",
    quality: "76 / 66",
    request: "83.228",
    steady: "175.2089",
  },
];

const heroFactValues = ["3", "4", "23 / 23"];
const benchmarkMetricValues = ["172.835", "46.923", "63.42%", "23 / 23"];

const homeCopy = {
  zh: {
    titleFull: "通用 Rust 模型推理运行时。",
    title: "通用 Rust 推理",
    titleAccent: "运行时。",
    summary:
      "Power 为语言、视觉、OCR、嵌入、音频和自有计算图统一管理制品、设备、调度与证据；模型语义留在后端和模型 crate。",
    primaryAction: "查看优化体系",
    evidenceAction: "查看性能数据",
    cargoLabel: "Cargo 依赖",
    facts: [
      "CPU、CUDA、Metal 类型化设备",
      "后端与模型自有图入口",
      "当前 Q6_K 证据离线验真",
    ],
    principlesTitle: "资源限制、执行身份和验收规则都写进接口。",
    principles: [
      {
        title: "先做资源准入",
        body: "请求进入执行前检查内存、算力和队列容量；取消与微批处理使用同一组限制。",
      },
      {
        title: "记录本次执行",
        body: "回执绑定模型字节、运行策略、设备路径、输入与输出，可由调用方独立校验。",
      },
      {
        title: "由客户端验收",
        body: "客户端定义允许的度量、哈希和证据字段，服务端不能降低验收门槛。",
      },
    ],
    measuredTitle: "测量执行策略，不拿模型名称代替证据。",
    reproduce: "复现基准测试",
    benchmarkContext:
      "RTX 4090 · Qwen3.8-27B · 原始 Q6_K · clean f6326bb · K7/S6/B11 · 1 次预热 + 9 × 1,024 token",
    metrics: [
      "共享 WDDM 桌面稳态中位数 token/s",
      "通用 K6/S6/B8 请求全程 token/s",
      "相对配对自回归请求全程提升",
      "离线验真的证据文件",
    ],
    columns: ["模型制品 / 模式", "宽松 / 严格", "请求全程 t/s", "稳态 t/s"],
    current: "当前",
    note:
      "* 较早的稳态记录。† 当前 12 题配对校准有 3 题触及输出上限；172.835 是 5–8% WDDM 背景负载下的干净九次中位数，较安静主机的历史高水位为 176.6109。Qwen 行只是一个后端案例，不是引擎边界。",
    surfacesTitle: "库、服务和制品安装器共用同一套运行时约束。",
    surfaces: [
      {
        label: "库",
        title: "嵌入模型运行时",
        body: "在进程内复用 GPU、准入、放置、状态与回执；模型 crate 保留拓扑、分词和预处理。",
        meta: "embedded-inference",
      },
      {
        label: "服务",
        title: "暴露 OpenAI 兼容 API",
        body: "通过 HTTP、RA-TLS 或 vsock 提供对话、补全、嵌入、模型生命周期、指标与证明接口。",
        meta: "server + backend",
      },
      {
        label: "制品安装器",
        title: "安装精确的制品包",
        body: "把制品流式写入私有暂存区，检查大小和 SHA-256，再在跨进程锁内原子提交。",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power 管执行，模型 crate 管模型逻辑。",
    boundaryBody:
      "Power 不接管拓扑、分词、预处理或质量策略。它只为模型代码提供共享设备、资源、完整性、状态、隐私与证据机制。",
    architectureAction: "阅读架构设计",
    trace: [
      ["准入", "约束队列与内存"],
      ["执行", "选择精确设备路径"],
      ["提交", "绑定制品、策略与输入输出"],
      ["验证", "按客户端策略接受"],
    ],
    ctaTitle: "从一个模型开始，但不要把运行时写死在模型里。",
    ctaBody: "按设备、形状、调度、权重路径和验收证据组合优化；模型 crate 保留自己的拓扑与数值语义。",
    optimizationAction: "阅读优化手册",
    sourceAction: "查看源码",
  },
  en: {
    titleFull: "A general-purpose Rust inference runtime.",
    title: "General-purpose Rust",
    titleAccent: "inference runtime.",
    summary:
      "Power gives language, vision, OCR, embedding, audio, and caller-owned graphs one artifact, device, scheduling, and evidence layer while model semantics stay in backends and model crates.",
    primaryAction: "Explore the optimization system",
    evidenceAction: "View benchmark data",
    cargoLabel: "Cargo dependency",
    facts: [
      "typed CPU, CUDA, and Metal devices",
      "backend and model-owned graph paths",
      "current Q6_K evidence files verified offline",
    ],
    principlesTitle:
      "Resource limits, execution identity, and acceptance rules are explicit in the interface.",
    principles: [
      {
        title: "Check resources first",
        body: "Memory, compute, and queue capacity are checked before execution. Cancellation and microbatching use the same limits.",
      },
      {
        title: "Record each execution",
        body: "The receipt binds artifact bytes, runtime policy, device path, input, and output for independent verification.",
      },
      {
        title: "Let clients verify",
        body: "Clients define accepted measurements, hashes, and evidence fields. The server cannot lower that threshold.",
      },
    ],
    measuredTitle: "Measure the execution policy, not the model label.",
    reproduce: "Reproduce the benchmark",
    benchmarkContext:
      "RTX 4090 · Qwen3.8-27B · untouched Q6_K · clean f6326bb · K7/S6/B11 · 1 warm-up + 9 × 1,024 tokens",
    metrics: [
      "median steady token/s on a shared WDDM desktop",
      "general K6/S6/B8 request-wide token/s",
      "request-wide gain over paired autoregressive",
      "offline-verified evidence files",
    ],
    columns: [
      "Artifact / mode",
      "Lenient / strict",
      "Request-wide t/s",
      "Steady t/s",
    ],
    current: "CURRENT",
    note:
      "* Earlier steady capture. † The current 12-task paired calibration truncated 3 tasks. The clean nine-run median is 172.835 under 5–8% WDDM background load; the earlier quiet-host high-water mark is 176.6109. Qwen is one backend case study, not the engine boundary.",
    surfacesTitle:
      "Library, service, and provisioner paths use the same runtime constraints.",
    surfaces: [
      {
        label: "LIBRARY",
        title: "Embed the model runtime",
        body: "Reuse GPU access, admission, placement, state, and receipts in process. The model crate keeps topology, tokenization, and preprocessing.",
        meta: "embedded-inference",
      },
      {
        label: "SERVICE",
        title: "Expose an OpenAI-compatible API",
        body: "Serve chat, completions, embeddings, model lifecycle, metrics, and attestation over HTTP, RA-TLS, or vsock.",
        meta: "server + backend",
      },
      {
        label: "PROVISIONER",
        title: "Install an exact artifact bundle",
        body: "Stream artifacts into private staging, check size and SHA-256, then commit atomically under a cross-process lock.",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power handles execution. Model crates handle model logic.",
    boundaryBody:
      "Power does not take over topology, tokenization, preprocessing, or quality policy. It supplies shared device, resource, integrity, state, privacy, and evidence mechanisms.",
    architectureAction: "Read the architecture",
    trace: [
      ["ADMIT", "Bound queue and memory"],
      ["EXECUTE", "Select an exact device path"],
      ["COMMIT", "Bind artifacts, policy, and I/O"],
      ["VERIFY", "Accept against client policy"],
    ],
    ctaTitle:
      "Start with one model without hard-wiring the runtime to that model.",
    ctaBody:
      "Compose optimization by device, shape, scheduling, weight path, and acceptance evidence while the model crate retains topology and numerical semantics.",
    optimizationAction: "Read the optimization playbook",
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
  const optimizationHref =
    version && version !== defaultVersion
      ? withBase("/optimization")
      : route("/optimization");
  const benchmarkHref = route("/performance");
  const reproduceHref = route("/reproduction");

  return (
    <main className="power-home">
      <section className="power-hero" aria-labelledby="power-title">
        <div className="power-hero__copy">
          <h1 aria-label={copy.titleFull} id="power-title">
            {copy.title}
            <span>{copy.titleAccent}</span>
          </h1>
          <p className="power-hero__summary">{copy.summary}</p>
          <div className="power-hero__actions">
            <a className="power-action power-action--primary" href={optimizationHref}>
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
            <div><dt>{heroFactValues[0]}</dt><dd>{copy.facts[0]}</dd></div>
            <div><dt>{heroFactValues[1]}</dt><dd>{copy.facts[1]}</dd></div>
            <div><dt className="power-is-verified">{heroFactValues[2]}</dt><dd>{copy.facts[2]}</dd></div>
          </dl>
        </div>

        <div className="power-hero__specimen">
          <SpeculativeDecodingDemo locale={locale} />
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

      <OptimizationAtlas href={optimizationHref} locale={locale} />

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
          <a className="power-action power-action--primary" href={optimizationHref}>
            {copy.optimizationAction} <ArrowIcon />
          </a>
          <a className="power-action power-action--secondary" href="https://github.com/A3S-Lab/Power">
            {copy.sourceAction}
          </a>
        </div>
      </section>
    </main>
  );
}
