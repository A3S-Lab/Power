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
    quality: { zh: "67 / 60", en: "67 / 60" },
    request: "30.883",
    steady: "35.5793",
  },
  {
    mode: "Q6_K + MTP / FR · K6/S6/B8",
    quality: { zh: "9 / 9", en: "9 / 9" },
    request: "46.923",
    steady: "—",
  },
  {
    mode: "Q6_K + MTP / FR · K7/S6/B11",
    quality: { zh: "输出哈希一致", en: "Matching digest" },
    request: "—",
    steady: "172.835",
    current: true,
  },
  {
    mode: "TBQ4 + MTP / full",
    quality: { zh: "76 / 66", en: "76 / 66" },
    request: "83.228",
    steady: "175.2089",
  },
];

const benchmarkMetricValues = ["172.835", "46.923", "63.42%", "23 / 23"];

const homeCopy = {
  zh: {
    titleFull: "模型由你定义。执行交给 Power。",
    title: "模型由你定义。",
    titleAccent: "执行交给 Power。",
    summary:
      "在 Rust 进程或服务端运行语言、视觉、OCR、嵌入和音频模型。Power 统一处理设备、队列、权重和执行记录，不改写模型逻辑。",
    primaryAction: "开始接入",
    evidenceAction: "查看实测性能",
    cargoLabel: "Cargo 依赖",
    facts: [
      { value: "3", label: "CPU、CUDA、Metal 设备" },
      { value: "4", label: "种后端与嵌入路径" },
      { value: "23 / 23", label: "性能证据通过校验" },
    ],
    principlesTitle: "一套运行时，守住三件事。",
    principles: [
      {
        title: "资源有边界",
        body: "先检查内存、设备和队列，再执行。超出限制就明确拒绝，不靠隐式回退兜底。",
      },
      {
        title: "结果可追溯",
        body: "模型、配置、设备、输入和输出写入同一份执行回执，之后可以独立核对。",
      },
      {
        title: "验收权在调用方",
        body: "客户端按自己的哈希和策略验收，服务端不能悄悄放宽条件。",
      },
    ],
    measuredTitle: "一台 RTX 4090，一组可复现结果。",
    reproduce: "复现这组数据",
    benchmarkContext:
      "Qwen3.8-27B · 原始 Q6_K · revision f6326bb · K7/S6/B11 · 1 次预热 + 9 × 1,024 token",
    metrics: [
      "Q6_K 稳态解码中位数 token/s",
      "通用配置的请求全程 token/s",
      "相对自回归的请求全程提升",
      "证据文件通过哈希校验",
    ],
    columns: ["模式", "质量证据", "请求全程 t/s", "稳态 t/s"],
    current: "当前",
    note:
      "172.835 token/s 是共享 WDDM 桌面上的 9 次稳态中位数，不是服务保底值。各行质量数据使用对应的归档校准集，完整口径见性能页。",
    surfacesTitle: "按你的方式接入。",
    surfaces: [
      {
        label: "Rust 库",
        title: "直接嵌入进程",
        body: "不监听端口，不启动子进程。模型代码直接使用设备、队列、状态和执行回执。",
        meta: "embedded-inference",
      },
      {
        label: "推理服务",
        title: "接入现有客户端",
        body: "提供 OpenAI 兼容 API，支持对话、补全、嵌入和模型管理；需要时可启用 RA-TLS 或 vsock。",
        meta: "server + backend",
      },
      {
        label: "制品安装",
        title: "安全落盘模型",
        body: "流式下载模型，校验大小与 SHA-256，最后在跨进程锁内一次原子提交。",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power 管执行，不替模型做决定。",
    boundaryBody:
      "模型 crate 决定拓扑、分词、预处理和质量策略；Power 提供设备、调度、完整性、隐私和验证。接入新模型不需要修改运行时核心。",
    architectureAction: "查看架构边界",
    trace: [
      ["准入", "内存和队列够不够？"],
      ["执行", "使用哪条设备路径？"],
      ["提交", "这次输出绑定了什么？"],
      ["验收", "调用方是否接受？"],
    ],
    ctaTitle: "先接入一个模型。",
    ctaBody: "用嵌入式运行时开始，或直接启动 OpenAI 兼容服务。性能优化和执行验证都可以沿用同一套接口。",
    gettingStartedAction: "快速开始",
    sourceAction: "查看源码",
  },
  en: {
    titleFull: "You define the model. Power runs it.",
    title: "You define the model.",
    titleAccent: "Power runs it.",
    summary:
      "Run language, vision, OCR, embedding, and audio models inside a Rust process or behind an API. Power handles devices, queues, weights, and execution records without rewriting model logic.",
    primaryAction: "Get started",
    evidenceAction: "See measured performance",
    cargoLabel: "Cargo dependency",
    facts: [
      { value: "3", label: "CPU, CUDA, and Metal devices" },
      { value: "4", label: "backend and embedded paths" },
      { value: "23 / 23", label: "performance artifacts verified" },
    ],
    principlesTitle: "One runtime. Three hard guarantees.",
    principles: [
      {
        title: "Resources stay bounded",
        body: "Power checks memory, devices, and queues before execution. It rejects work that cannot fit instead of hiding a fallback.",
      },
      {
        title: "Results stay traceable",
        body: "Model, configuration, device, input, and output are bound into one execution receipt that can be checked later.",
      },
      {
        title: "Callers set the bar",
        body: "Clients verify against their own hashes and policy. The server cannot quietly relax the acceptance rules.",
      },
    ],
    measuredTitle: "One RTX 4090. One reproducible result.",
    reproduce: "Reproduce these numbers",
    benchmarkContext:
      "Qwen3.8-27B · untouched Q6_K · revision f6326bb · K7/S6/B11 · 1 warm-up + 9 × 1,024 tokens",
    metrics: [
      "median Q6_K steady decode token/s",
      "request-wide token/s on the general profile",
      "request-wide gain over autoregressive",
      "evidence artifacts passed hash checks",
    ],
    columns: ["Mode", "Quality evidence", "Request-wide t/s", "Steady t/s"],
    current: "CURRENT",
    note:
      "172.835 token/s is the median of nine steady runs on a shared WDDM desktop, not a service floor. Each row uses its corresponding archived quality set; see Performance for the full methodology.",
    surfacesTitle: "Use Power your way.",
    surfaces: [
      {
        label: "Rust library",
        title: "Embed it in your process",
        body: "No listener and no child process. Model code uses devices, queues, state, and execution receipts directly.",
        meta: "embedded-inference",
      },
      {
        label: "Inference service",
        title: "Keep your existing clients",
        body: "Use an OpenAI-compatible API for chat, completions, embeddings, and model management, with RA-TLS or vsock when needed.",
        meta: "server + backend",
      },
      {
        label: "Artifact install",
        title: "Put models on disk safely",
        body: "Stream the model, verify size and SHA-256, then commit it once under a cross-process lock.",
        meta: "artifact-provisioning",
      },
    ],
    boundaryTitle: "Power runs the model. It does not redefine it.",
    boundaryBody:
      "Model crates own topology, tokenization, preprocessing, and quality policy. Power supplies devices, scheduling, integrity, privacy, and verification, so a new model does not require a new runtime core.",
    architectureAction: "See the architecture boundary",
    trace: [
      ["ADMIT", "Do memory and queue capacity fit?"],
      ["EXECUTE", "Which device path runs this request?"],
      ["COMMIT", "What does this output bind?"],
      ["ACCEPT", "Does the caller accept it?"],
    ],
    ctaTitle: "Start with one model.",
    ctaBody:
      "Embed the runtime or launch the OpenAI-compatible service. Performance tuning and execution verification use the same interfaces when you need them.",
    gettingStartedAction: "Get started",
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
  const gettingStartedHref = route("/getting-started");
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
            <a className="power-action power-action--primary" href={gettingStartedHref}>
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
            {copy.facts.map((fact, index) => (
              <div key={fact.value}>
                <dt className={index === 2 ? "power-is-verified" : undefined}>
                  {fact.value}
                </dt>
                <dd>{fact.label}</dd>
              </div>
            ))}
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
                  <td>{mode.quality[locale]}</td><td>{mode.request}</td><td>{mode.steady}</td>
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
          <a className="power-action power-action--primary" href={gettingStartedHref}>
            {copy.gettingStartedAction} <ArrowIcon />
          </a>
          <a className="power-action power-action--secondary" href="https://github.com/A3S-Lab/Power">
            {copy.sourceAction}
          </a>
        </div>
      </section>
    </main>
  );
}
