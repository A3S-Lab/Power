import { withBase } from "@rspress/core/runtime";

const principles = [
  {
    index: "01",
    title: "Bound every resource",
    body: "Finite memory, compute, and queue capacity become explicit admission, placement, microbatch, and cancellation contracts.",
  },
  {
    index: "02",
    title: "Bind execution identity",
    body: "Artifact bytes, runtime policy, device path, input, and output are committed into one canonical receipt.",
  },
  {
    index: "03",
    title: "Move trust to the verifier",
    body: "The client selects accepted measurements, hashes, evidence, and receipt fields; the server cannot weaken them.",
  },
];

const modes = [
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
    mode: "TBQ4 + MTP / K7-S7",
    quality: "76 / 66",
    request: "83.228",
    steady: "175.2089",
    current: true,
  },
];

const surfaces = [
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
];

function ArrowIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 16 16">
      <path d="M3 8h9M8.5 3.5 13 8l-4.5 4.5" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 20 20">
      <path d="m5 10 3 3 7-7" />
    </svg>
  );
}

export function HomeLayout() {
  const architectureHref = withBase("/architecture");
  const benchmarkHref = withBase("/performance");
  const reproduceHref = withBase("/performance#reproduce-the-boundary");
  const speculationHref = withBase("/speculative-decoding");

  return (
    <main className="power-home">
      <section className="power-hero" aria-labelledby="power-title">
        <div className="power-hero__copy">
          <p className="power-kicker">
            <span aria-hidden="true" /> A3S VERIFIABLE INFERENCE RUNTIME
          </p>
          <h1 id="power-title">
            Run the model.
            <span>Prove the boundary.</span>
          </h1>
          <p className="power-hero__summary">
            A model-neutral Rust runtime for bounded execution, canonical
            receipts, and verifier-owned trust - embedded or OpenAI-compatible.
          </p>
          <div className="power-hero__actions">
            <a className="power-action power-action--primary" href={architectureHref}>
              Start with the runtime <ArrowIcon />
            </a>
            <a className="power-action power-action--secondary" href={benchmarkHref}>
              Inspect the evidence
            </a>
          </div>
          <div className="power-install" aria-label="Cargo dependency">
            <span>$</span>
            <code>cargo add a3s-power --no-default-features -F embedded-inference</code>
          </div>
          <dl className="power-hero__facts">
            <div>
              <dt>175.2089</dt>
              <dd>steady token/s boundary</dd>
            </div>
            <div>
              <dt>83.228</dt>
              <dd>request-wide token/s</dd>
            </div>
            <div>
              <dt className="power-is-verified">900/900</dt>
              <dd>matrix requests completed</dd>
            </div>
          </dl>
        </div>

        <div className="power-hero__specimen" aria-label="A3S Power execution boundary">
          <div className="power-specimen__bar">
            <span />
            <span />
            <strong>ATTESTED EXECUTION</strong>
          </div>
          <div className="power-entry-grid">
            <div>
              <small>EMBEDDED</small>
              <strong>Rust model graph</strong>
            </div>
            <div>
              <small>HOSTED</small>
              <strong>OpenAI API</strong>
            </div>
          </div>
          <div className="power-flow-line" aria-hidden="true"><span /></div>
          <div className="power-runtime-card">
            <small>ONE RUNTIME BOUNDARY</small>
            <div>
              <strong>Admission</strong>
              <strong>Placement</strong>
              <strong>Cancellation</strong>
            </div>
            <p>weights + policy / CPU + CUDA + Metal</p>
          </div>
          <div className="power-flow-line power-flow-line--short" aria-hidden="true"><span /></div>
          <div className="power-proof-row">
            <div>
              <small>CANONICAL RECEIPT</small>
              <strong>model + policy + I/O digests</strong>
            </div>
            <i aria-hidden="true" />
            <div className="power-verifier">
              <span><CheckIcon /></span>
              <p><small>CLIENT</small><strong>verifies</strong></p>
            </div>
          </div>
          <div className="power-trace-meta">
            <span><i /> IDENTITY</span>
            <span><i /> RESOURCES</span>
            <span><i /> EVIDENCE</span>
          </div>
        </div>
      </section>

      <section className="power-section power-principles" aria-labelledby="principles-title">
        <header className="power-section__header">
          <p>FIRST PRINCIPLES</p>
          <h2 id="principles-title">Inference becomes trustworthy when its hidden assumptions become contracts.</h2>
        </header>
        <div className="power-principles__grid">
          {principles.map((principle) => (
            <article key={principle.index}>
              <span>{principle.index}</span>
              <h3>{principle.title}</h3>
              <p>{principle.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="power-section power-proof" aria-labelledby="proof-title">
        <header className="power-section__header power-section__header--split">
          <div>
            <p>MEASURED ON THE REAL API</p>
            <h2 id="proof-title">A performance boundary with receipts, not a headline without context.</h2>
          </div>
          <a href={reproduceHref}>Reproduce the benchmark <ArrowIcon /></a>
        </header>
        <div className="power-proof__table-wrap">
          <table>
            <thead>
              <tr>
                <th>Artifact / mode</th>
                <th>Lenient / strict</th>
                <th>Request-wide t/s</th>
                <th>Steady t/s</th>
              </tr>
            </thead>
            <tbody>
              {modes.map((mode) => (
                <tr className={mode.current ? "power-current-row" : undefined} key={mode.mode}>
                  <th>{mode.mode}{mode.current && <span>CURRENT</span>}</th>
                  <td>{mode.quality}</td>
                  <td>{mode.request}</td>
                  <td>{mode.steady}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="power-proof__note">
          * Earlier steady capture. The 175+ result is a steady-decode boundary
          for a Q6_K-derived mixed artifact, not an untouched 6-bit service floor.
          Quality values are fixed-task proxies, not general intelligence scores.
        </p>
      </section>

      <section className="power-section power-surfaces" aria-labelledby="surfaces-title">
        <header className="power-section__header">
          <p>ONE CORE / THREE SURFACES</p>
          <h2 id="surfaces-title">Choose where inference enters. Keep the execution contract.</h2>
        </header>
        <div className="power-surfaces__grid">
          {surfaces.map((surface) => (
            <article key={surface.label}>
              <small>{surface.label}</small>
              <h3>{surface.title}</h3>
              <p>{surface.body}</p>
              <code>{surface.meta}</code>
            </article>
          ))}
        </div>
      </section>

      <section className="power-section power-contract" aria-labelledby="contract-title">
        <div className="power-contract__copy">
          <p>RESPONSIBILITY BOUNDARY</p>
          <h2 id="contract-title">Power owns execution. Model crates own meaning.</h2>
          <p>
            Power does not duplicate topology, tokenization, preprocessing, or
            quality policy. It supplies the shared device, resource, integrity,
            state, privacy, and evidence mechanisms around reviewed model code.
          </p>
          <a href={architectureHref}>Read the architecture <ArrowIcon /></a>
        </div>
        <div className="power-contract__trace">
          <ol>
            <li><span>01</span><div><small>ADMIT</small><strong>Bound queue and memory</strong></div></li>
            <li><span>02</span><div><small>EXECUTE</small><strong>Select an exact device path</strong></div></li>
            <li><span>03</span><div><small>COMMIT</small><strong>Bind artifacts, policy, and I/O</strong></div></li>
            <li className="is-verified"><span><CheckIcon /></span><div><small>VERIFY</small><strong>Accept against client policy</strong></div></li>
          </ol>
        </div>
      </section>

      <section className="power-cta">
        <div>
          <p>EXACT SPECULATION / EXPLICIT EVIDENCE</p>
          <h2>Optimize the path without moving the trust boundary.</h2>
          <span>Native MTP, rollback-complete verification, reproducible matrices, and honest limits.</span>
        </div>
        <div>
          <a className="power-action power-action--primary" href={speculationHref}>
            Read speculative decoding <ArrowIcon />
          </a>
          <a className="power-action power-action--secondary" href="https://github.com/A3S-Lab/Power">
            View source
          </a>
        </div>
      </section>
    </main>
  );
}
