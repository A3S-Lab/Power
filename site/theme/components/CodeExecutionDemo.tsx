"use client";

import { useState } from "react";

import "./CodeExecutionDemo.css";

type Locale = "zh" | "en";

const demoCopy = {
  zh: {
    ariaLabel: "A3S Power Rust API 执行动画",
    eyebrow: "真实 Rust API",
    replay: "重播",
    running: "执行轨迹",
    device: "设备与限制",
    deviceResult: "已解析",
    admission: "有界准入",
    admissionResult: "已授权",
    receipt: "规范化回执",
    receiptResult: "已提交",
    verifier: "客户端策略",
    verifierResult: "已接受",
    footer: "执行身份、资源边界与证据由同一条路径绑定",
  },
  en: {
    ariaLabel: "Animated A3S Power Rust API execution",
    eyebrow: "REAL RUST API",
    replay: "Replay",
    running: "Execution trace",
    device: "Device and limits",
    deviceResult: "resolved",
    admission: "Bounded admission",
    admissionResult: "granted",
    receipt: "Canonical receipt",
    receiptResult: "committed",
    verifier: "Client policy",
    verifierResult: "accepted",
    footer: "One path binds execution identity, resource bounds, and evidence",
  },
} as const;

export function CodeExecutionDemo({ locale }: { locale: Locale }) {
  const [run, setRun] = useState(0);
  const copy = demoCopy[locale];

  return (
    <section className="power-code-demo" aria-label={copy.ariaLabel}>
      <div className="power-code-demo__label">
        <span>{copy.eyebrow}</span>
        <button type="button" onClick={() => setRun((value) => value + 1)}>
          {copy.replay}
        </button>
      </div>

      <div className="power-code-frame" key={run}>
        <header className="power-code-frame__bar">
          <span aria-hidden="true">A3S</span>
          <strong>embedded.rs</strong>
          <small>{copy.running}</small>
        </header>

        <div className="power-code-frame__body">
          <pre>
            <code>
              <span className="power-code-line">
                <i aria-hidden="true">1</i>
                <b>use</b> a3s_power::inference::&#123;
              </span>
              <span className="power-code-line">
                <i aria-hidden="true">2</i>
                &nbsp;&nbsp;DevicePreference, EmbeddedRuntime,
              </span>
              <span className="power-code-line">
                <i aria-hidden="true">3</i>
                &nbsp;&nbsp;InferenceLimits,
              </span>
              <span className="power-code-line">
                <i aria-hidden="true">4</i>
                &#125;;
              </span>
              <span className="power-code-line power-code-line--runtime">
                <i aria-hidden="true">5</i>
                <b>let</b> runtime = EmbeddedRuntime::new(
              </span>
              <span className="power-code-line power-code-line--runtime">
                <i aria-hidden="true">6</i>
                &nbsp;&nbsp;DevicePreference::Auto,
              </span>
              <span className="power-code-line power-code-line--runtime">
                <i aria-hidden="true">7</i>
                &nbsp;&nbsp;InferenceLimits::default(),
              </span>
              <span className="power-code-line power-code-line--runtime">
                <i aria-hidden="true">8</i>
                )?;
              </span>
              <span className="power-code-line power-code-line--admission">
                <i aria-hidden="true">9</i>
                <b>let</b> permit = runtime
              </span>
              <span className="power-code-line power-code-line--admission">
                <i aria-hidden="true">10</i>
                &nbsp;&nbsp;.begin_wait(&amp;cancel).<b>await</b>?;
              </span>
              <span className="power-code-line power-code-line--receipt">
                <i aria-hidden="true">11</i>
                <b>let</b> receipt = runtime
              </span>
              <span className="power-code-line power-code-line--receipt">
                <i aria-hidden="true">12</i>
                &nbsp;&nbsp;.receipt(model, input, output);
              </span>
            </code>
          </pre>

          <div className="power-code-trace">
            <ol>
              <li className="power-code-event power-code-event--runtime">
                <span aria-hidden="true">✓</span>
                <div>
                  <strong>{copy.device}</strong>
                  <small>{copy.deviceResult}</small>
                </div>
              </li>
              <li className="power-code-event power-code-event--admission">
                <span aria-hidden="true">✓</span>
                <div>
                  <strong>{copy.admission}</strong>
                  <small>{copy.admissionResult}</small>
                </div>
              </li>
              <li className="power-code-event power-code-event--receipt">
                <span aria-hidden="true">✓</span>
                <div>
                  <strong>{copy.receipt}</strong>
                  <small>{copy.receiptResult}</small>
                </div>
              </li>
              <li className="power-code-event power-code-event--verifier">
                <span aria-hidden="true">✓</span>
                <div>
                  <strong>{copy.verifier}</strong>
                  <small>{copy.verifierResult}</small>
                </div>
              </li>
            </ol>
          </div>
        </div>

        <footer>
          <span aria-hidden="true">sha256</span>
          <p>{copy.footer}</p>
        </footer>
      </div>
    </section>
  );
}
