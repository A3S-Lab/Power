"use client";

import { useEffect, useRef, useState } from "react";

import "./SpeculativeDecodingDemo.css";

type Locale = "zh" | "en";
type Phase = "checkpoint" | "draft" | "verify" | "rollback" | "commit";

const phases: Phase[] = ["checkpoint", "draft", "verify", "rollback", "commit"];
const phaseDurations = [900, 1350, 1450, 1050, 1900];
const proposalTokenIds = ["0841", "0298", "1973", "0456", "0062", "1401", "0077"];
const acceptedProposalCount = 5;
const correctionTokenId = "0981";

const demoCopy = {
  zh: {
    ariaLabel:
      "MTP K7/S7 推测解码示例。Draft 提出七个 token，目标模型接受前五个，在第六个位置给出纠正 token，运行时回滚被拒后缀并提交六个输出 token。",
    title: "MTP 推测解码",
    profile: "全词表 · K7 / S7",
    example: "示例轮次",
    replay: "重新播放",
    phaseLabel: "当前步骤",
    phases: {
      checkpoint: {
        short: "快照",
        title: "保留循环状态",
        detail: "为七个 proposal 位置保存目标、draft、采样器和解码器状态。",
      },
      draft: {
        short: "提议",
        title: "Draft 一次提出 7 个 token",
        detail: "Proposal 只提供候选，不直接写入输出。",
      },
      verify: {
        short: "验证",
        title: "目标模型验证 proposal block",
        detail: "前五个位置匹配；采样在第一个不匹配位置停止。",
      },
      rollback: {
        short: "回滚",
        title: "恢复到最后一个已接受位置",
        detail: "丢弃第六、七个 proposal 的状态，并恢复 S5 快照。",
      },
      commit: {
        short: "提交",
        title: "提交接受前缀和 correction token",
        detail: "本示例用一次目标模型前向提交 6 个输出 token。",
      },
    },
    lanes: {
      proposal: "Draft 提议",
      target: "Target 验证",
      output: "提交输出",
    },
    laneMeta: {
      proposal: "7 proposals",
      target: "首个不匹配即停止",
      output: "5 accepted + 1 correction",
    },
    tokenStates: {
      proposal: "proposal",
      accepted: "匹配",
      correction: "纠正",
      skipped: "未采样",
      committed: "已提交",
    },
    snapshots: "循环状态快照",
    snapshotsResident: "S1-S7 常驻",
    rollbackTo: "回滚到 S5",
    result: "示例结果：接受 5/7 个 proposal，提交 1 个 correction token",
    disclaimer: "示意数据，不是基准测试样本",
  },
  en: {
    ariaLabel:
      "MTP K7/S7 speculative decoding example. The draft proposes seven tokens, the target accepts five and supplies a correction at position six, then the runtime rolls back the rejected suffix and commits six output tokens.",
    title: "MTP speculative decoding",
    profile: "FULL VOCABULARY · K7 / S7",
    example: "Illustrative round",
    replay: "Replay round",
    phaseLabel: "Current step",
    phases: {
      checkpoint: {
        short: "SNAPSHOT",
        title: "Retain recurrent state",
        detail: "Save target, draft, sampler, and decoder state for seven proposal positions.",
      },
      draft: {
        short: "PROPOSE",
        title: "Draft proposes 7 tokens",
        detail: "Proposals are candidates. They do not enter the output directly.",
      },
      verify: {
        short: "VERIFY",
        title: "Target verifies the proposal block",
        detail: "The first five positions match. Sampling stops at the first mismatch.",
      },
      rollback: {
        short: "ROLLBACK",
        title: "Restore the last accepted position",
        detail: "Discard proposal state for positions six and seven, then restore snapshot S5.",
      },
      commit: {
        short: "COMMIT",
        title: "Commit the accepted prefix and correction",
        detail: "This example commits 6 output tokens from one target-model pass.",
      },
    },
    lanes: {
      proposal: "Draft proposals",
      target: "Target check",
      output: "Committed output",
    },
    laneMeta: {
      proposal: "7 proposals",
      target: "stop at first mismatch",
      output: "5 accepted + 1 correction",
    },
    tokenStates: {
      proposal: "proposal",
      accepted: "match",
      correction: "correction",
      skipped: "not sampled",
      committed: "committed",
    },
    snapshots: "Recurrent state snapshots",
    snapshotsResident: "S1-S7 resident",
    rollbackTo: "rollback to S5",
    result: "Example result: 5/7 proposals accepted, then 1 correction token committed",
    disclaimer: "Illustrative data, not a benchmark sample",
  },
} as const;

function phaseClass(phase: Phase, currentPhase: Phase) {
  const phaseIndex = phases.indexOf(phase);
  const currentIndex = phases.indexOf(currentPhase);
  if (phaseIndex < currentIndex) return "is-complete";
  if (phaseIndex === currentIndex) return "is-current";
  return undefined;
}

export function SpeculativeDecodingDemo({ locale }: { locale: Locale }) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [run, setRun] = useState(0);
  const [inViewport, setInViewport] = useState(true);
  const [documentVisible, setDocumentVisible] = useState(true);
  const [reducedMotion, setReducedMotion] = useState(false);
  const rootRef = useRef<HTMLElement>(null);
  const copy = demoCopy[locale];
  const phase = phases[phaseIndex];

  useEffect(() => {
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const updateMotionPreference = () => setReducedMotion(motionQuery.matches);
    const updateDocumentVisibility = () =>
      setDocumentVisible(document.visibilityState === "visible");
    const observer = new IntersectionObserver(
      ([entry]) => setInViewport(entry?.isIntersecting ?? true),
      { threshold: 0.2 },
    );

    updateMotionPreference();
    updateDocumentVisibility();
    motionQuery.addEventListener("change", updateMotionPreference);
    document.addEventListener("visibilitychange", updateDocumentVisibility);
    if (rootRef.current) observer.observe(rootRef.current);

    return () => {
      motionQuery.removeEventListener("change", updateMotionPreference);
      document.removeEventListener("visibilitychange", updateDocumentVisibility);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    if (reducedMotion) {
      setPhaseIndex(phases.length - 1);
      return undefined;
    }
    if (!inViewport || !documentVisible) return undefined;

    const timer = window.setTimeout(() => {
      setPhaseIndex((current) => (current + 1) % phases.length);
    }, phaseDurations[phaseIndex]);

    return () => window.clearTimeout(timer);
  }, [documentVisible, inViewport, phaseIndex, reducedMotion, run]);

  const replay = () => {
    setRun((current) => current + 1);
    setPhaseIndex(reducedMotion ? phases.length - 1 : 0);
  };

  return (
    <section
      className={`power-mtp-demo is-phase-${phase}`}
      data-motion={reducedMotion ? "reduced" : "full"}
      data-phase={phase}
      data-run={run}
      ref={rootRef}
    >
      <div className="power-mtp-demo__label">
        <div>
          <strong>{copy.title}</strong>
          <span>{copy.profile}</span>
        </div>
        <button type="button" onClick={replay}>
          {copy.replay}
        </button>
      </div>

      <div className="power-mtp-frame" role="img" aria-label={copy.ariaLabel}>
        <header className="power-mtp-frame__bar">
          <span>{copy.example}</span>
          <code>spec_mode = "mtp"</code>
        </header>

        <ol className="power-mtp-phases" aria-label={copy.phaseLabel}>
          {phases.map((item) => (
            <li className={phaseClass(item, phase)} key={item}>
              <span aria-hidden="true" />
              <small>{copy.phases[item].short}</small>
            </li>
          ))}
        </ol>

        <div className="power-mtp-frame__body" key={`${run}-${phase}`}>
          <div className="power-mtp-state">
            <small>{copy.phases[phase].short}</small>
            <strong>{copy.phases[phase].title}</strong>
            <p>{copy.phases[phase].detail}</p>
          </div>

          <div className="power-mtp-lanes">
            <div className="power-mtp-lane power-mtp-lane--proposal">
              <div className="power-mtp-lane__label">
                <strong>{copy.lanes.proposal}</strong>
                <small>{copy.laneMeta.proposal}</small>
              </div>
              <ol>
                {proposalTokenIds.map((tokenId, index) => (
                  <li
                    className={
                      index >= acceptedProposalCount
                        ? "power-mtp-token is-rejected"
                        : "power-mtp-token is-accepted"
                    }
                    key={tokenId}
                  >
                    <span>#{tokenId}</span>
                    <small>P{index + 1}</small>
                    <em>{copy.tokenStates.proposal}</em>
                  </li>
                ))}
              </ol>
            </div>

            <div className="power-mtp-lane power-mtp-lane--target">
              <div className="power-mtp-lane__label">
                <strong>{copy.lanes.target}</strong>
                <small>{copy.laneMeta.target}</small>
              </div>
              <ol>
                {proposalTokenIds.map((tokenId, index) => {
                  const accepted = index < acceptedProposalCount;
                  const correction = index === acceptedProposalCount;
                  return (
                    <li
                      className={`power-mtp-token ${
                        accepted
                          ? "is-accepted"
                          : correction
                            ? "is-correction"
                            : "is-skipped"
                      }`}
                      key={tokenId}
                    >
                      <span>#{correction ? correctionTokenId : tokenId}</span>
                      <small>T{index + 1}</small>
                      <em>
                        {accepted
                          ? copy.tokenStates.accepted
                          : correction
                            ? copy.tokenStates.correction
                            : copy.tokenStates.skipped}
                      </em>
                    </li>
                  );
                })}
              </ol>
            </div>

            <div className="power-mtp-lane power-mtp-lane--output">
              <div className="power-mtp-lane__label">
                <strong>{copy.lanes.output}</strong>
                <small>{copy.laneMeta.output}</small>
              </div>
              <ol>
                {[...proposalTokenIds.slice(0, acceptedProposalCount), correctionTokenId].map(
                  (tokenId, index) => (
                    <li
                      className={`power-mtp-token ${
                        index === acceptedProposalCount ? "is-correction" : "is-accepted"
                      }`}
                      key={`${tokenId}-${index}`}
                    >
                      <span>#{tokenId}</span>
                      <small>O{index + 1}</small>
                      <em>{copy.tokenStates.committed}</em>
                    </li>
                  ),
                )}
              </ol>
            </div>
          </div>

          <div className="power-mtp-snapshots">
            <div>
              <strong>{copy.snapshots}</strong>
              <small>
                {phaseIndex >= phases.indexOf("rollback")
                  ? copy.rollbackTo
                  : copy.snapshotsResident}
              </small>
            </div>
            <ol>
              {proposalTokenIds.map((_, index) => (
                <li className={index >= acceptedProposalCount ? "is-discarded" : undefined} key={index}>
                  S{index + 1}
                </li>
              ))}
            </ol>
          </div>
        </div>

        <footer>
          <strong>{copy.result}</strong>
          <small>{copy.disclaimer}</small>
        </footer>
      </div>
    </section>
  );
}
