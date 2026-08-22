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
      "MTP K7/S7 推测解码示意。草稿提出七个 token，目标模型接受前五个并纠正第六个，Power 撤销未通过的后缀后提交六个 token。",
    title: "一次前向，验证多个 token",
    profile: "MTP · K7 / S7",
    example: "一轮解码",
    replay: "重新播放",
    phaseLabel: "当前步骤",
    phases: {
      checkpoint: {
        short: "快照",
        title: "保存可回滚状态",
        detail: "为 7 个候选位置保留目标模型、草稿模型、采样器和解码器状态。",
      },
      draft: {
        short: "草稿",
        title: "先提出 7 个候选",
        detail: "候选只用于校验，不会直接写入输出。",
      },
      verify: {
        short: "校验",
        title: "目标模型逐位校验",
        detail: "前 5 个位置匹配；第 6 个不匹配时停止。",
      },
      rollback: {
        short: "回滚",
        title: "丢弃未通过的后缀",
        detail: "恢复到 S5，撤销第 6、7 个位置的状态。",
      },
      commit: {
        short: "提交",
        title: "一次提交 6 个 token",
        detail: "写入 5 个已接受 token 和 1 个由目标模型纠正的 token。",
      },
    },
    lanes: {
      proposal: "草稿候选",
      target: "目标校验",
      output: "最终输出",
    },
    laneMeta: {
      proposal: "7 个候选",
      target: "遇到不匹配即停止",
      output: "5 个通过 + 1 个纠正",
    },
    tokenStates: {
      proposal: "候选",
      accepted: "通过",
      correction: "已纠正",
      skipped: "跳过",
      committed: "已提交",
    },
    snapshots: "状态快照",
    snapshotsResident: "保留 S1-S7",
    rollbackTo: "回滚到 S5",
    result: "本轮：5 个候选通过，1 个 token 由目标模型纠正",
    disclaimer: "算法示意，非性能数据",
  },
  en: {
    ariaLabel:
      "MTP K7/S7 speculative decoding. The draft proposes seven tokens, the target accepts five and corrects the sixth, then Power discards the rejected suffix and commits six tokens.",
    title: "Verify more tokens per forward pass",
    profile: "MTP · K7 / S7",
    example: "One decode round",
    replay: "Replay round",
    phaseLabel: "Current step",
    phases: {
      checkpoint: {
        short: "SNAPSHOT",
        title: "Save rollback state",
        detail: "Keep target, draft, sampler, and decoder state for seven candidate positions.",
      },
      draft: {
        short: "DRAFT",
        title: "Propose 7 candidates",
        detail: "Candidates are checked first. They never enter the output directly.",
      },
      verify: {
        short: "CHECK",
        title: "Let the target model check each position",
        detail: "The first five match. Verification stops when position six does not.",
      },
      rollback: {
        short: "ROLLBACK",
        title: "Discard the rejected suffix",
        detail: "Restore S5 and remove state for positions six and seven.",
      },
      commit: {
        short: "COMMIT",
        title: "Commit 6 tokens at once",
        detail: "Write five accepted tokens and one token corrected by the target model.",
      },
    },
    lanes: {
      proposal: "Draft proposals",
      target: "Target verification",
      output: "Final output",
    },
    laneMeta: {
      proposal: "7 proposals",
      target: "stop on the first mismatch",
      output: "5 matches + 1 correction",
    },
    tokenStates: {
      proposal: "candidate",
      accepted: "match",
      correction: "corrected",
      skipped: "skipped",
      committed: "committed",
    },
    snapshots: "State snapshots",
    snapshotsResident: "keep S1-S7",
    rollbackTo: "rollback to S5",
    result: "This round: 5 candidates matched and the target corrected 1 token",
    disclaimer: "Algorithm illustration, not performance data",
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
