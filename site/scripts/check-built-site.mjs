import { createHash } from "node:crypto";
import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
} from "node:fs";
import { fileURLToPath } from "node:url";
import * as path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const buildRoot = path.join(siteRoot, "doc_build");

const htmlFiles = [];
const visit = (directory) => {
  for (const entry of readdirSync(directory)) {
    const candidate = path.join(directory, entry);
    if (statSync(candidate).isDirectory()) visit(candidate);
    else if (entry.endsWith(".html")) htmlFiles.push(candidate);
  }
};

if (!existsSync(buildRoot)) {
  throw new Error(`Missing built documentation directory: ${buildRoot}`);
}
visit(buildRoot);

const routeChecks = [
  {
    file: "index.html",
    lang: "zh",
    copy: [
      "模型由你定义。",
      "执行交给 Power。",
      "简体中文",
      "v1.0.0",
      "v0.9.0",
      "一次前向，验证多个 token",
      "草稿候选",
      "本轮：5 个候选通过",
      "算法示意，非性能数据",
      "文档",
      "性能",
      "复现",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: path.join("en", "index.html"),
    lang: "en",
    copy: [
      "You define the model.",
      "Power runs it.",
      "English",
      "v1.0.0",
      "v0.9.0",
      "Verify more tokens per forward pass",
      "Draft proposals",
      "This round: 5 candidates matched",
      "Algorithm illustration, not performance data",
      "Docs",
      "Performance",
      "Reproduce",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: path.join("v1.0.0", "index.html"),
    lang: "zh",
    copy: [
      "模型由你定义。",
      "执行交给 Power。",
      "简体中文",
      "next",
      "一次前向，验证多个 token",
      "草稿候选",
      "本轮：5 个候选通过",
      "算法示意，非性能数据",
      "文档",
      "性能",
      "复现",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: path.join("v1.0.0", "en", "index.html"),
    lang: "en",
    copy: [
      "You define the model.",
      "Power runs it.",
      "English",
      "next",
      "Verify more tokens per forward pass",
      "Draft proposals",
      "This round: 5 candidates matched",
      "Algorithm illustration, not performance data",
      "Docs",
      "Performance",
      "Reproduce",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: path.join("v0.9.0", "index.html"),
    lang: "zh",
    copy: [
      "模型由你定义。",
      "执行交给 Power。",
      "简体中文",
      "next",
      "一次前向，验证多个 token",
      "草稿候选",
      "本轮：5 个候选通过",
      "算法示意，非性能数据",
      "文档",
      "性能",
      "复现",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: path.join("v0.9.0", "en", "index.html"),
    lang: "en",
    copy: [
      "You define the model.",
      "Power runs it.",
      "English",
      "next",
      "Verify more tokens per forward pass",
      "Draft proposals",
      "This round: 5 candidates matched",
      "Algorithm illustration, not performance data",
      "Docs",
      "Performance",
      "Reproduce",
      "174.413",
      "6 / 6",
    ],
  },
  {
    file: "optimization.html",
    lang: "zh",
    copy: [
      "优化手册",
      "完整优化地图",
      "Power 可以约束并记录模型自有决策",
      "K7/S6/B11",
      "73.57%",
    ],
  },
  {
    file: path.join("en", "optimization.html"),
    lang: "en",
    copy: [
      "Optimization playbook",
      "Complete optimization map",
      "Power may constrain and record model-owned choices",
      "K7/S6/B11",
      "73.57%",
    ],
  },
  {
    file: "performance.html",
    lang: "zh",
    copy: ["性能证据", "174.413", "1.736", "智力水平下降了吗？"],
  },
  {
    file: path.join("en", "performance.html"),
    lang: "en",
    copy: ["Performance evidence", "174.413", "1.736", "Did quality fall?"],
  },
  {
    file: path.join("v1.0.0", "performance.html"),
    lang: "zh",
    copy: ["性能证据", "174.413", "智力水平下降了吗？"],
  },
  {
    file: path.join("v1.0.0", "en", "performance.html"),
    lang: "en",
    copy: ["Performance evidence", "174.413", "Did quality fall?"],
  },
  {
    file: path.join("v0.9.0", "performance.html"),
    lang: "zh",
    copy: ["性能证据", "176.6109"],
  },
  {
    file: path.join("v0.9.0", "en", "performance.html"),
    lang: "en",
    copy: ["Performance evidence", "176.6109"],
  },
  {
    file: "reproduction.html",
    lang: "zh",
    copy: [
      "复现实验",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: path.join("en", "reproduction.html"),
    lang: "en",
    copy: [
      "Reproduction",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: path.join("v1.0.0", "reproduction.html"),
    lang: "zh",
    copy: [
      "复现实验",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: path.join("v1.0.0", "en", "reproduction.html"),
    lang: "en",
    copy: [
      "Reproduction",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: path.join("v0.9.0", "reproduction.html"),
    lang: "zh",
    copy: [
      "复现实验",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: path.join("v0.9.0", "en", "reproduction.html"),
    lang: "en",
    copy: [
      "Reproduction",
      "verify-qwen38-q6k-evidence.ps1",
      "run-qwen38-q6k-benchmark.ps1",
      "562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727",
    ],
  },
  {
    file: "verification.html",
    lang: "zh",
    copy: [
      "复现外部硬件捕获",
      "--promote-capture",
      "模型中立",
      "生产发布信任链",
    ],
  },
  {
    file: path.join("en", "verification.html"),
    lang: "en",
    copy: [
      "Reproduce external hardware capture",
      "--promote-capture",
      "model-neutral",
      "Production release trust chain",
    ],
  },
  {
    file: path.join("v1.0.0", "verification.html"),
    lang: "zh",
    copy: [
      "复现外部硬件捕获",
      "--promote-capture",
      "模型中立",
      "生产发布信任链",
    ],
  },
  {
    file: path.join("v1.0.0", "en", "verification.html"),
    lang: "en",
    copy: [
      "Reproduce external hardware capture",
      "--promote-capture",
      "model-neutral",
      "Production release trust chain",
    ],
  },
];

for (const route of routeChecks) {
  const renderedPath = path.join(buildRoot, route.file);
  if (!existsSync(renderedPath)) {
    throw new Error(`Missing rendered route: ${route.file}`);
  }

  const html = readFileSync(renderedPath, "utf8");
  const renderedText = html.replace(/<[^>]+>/g, "");
  if (!html.includes(`<html lang="${route.lang}">`)) {
    throw new Error(`Rendered route has the wrong language: ${route.file}`);
  }
  if (!html.includes("a3s-os-logo.png")) {
    throw new Error(`Rendered route is missing the A3S OS logo: ${route.file}`);
  }
  for (const copy of route.copy) {
    if (!html.includes(copy) && !renderedText.includes(copy)) {
      throw new Error(`Rendered route ${route.file} is missing copy: ${copy}`);
    }
  }
}

const assetChecks = ["a3s-os-logo.png", "social-card.svg"];
for (const asset of assetChecks) {
  if (!existsSync(path.join(buildRoot, asset))) {
    throw new Error(`Built documentation is missing public asset: ${asset}`);
  }
}

const logoHash = createHash("sha256")
  .update(readFileSync(path.join(buildRoot, "a3s-os-logo.png")))
  .digest("hex");
const expectedLogoHash =
  "72b94cf69a95dc6153f865c4f8742c0f67079caa876f35f8b2b5f970ea795a2d";
if (logoHash !== expectedLogoHash) {
  throw new Error(`Built A3S OS logo has an unexpected SHA-256: ${logoHash}`);
}

if (htmlFiles.length < 51) {
  throw new Error(
    `Expected 50 localized/versioned pages plus 404, found ${htmlFiles.length}`,
  );
}

console.log(
  `Verified ${htmlFiles.length} rendered pages, six locale/version roots, and official A3S OS assets.`,
);
