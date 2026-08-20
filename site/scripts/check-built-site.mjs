import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as path from "node:path";

const siteRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const buildRoot = path.join(siteRoot, "doc_build");
const indexPath = path.join(buildRoot, "index.html");

if (!existsSync(indexPath)) {
  throw new Error(`Missing built homepage: ${indexPath}`);
}

const htmlFiles = [];
const visit = (directory) => {
  for (const entry of readdirSync(directory)) {
    const candidate = path.join(directory, entry);
    if (statSync(candidate).isDirectory()) visit(candidate);
    else if (entry.endsWith(".html")) htmlFiles.push(candidate);
  }
};
visit(buildRoot);

const renderedHtml = htmlFiles
  .map((file) => readFileSync(file, "utf8"))
  .join("\n");
const requiredCopy = [
  "Run the model.",
  "Prove the boundary.",
  "175.2089",
  "Embedded Inference Architecture",
  "Model-neutral Speculative Decoding",
];

for (const copy of requiredCopy) {
  if (!renderedHtml.includes(copy)) {
    throw new Error(`Built documentation is missing required copy: ${copy}`);
  }
}

for (const asset of ["power-mark.svg", "social-card.svg"]) {
  if (!existsSync(path.join(buildRoot, asset))) {
    throw new Error(`Built documentation is missing public asset: ${asset}`);
  }
}

if (htmlFiles.length < 7) {
  throw new Error(`Expected at least 7 rendered documentation pages, found ${htmlFiles.length}`);
}

console.log(`Verified ${htmlFiles.length} rendered pages and public assets.`);
