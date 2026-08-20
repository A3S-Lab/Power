import * as path from "node:path";
import { defineConfig, type UserConfig } from "@rspress/core";

const base = process.env.DOCS_BASE ?? "/Power/";
const siteOrigin = process.env.DOCS_ORIGIN ?? "https://a3s-lab.github.io";

const config: UserConfig = {
  root: path.join(__dirname, "docs"),
  base,
  siteOrigin,
  title: "A3S Power",
  description:
    "A model-neutral Rust runtime for bounded, verifiable inference in embedded, hosted, and TEE environments.",
  lang: "en",
  icon: "/power-mark.svg",
  logo: "/power-mark.svg",
  logoText: "A3S Power",
  outDir: "doc_build",
  llms: true,
  head: [
    ["meta", { name: "theme-color", content: "#f7f7f8" }],
    ["meta", { property: "og:type", content: "website" }],
    ["meta", { property: "og:site_name", content: "A3S Power" }],
    [
      "meta",
      {
        property: "og:image",
        content: `${siteOrigin}${base}social-card.svg`,
      },
    ],
    ["meta", { name: "twitter:card", content: "summary_large_image" }],
    (route) => [
      "link",
      {
        rel: "canonical",
        href: `${siteOrigin}${base.replace(/\/$/, "")}${route.routePath}`,
      },
    ],
  ],
  themeConfig: {
    search: true,
    enableContentAnimation: true,
    editLink: {
      docRepoBaseUrl: "https://github.com/A3S-Lab/Power/tree/main/site/docs",
    },
    lastUpdated: true,
    llmsUI: {
      placement: "outline",
      viewOptions: ["markdownLink", "chatgpt", "claude"],
    },
    socialLinks: [
      {
        icon: "github",
        mode: "link",
        content: "https://github.com/A3S-Lab/Power",
      },
    ],
  },
};

export default defineConfig(config);
