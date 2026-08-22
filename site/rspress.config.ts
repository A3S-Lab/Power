import * as path from "node:path";
import {
  defineConfig,
  type Nav,
  type NavItem,
  type UserConfig,
} from "@rspress/core";

const base = process.env.DOCS_BASE ?? "/Power/";
const siteOrigin = process.env.DOCS_ORIGIN ?? "https://a3s-lab.github.io";
const docVersions = ["next", "v1.0.0", "v0.9.0"] as const;

interface NavigationLabels {
  architecture: string;
  docs: string;
  gettingStarted: string;
  optimization: string;
  operations: string;
  performance: string;
  reproduce: string;
  speculativeDecoding: string;
  verification: string;
}

function createNavigation(
  localePath: "" | "en",
  labels: NavigationLabels,
): Nav {
  const navigation: Record<string, NavItem[]> = {};

  for (const version of docVersions) {
    const routePrefix = [version === "next" ? "" : version, localePath]
      .filter(Boolean)
      .join("/");
    const route = (page: string) =>
      `/${[routePrefix, page].filter(Boolean).join("/")}`;

    navigation[version] = [
      {
        text: labels.docs,
        link: route("getting-started"),
        position: "left",
        items: [
          { text: labels.gettingStarted, link: route("getting-started") },
          { text: labels.architecture, link: route("architecture") },
          {
            text: labels.speculativeDecoding,
            link: route("speculative-decoding"),
          },
          { text: labels.verification, link: route("verification") },
          { text: labels.operations, link: route("operations") },
        ],
      },
      ...(version === "next"
        ? [
            {
              text: labels.optimization,
              link: route("optimization"),
              activeMatch: "/optimization",
              position: "left" as const,
            },
          ]
        : []),
      {
        text: labels.performance,
        link: route("performance"),
        activeMatch: "/performance",
        position: "left",
      },
      {
        text: labels.reproduce,
        link: route("reproduction"),
        activeMatch: "/reproduction",
        position: "left",
      },
    ];
  }

  return navigation;
}

const zhNavigation = createNavigation("", {
  architecture: "架构设计",
  docs: "文档",
  gettingStarted: "快速开始",
  optimization: "优化",
  operations: "部署",
  performance: "性能",
  reproduce: "复现",
  speculativeDecoding: "推测解码",
  verification: "验证",
});

const enNavigation = createNavigation("en", {
  architecture: "Architecture",
  docs: "Docs",
  gettingStarted: "Getting started",
  optimization: "Optimization",
  operations: "Operations",
  performance: "Performance",
  reproduce: "Reproduce",
  speculativeDecoding: "Speculative decoding",
  verification: "Verification",
});

const config: UserConfig = {
  root: path.join(__dirname, "docs"),
  base,
  siteOrigin,
  title: "A3S Power",
  description:
    "在 Rust 进程或服务端运行语言、视觉、OCR、嵌入和音频模型，统一管理设备、调度、制品和执行验证。",
  lang: "zh",
  icon: "/a3s-os-logo.png",
  logo: "/a3s-os-logo.png",
  logoText: "A3S Power",
  outDir: "doc_build",
  llms: true,
  route: {
    localeRedirect: "never",
  },
  multiVersion: {
    default: "next",
    versions: [...docVersions],
  },
  locales: [
    {
      lang: "zh",
      label: "简体中文",
      title: "A3S Power",
      description:
        "在 Rust 进程或服务端运行语言、视觉、OCR、嵌入和音频模型，统一管理设备、调度、制品和执行验证。",
    },
    {
      lang: "en",
      label: "English",
      title: "A3S Power",
      description:
        "Run language, vision, OCR, embedding, and audio models in a Rust process or behind an API, with shared device, scheduling, artifact, and verification controls.",
    },
  ],
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
    locales: [
      { lang: "zh", label: "简体中文", nav: zhNavigation },
      { lang: "en", label: "English", nav: enNavigation },
    ],
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
