import { initializePresetManager } from "./presets/app.js";
import { initializeStyleGallery } from "./style-gallery/app.js";

const bridge = window.AstrBotPluginPage;
const tabs = [...document.querySelectorAll(".management-tab[data-view]")];
const panels = {
  presets: document.getElementById("presetsPanel"),
  styles: document.getElementById("stylesPanel"),
};
const shellStyle = document.querySelector('link[href*="styles.css"]');
const loadingState = document.getElementById("shellLoading");
const errorState = document.getElementById("shellError");
const errorMessage = document.getElementById("shellErrorMessage");
const retryButton = document.getElementById("shellRetryButton");
const viewStyles = {};
let activeView = "presets";

function assetUrl(path) {
  const url = new URL(path, window.location.href);
  const current = new URLSearchParams(window.location.search);
  current.forEach((value, key) => url.searchParams.set(key, value));
  url.hash = "";
  return url;
}

function namespacePanel(panel, prefix) {
  const idMap = new Map();
  panel.querySelectorAll("[id]").forEach((element) => {
    const originalId = element.id;
    const namespacedId = `${prefix}-${originalId}`;
    idMap.set(originalId, namespacedId);
    element.dataset.viewId = originalId;
    element.id = namespacedId;
  });

  const singleIdAttributes = ["for", "list"];
  const multipleIdAttributes = [
    "aria-controls",
    "aria-describedby",
    "aria-labelledby",
    "aria-owns",
  ];
  panel.querySelectorAll("*").forEach((element) => {
    singleIdAttributes.forEach((attribute) => {
      const value = element.getAttribute(attribute);
      if (value && idMap.has(value)) {
        element.setAttribute(attribute, idMap.get(value));
      }
    });
    multipleIdAttributes.forEach((attribute) => {
      const value = element.getAttribute(attribute);
      if (!value) return;
      element.setAttribute(
        attribute,
        value
          .split(/\s+/)
          .map((id) => idMap.get(id) || id)
          .join(" "),
      );
    });
    const href = element.getAttribute("href");
    if (href?.startsWith("#") && idMap.has(href.slice(1))) {
      element.setAttribute("href", `#${idMap.get(href.slice(1))}`);
    }
  });
}

function installStylesheet(documentFragment, view) {
  const source = documentFragment.querySelector('link[rel="stylesheet"]');
  if (!source?.href) {
    throw new Error(`${view} 视图缺少样式表。`);
  }
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = source.href;
  link.dataset.viewStyles = view;
  shellStyle.before(link);
  viewStyles[view] = link;
  return new Promise((resolve, reject) => {
    link.addEventListener("load", resolve, { once: true });
    link.addEventListener(
      "error",
      () => reject(new Error(`${view} 视图样式加载失败。`)),
      { once: true },
    );
  });
}

async function loadView(view, path) {
  const response = await fetch(assetUrl(path));
  if (!response.ok) {
    throw new Error(`${view} 视图请求失败（HTTP ${response.status}）。`);
  }
  const parsed = new DOMParser().parseFromString(await response.text(), "text/html");
  const styleReady = installStylesheet(parsed, view);
  const content = document.createDocumentFragment();
  [...parsed.body.childNodes].forEach((node) => {
    if (node.nodeType === Node.ELEMENT_NODE && node.tagName === "SCRIPT") return;
    content.appendChild(document.importNode(node, true));
  });
  panels[view].replaceChildren(content);
  namespacePanel(panels[view], view);
  await styleReady;
}

function activateView(view, { updateHash = true, focus = false } = {}) {
  if (!(view in panels)) return;
  activeView = view;
  tabs.forEach((tab) => {
    const selected = tab.dataset.view === view;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
    if (selected && focus) tab.focus();
  });
  Object.entries(panels).forEach(([name, panel]) => {
    panel.hidden = name !== view;
  });
  Object.entries(viewStyles).forEach(([name, stylesheet]) => {
    stylesheet.disabled = name !== view;
  });
  if (updateHash) {
    const nextHash = view === "styles" ? "#styles" : "#presets";
    if (window.location.hash !== nextHash) window.location.hash = nextHash;
  }
}

function bindTabs() {
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activateView(tab.dataset.view));
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      activateView(tabs[nextIndex].dataset.view, { focus: true });
    });
  });
  window.addEventListener("hashchange", () => {
    const view = window.location.hash === "#styles" ? "styles" : "presets";
    if (view !== activeView) activateView(view, { updateHash: false });
  });
  retryButton.addEventListener("click", () => window.location.reload());
}

function showShellError(error) {
  loadingState.hidden = true;
  errorMessage.textContent = error?.message || "请刷新页面后重试。";
  errorState.hidden = false;
}

async function bootstrap() {
  bindTabs();
  try {
    await Promise.all([
      loadView("presets", "./presets/index.html"),
      loadView("styles", "./style-gallery/index.html"),
      bridge.ready(),
    ]);
    const initialView = window.location.hash === "#styles" ? "styles" : "presets";
    activateView(initialView, { updateHash: false });
    await Promise.all([
      initializePresetManager(panels.presets),
      initializeStyleGallery(panels.styles),
    ]);
    document.title = bridge.t("pages.preset-manager.title", "NovelAI 管理");
    loadingState.hidden = true;
  } catch (error) {
    showShellError(error);
  }
}

bootstrap();
