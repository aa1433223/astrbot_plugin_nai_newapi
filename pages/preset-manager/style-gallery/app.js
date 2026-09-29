import { createStyleApi } from "./api.js";

// This view is mounted into the shared NovelAI management page.

const bridge = window.AstrBotPluginPage;
let viewRoot = null;

const ICONS = {
  edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 20 4.2-1 10.6-10.6-3.2-3.2L5 15.8zM14.5 6.5l3 3"/></svg>',
  delete:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16m-10 4v6m4-6v6M9 7l1-3h6l1 3m1 0-1 14H7L6 7"/></svg>',
  image:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 5.5h17v13h-17zM3.5 15l5-5 4 4 2.5-2.5 5.5 5.5"/><circle cx="16.5" cy="9" r="1.5"/></svg>',
};

let els = null;

function collectElements(root) {
  const byId = (id) => root.querySelector(`[data-view-id="${id}"]`);
  return {
    totalBadge: byId("totalBadge"),
    coverageText: byId("coverageText"),
    syncStatus: byId("syncStatus"),
    refreshButton: byId("refreshButton"),
    newStyleButton: byId("newStyleButton"),
    searchInput: byId("searchInput"),
    styleGallery: byId("styleGallery"),
    loadingState: byId("loadingState"),
    editorDialog: byId("editorDialog"),
    editorForm: byId("editorForm"),
    editorTitle: byId("editorTitle"),
    closeEditorButton: byId("closeEditorButton"),
    cancelEditorButton: byId("cancelEditorButton"),
    styleName: byId("styleName"),
    positivePrompt: byId("positivePrompt"),
    negativePrompt: byId("negativePrompt"),
    positiveLength: byId("positiveLength"),
    negativeLength: byId("negativeLength"),
    previewInput: byId("previewInput"),
    uploadPreviewButton: byId("uploadPreviewButton"),
    uploadStatus: byId("uploadStatus"),
    editorPreview: byId("editorPreview"),
    editorPreviewImage: byId("editorPreviewImage"),
    editorPreviewEmpty: byId("editorPreviewEmpty"),
    editorError: byId("editorError"),
    saveStyleButton: byId("saveStyleButton"),
    imageDialog: byId("imageDialog"),
    closeImageButton: byId("closeImageButton"),
    largePreviewImage: byId("largePreviewImage"),
    largePreviewTitle: byId("largePreviewTitle"),
    confirmDialog: byId("confirmDialog"),
    confirmMessage: byId("confirmMessage"),
    cancelDeleteButton: byId("cancelDeleteButton"),
    confirmDeleteButton: byId("confirmDeleteButton"),
    toastRegion: byId("toastRegion"),
  };
}

let api = null;
let libraryState = { styles: [], stats: { total: 0, with_preview: 0 } };
let editingOriginalName = null;
let currentPreviewImage = "";
let currentPreviewUrl = "";
let pendingDeleteName = "";
let operationPending = false;
let detachContextHandler = null;
let searchRenderTimer = null;

function normalizeState(payload) {
  if (!payload || !Array.isArray(payload.styles)) {
    throw new Error("服务器返回了无效的画风数据。");
  }
  const styles = payload.styles.map((style) => ({
    name: String(style.name || ""),
    positive_prompt: String(style.positive_prompt || ""),
    negative_prompt: String(style.negative_prompt || ""),
    preview_image: String(style.preview_image || ""),
    preview_url: String(style.preview_url || ""),
  }));
  return {
    styles,
    stats: {
      total: Number(payload.stats?.total ?? styles.length),
      with_preview: Number(
        payload.stats?.with_preview ?? styles.filter((style) => style.preview_url).length,
      ),
    },
  };
}

function getVisibleStyles() {
  const query = String(els.searchInput.value || "").trim().toLocaleLowerCase();
  if (!query) return libraryState.styles;
  return libraryState.styles.filter((style) => {
    return [style.name, style.positive_prompt, style.negative_prompt].some((value) =>
      value.toLocaleLowerCase().includes(query),
    );
  });
}

function setSyncStatus(mode, label) {
  els.syncStatus.dataset.status = mode;
  els.syncStatus.lastElementChild.textContent = label;
}

function setOperationPending(pending) {
  operationPending = pending;
  els.refreshButton.disabled = pending;
  els.newStyleButton.disabled = pending;
  els.saveStyleButton.disabled = pending;
  els.uploadPreviewButton.disabled = pending;
  els.closeEditorButton.disabled = pending;
  els.cancelEditorButton.disabled = pending;
  els.styleGallery.setAttribute("aria-busy", String(pending));
  if (pending) setSyncStatus("saving", "正在保存");
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");
  const marker = document.createElement("span");
  marker.className = "toast-marker";
  marker.setAttribute("aria-hidden", "true");
  const text = document.createElement("span");
  text.textContent = message;
  toast.append(marker, text);
  els.toastRegion.appendChild(toast);
  window.setTimeout(() => {
    toast.classList.add("toast-leaving");
    window.setTimeout(() => toast.remove(), 180);
  }, 3200);
}

function applyState(payload) {
  libraryState = normalizeState(payload);
  els.totalBadge.textContent = `${libraryState.stats.total} 个画风`;
  els.coverageText.textContent = `示例图覆盖 ${libraryState.stats.with_preview} / ${libraryState.stats.total}`;
  renderGallery();
}

function renderGallery() {
  const visible = getVisibleStyles();
  if (!visible.length) {
    const hasSearch = Boolean(String(els.searchInput.value || "").trim());
    renderEmptyState(
      hasSearch ? "没有匹配的画风" : "画风库还是空的",
      hasSearch
        ? "尝试缩短搜索词，或搜索正面、负面提示词内容。"
        : "新建第一个画风，并上传一张代表性效果图。",
      false,
    );
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const style of visible) fragment.appendChild(createStyleCard(style));
  els.styleGallery.replaceChildren(fragment);
}

function renderEmptyState(title, description, isError) {
  const empty = document.createElement("div");
  empty.className = `empty-state${isError ? " error-state" : ""}`;
  const icon = document.createElement("div");
  icon.className = "empty-icon";
  icon.innerHTML = ICONS.image;
  const heading = document.createElement("strong");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = description;
  empty.append(icon, heading, copy);
  els.styleGallery.replaceChildren(empty);
}

function createStyleCard(style) {
  const article = document.createElement("article");
  article.className = "style-card";

  const media = document.createElement(style.preview_url ? "button" : "div");
  if (style.preview_url) {
    media.type = "button";
    media.setAttribute("aria-label", `查看 ${style.name} 的示例大图`);
    media.addEventListener("click", () => openImage(style));
  }
  media.className = "style-media";
  if (style.preview_url) {
    const image = document.createElement("img");
    image.src = style.preview_url;
    image.alt = `${style.name}画风示例图`;
    image.loading = "lazy";
    image.decoding = "async";
    image.addEventListener("error", () => {
      media.classList.add("media-error");
      media.replaceChildren(createMissingPreview());
    });
    media.appendChild(image);
  } else {
    media.classList.add("media-missing");
    media.appendChild(createMissingPreview());
  }

  const body = document.createElement("div");
  body.className = "style-body";
  const header = document.createElement("div");
  header.className = "style-card-header";
  const title = document.createElement("h3");
  title.textContent = style.name;
  const actions = document.createElement("div");
  actions.className = "card-actions";
  actions.append(
    createCardAction("编辑", "edit", () => openEditor(style)),
    createCardAction("删除", "delete", () => requestDelete(style.name), true),
  );
  header.append(title, actions);

  const prompts = document.createElement("div");
  prompts.className = "prompt-pair";
  prompts.append(
    createPromptBlock("正面", style.positive_prompt, "positive"),
    createPromptBlock("负面", style.negative_prompt || "待补充负面提示词", "negative"),
  );
  body.append(header, prompts);
  article.append(media, body);
  return article;
}

function createMissingPreview() {
  const empty = document.createElement("span");
  empty.className = "missing-preview";
  empty.innerHTML = ICONS.image;
  const label = document.createElement("strong");
  label.textContent = "待上传示例图";
  empty.appendChild(label);
  return empty;
}

function createPromptBlock(labelText, content, kind) {
  const block = document.createElement("div");
  block.className = `prompt-block prompt-${kind}`;
  const label = document.createElement("span");
  label.textContent = labelText;
  const code = document.createElement("code");
  code.textContent = content;
  code.title = content;
  block.append(label, code);
  return block;
}

function createCardAction(label, iconName, handler, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `card-action${danger ? " danger" : ""}`;
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = ICONS[iconName];
  button.addEventListener("click", handler);
  return button;
}

function setEditorPreview(previewImage, previewUrl) {
  currentPreviewImage = previewImage || "";
  currentPreviewUrl = previewUrl || "";
  const hasPreview = Boolean(currentPreviewUrl);
  els.editorPreview.dataset.empty = String(!hasPreview);
  els.editorPreviewImage.hidden = !hasPreview;
  els.editorPreviewEmpty.hidden = hasPreview;
  if (hasPreview) els.editorPreviewImage.src = currentPreviewUrl;
  else els.editorPreviewImage.removeAttribute("src");
}

function openEditor(style = null) {
  if (operationPending) return;
  editingOriginalName = style?.name || null;
  els.editorTitle.textContent = style ? "编辑画风" : "新建画风";
  els.styleName.value = style?.name || "";
  els.positivePrompt.value = style?.positive_prompt || "";
  els.negativePrompt.value = style?.negative_prompt || "";
  els.positiveLength.textContent = String(els.positivePrompt.value.length);
  els.negativeLength.textContent = String(els.negativePrompt.value.length);
  els.previewInput.value = "";
  els.uploadStatus.textContent = style?.preview_url
    ? "已加载当前示例图"
    : "请上传一张示例图后保存";
  setEditorPreview(style?.preview_image, style?.preview_url);
  els.editorError.hidden = true;
  els.editorError.textContent = "";
  els.editorDialog.showModal();
  window.setTimeout(() => els.styleName.focus(), 0);
}

function closeEditor() {
  if (!operationPending && els.editorDialog.open) els.editorDialog.close();
}

async function uploadPreview() {
  const file = els.previewInput.files?.[0];
  if (!file || operationPending) return;
  if (file.size > 6 * 1024 * 1024) {
    els.editorError.textContent = "示例图不能超过 6 MB。";
    els.editorError.hidden = false;
    return;
  }
  setOperationPending(true);
  els.uploadStatus.textContent = "正在压缩并上传…";
  els.editorError.hidden = true;
  try {
    const result = await api.uploadPreview(file);
    setEditorPreview(String(result.preview_image || ""), String(result.preview_url || ""));
    els.uploadStatus.textContent = "示例图已上传，保存画风后生效";
    setSyncStatus("synced", "图片已上传");
  } catch (error) {
    els.uploadStatus.textContent = "上传失败，请重新选择";
    els.editorError.textContent = error.message || "示例图上传失败。";
    els.editorError.hidden = false;
    setSyncStatus("error", "上传失败");
  } finally {
    setOperationPending(false);
  }
}

async function saveEditor(event) {
  event.preventDefault();
  if (operationPending) return;
  const payload = {
    name: els.styleName.value.trim(),
    positive_prompt: els.positivePrompt.value.trim(),
    negative_prompt: els.negativePrompt.value.trim(),
    preview_image: currentPreviewImage,
  };
  if (!payload.name || !payload.positive_prompt || !payload.negative_prompt) {
    els.editorError.textContent = "请填写画风名称、正面提示词和负面提示词。";
    els.editorError.hidden = false;
    return;
  }
  if (!payload.preview_image) {
    els.editorError.textContent = "请上传一张画风示例图。";
    els.editorError.hidden = false;
    return;
  }

  setOperationPending(true);
  els.saveStyleButton.textContent = "正在保存…";
  els.editorError.hidden = true;
  try {
    const state = editingOriginalName
      ? await api.update({ original_name: editingOriginalName, ...payload })
      : await api.create(payload);
    applyState(state);
    els.editorDialog.close();
    setSyncStatus("synced", "已同步");
    showToast(editingOriginalName ? "画风修改已保存" : "画风已创建");
  } catch (error) {
    els.editorError.textContent = error.message || "画风保存失败，请重试。";
    els.editorError.hidden = false;
    setSyncStatus("error", "保存失败");
  } finally {
    setOperationPending(false);
    els.saveStyleButton.textContent = "保存画风";
  }
}

function openImage(style) {
  els.largePreviewImage.src = style.preview_url;
  els.largePreviewImage.alt = `${style.name}画风示例大图`;
  els.largePreviewTitle.textContent = style.name;
  els.imageDialog.showModal();
  window.setTimeout(() => els.closeImageButton.focus(), 0);
}

function requestDelete(name) {
  if (!name || operationPending) return;
  pendingDeleteName = name;
  els.confirmMessage.textContent = `将永久删除“${name}”及其示例图。`;
  els.confirmDialog.showModal();
  window.setTimeout(() => els.cancelDeleteButton.focus(), 0);
}

async function confirmDelete() {
  if (!pendingDeleteName || operationPending) return;
  const name = pendingDeleteName;
  setOperationPending(true);
  els.confirmDeleteButton.disabled = true;
  els.cancelDeleteButton.disabled = true;
  els.confirmDeleteButton.textContent = "正在删除…";
  try {
    const state = await api.delete({ name });
    applyState(state);
    els.confirmDialog.close();
    setSyncStatus("synced", "已同步");
    showToast("画风已删除");
  } catch (error) {
    setSyncStatus("error", "删除失败");
    showToast(error.message || "画风删除失败", "error");
  } finally {
    pendingDeleteName = "";
    setOperationPending(false);
    els.confirmDeleteButton.disabled = false;
    els.cancelDeleteButton.disabled = false;
    els.confirmDeleteButton.textContent = "确认删除";
  }
}

async function loadState({ initial = false } = {}) {
  if (operationPending) return;
  if (!initial) setOperationPending(true);
  setSyncStatus("saving", initial ? "正在连接" : "正在刷新");
  try {
    const payload = await api.loadState();
    applyState(payload);
    els.loadingState.hidden = true;
    setSyncStatus("synced", "已同步");
    if (!initial) showToast("画风库已刷新");
  } catch (error) {
    els.loadingState.hidden = true;
    setSyncStatus("error", "连接失败");
    renderEmptyState("无法加载画风库", error.message || "请刷新页面后重试。", true);
    showToast(error.message || "画风库加载失败", "error");
  } finally {
    if (!initial) setOperationPending(false);
  }
}

function bindEvents() {
  els.refreshButton.addEventListener("click", () => loadState());
  els.newStyleButton.addEventListener("click", () => openEditor());
  els.searchInput.addEventListener("input", () => {
    window.clearTimeout(searchRenderTimer);
    searchRenderTimer = window.setTimeout(renderGallery, 100);
  });
  els.closeEditorButton.addEventListener("click", closeEditor);
  els.cancelEditorButton.addEventListener("click", closeEditor);
  els.editorForm.addEventListener("submit", saveEditor);
  els.uploadPreviewButton.addEventListener("click", () => els.previewInput.click());
  els.previewInput.addEventListener("change", uploadPreview);
  els.positivePrompt.addEventListener("input", () => {
    els.positiveLength.textContent = String(els.positivePrompt.value.length);
  });
  els.negativePrompt.addEventListener("input", () => {
    els.negativeLength.textContent = String(els.negativePrompt.value.length);
  });
  els.closeImageButton.addEventListener("click", () => els.imageDialog.close());
  els.cancelDeleteButton.addEventListener("click", () => {
    pendingDeleteName = "";
    els.confirmDialog.close();
  });
  els.confirmDeleteButton.addEventListener("click", confirmDelete);
  document.addEventListener("keydown", (event) => {
    if (viewRoot.hidden) return;
    if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      els.searchInput.focus();
      els.searchInput.select();
    }
  });
}

function syncContext(context) {
  const locale = String(context?.locale || "zh-CN");
  document.documentElement.lang = locale;
}

async function bootstrap() {
  bindEvents();
  try {
    const context = await bridge.ready();
    syncContext(context);
    detachContextHandler = bridge.onContext(syncContext);
    api = createStyleApi(bridge);
    await loadState({ initial: true });
  } catch (error) {
    els.loadingState.hidden = true;
    setSyncStatus("error", "连接失败");
    renderEmptyState(
      "无法连接 AstrBot",
      error.message || "插件页面 Bridge 不可用。",
      true,
    );
  }
}

window.addEventListener("beforeunload", () => {
  if (typeof detachContextHandler === "function") detachContextHandler();
});

export function initializeStyleGallery(root) {
  if (!(root instanceof HTMLElement)) {
    throw new Error("画风库视图挂载点不可用。");
  }
  viewRoot = root;
  els = collectElements(root);
  return bootstrap();
}
