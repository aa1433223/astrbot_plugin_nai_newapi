import { createPresetApi } from "./api.js";

// This view is mounted into the shared NovelAI management page.

const bridge = window.AstrBotPluginPage;
let viewRoot = null;

const CATEGORY_ICONS = {
  unclassified:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 7.5h6l2-2h9v13h-17z"/><path d="M8 12h8m-4-4v8"/></svg>',
  character:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="8" r="3.5"/><path d="M5 20c.6-4 3-6 7-6s6.4 2 7 6"/></svg>',
  clothing:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 4 2 4-2 4 3-3 4v8H7v-8L4 8z"/><path d="M10 5c.2 1.3.9 2 2 2s1.8-.7 2-2"/></svg>',
  scene:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 19.5h17l-5.5-8-3.5 5-2.5-3z"/><circle cx="8" cy="8" r="2.5"/></svg>',
  action:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="13" cy="5" r="2"/><path d="m10 21 1.5-6-3-3 3-4 4 2 3-1M6 21l3-5"/></svg>',
};

const ACTION_ICONS = {
  edit:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 20 4.2-1 10.6-10.6-3.2-3.2L5 15.8zM14.5 6.5l3 3"/></svg>',
  delete:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16m-10 4v6m4-6v6M9 7l1-3h6l1 3m1 0-1 14H7L6 7"/></svg>',
  grip:
    '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="9" cy="7" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="7" r="1" fill="currentColor" stroke="none"/><circle cx="9" cy="12" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="12" r="1" fill="currentColor" stroke="none"/><circle cx="9" cy="17" r="1" fill="currentColor" stroke="none"/><circle cx="15" cy="17" r="1" fill="currentColor" stroke="none"/></svg>',
};

let els = null;

function collectElements(root) {
  const byId = (id) => root.querySelector(`[data-view-id="${id}"]`);
  return {
    totalBadge: byId("totalBadge"),
    syncStatus: byId("syncStatus"),
    refreshButton: byId("refreshButton"),
    newPresetButton: byId("newPresetButton"),
    categoryList: byId("categoryList"),
    currentCategoryTitle: byId("currentCategoryTitle"),
    currentCategoryDescription: byId("currentCategoryDescription"),
    visibleCount: byId("visibleCount"),
    searchInput: byId("searchInput"),
    bulkBar: byId("bulkBar"),
    selectedCount: byId("selectedCount"),
    clearSelectionButton: byId("clearSelectionButton"),
    bulkCategory: byId("bulkCategory"),
    bulkMoveButton: byId("bulkMoveButton"),
    bulkDeleteButton: byId("bulkDeleteButton"),
    selectVisible: byId("selectVisible"),
    presetList: byId("presetList"),
    presetTable: root.querySelector(".preset-table"),
    loadingState: byId("loadingState"),
    editorDialog: byId("editorDialog"),
    editorForm: byId("editorForm"),
    editorTitle: byId("editorTitle"),
    closeEditorButton: byId("closeEditorButton"),
    cancelEditorButton: byId("cancelEditorButton"),
    presetName: byId("presetName"),
    presetCategory: byId("presetCategory"),
    presetPrompt: byId("presetPrompt"),
    promptLength: byId("promptLength"),
    editorError: byId("editorError"),
    savePresetButton: byId("savePresetButton"),
    confirmDialog: byId("confirmDialog"),
    confirmMessage: byId("confirmMessage"),
    cancelDeleteButton: byId("cancelDeleteButton"),
    confirmDeleteButton: byId("confirmDeleteButton"),
    toastRegion: byId("toastRegion"),
  };
}

let api = null;
let managerState = { categories: [], stats: { total: 0, unclassified: 0 } };
let activeCategoryId = "unclassified";
let selectedNames = new Set();
let editingOriginalName = null;
let pendingDeleteNames = [];
let draggingNames = [];
let operationPending = false;
let detachContextHandler = null;
let searchRenderTimer = null;

function getCategory(categoryId = activeCategoryId) {
  return managerState.categories.find((category) => category.id === categoryId) || null;
}

function getVisiblePresets() {
  const category = getCategory();
  if (!category) return [];
  const query = String(els.searchInput.value || "").trim().toLocaleLowerCase();
  if (!query) return category.presets;
  return category.presets.filter((preset) => {
    return (
      preset.name.toLocaleLowerCase().includes(query) ||
      preset.prompt.toLocaleLowerCase().includes(query)
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
  els.newPresetButton.disabled = pending;
  els.bulkMoveButton.disabled = pending;
  els.bulkDeleteButton.disabled = pending;
  els.selectVisible.disabled = pending || getVisiblePresets().length === 0;
  els.presetTable.setAttribute("aria-busy", String(pending));
  if (pending) {
    setSyncStatus("saving", "正在保存");
  }
}

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");

  const marker = document.createElement("span");
  marker.className = "toast-marker";
  marker.setAttribute("aria-hidden", "true");
  toast.appendChild(marker);

  const text = document.createElement("span");
  text.textContent = message;
  toast.appendChild(text);
  els.toastRegion.appendChild(toast);

  window.setTimeout(() => {
    toast.classList.add("toast-leaving");
    window.setTimeout(() => toast.remove(), 180);
  }, 2800);
}

function normalizeState(payload) {
  if (!payload || !Array.isArray(payload.categories)) {
    throw new Error("服务器返回了无效的预设数据。");
  }
  return {
    categories: payload.categories.map((category) => ({
      id: String(category.id || ""),
      label: String(category.label || category.id || "未命名分类"),
      description: String(category.description || ""),
      count: Number(category.count || 0),
      presets: Array.isArray(category.presets)
        ? category.presets.map((preset) => ({
            name: String(preset.name || ""),
            prompt: String(preset.prompt || ""),
          }))
        : [],
    })),
    stats: {
      total: Number(payload.stats?.total || 0),
      unclassified: Number(payload.stats?.unclassified || 0),
    },
  };
}

function applyState(payload, { clearSelection = true } = {}) {
  managerState = normalizeState(payload);
  if (!getCategory(activeCategoryId)) {
    activeCategoryId = managerState.categories[0]?.id || "unclassified";
  }
  if (clearSelection) selectedNames.clear();
  renderPage();
  els.totalBadge.textContent = `${managerState.stats.total} 个预设`;
}

function renderPage() {
  renderCategories();
  renderCategorySummary();
  renderPresetList();
  renderBulkBar();
}

function renderCategories() {
  const fragment = document.createDocumentFragment();
  for (const category of managerState.categories) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "category-item";
    button.dataset.categoryId = category.id;
    button.title = `${category.label}：${category.description}`;
    button.setAttribute("aria-current", String(category.id === activeCategoryId));

    const icon = document.createElement("span");
    icon.className = "category-icon";
    icon.innerHTML = CATEGORY_ICONS[category.id] || CATEGORY_ICONS.unclassified;
    button.appendChild(icon);

    const copy = document.createElement("span");
    copy.className = "category-copy";
    const label = document.createElement("strong");
    label.textContent = category.label;
    const description = document.createElement("small");
    description.textContent = category.description;
    copy.append(label, description);
    button.appendChild(copy);

    const count = document.createElement("span");
    count.className = "category-count";
    count.textContent = String(category.count);
    button.appendChild(count);

    button.addEventListener("click", () => {
      if (operationPending || activeCategoryId === category.id) return;
      activeCategoryId = category.id;
      selectedNames.clear();
      renderPage();
    });
    button.addEventListener("dragover", (event) => {
      if (!draggingNames.length) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      button.classList.add("drop-target");
    });
    button.addEventListener("dragleave", () => button.classList.remove("drop-target"));
    button.addEventListener("drop", async (event) => {
      event.preventDefault();
      button.classList.remove("drop-target");
      if (!draggingNames.length || category.id === activeCategoryId) return;
      await movePresets(draggingNames, category.id);
    });
    fragment.appendChild(button);
  }
  els.categoryList.replaceChildren(fragment);
}

function renderCategorySummary() {
  const category = getCategory();
  if (!category) return;
  const visible = getVisiblePresets();
  const hasSearch = Boolean(String(els.searchInput.value || "").trim());
  els.currentCategoryTitle.textContent = category.label;
  els.currentCategoryDescription.textContent = category.description;
  els.visibleCount.textContent = hasSearch
    ? `${visible.length} / ${category.count}`
    : String(category.count);
}

function renderPresetList() {
  const category = getCategory();
  const visible = getVisiblePresets();
  const fragment = document.createDocumentFragment();

  if (!category) {
    renderListMessage("无法读取分类", "刷新页面后重试。", true);
    return;
  }

  if (!visible.length) {
    const hasSearch = Boolean(String(els.searchInput.value || "").trim());
    renderListMessage(
      hasSearch ? "没有匹配的预设" : "这个分类还是空的",
      hasSearch ? "尝试缩短搜索词，或搜索提示词内容。" : "新建预设，或从其他分类拖动到这里。",
      false,
    );
    updateSelectVisible(visible);
    return;
  }

  for (const preset of visible) {
    fragment.appendChild(createPresetRow(preset));
  }
  els.presetList.replaceChildren(fragment);
  updateSelectVisible(visible);
}

function renderListMessage(title, description, isError) {
  const empty = document.createElement("div");
  empty.className = `empty-state${isError ? " error-state" : ""}`;
  const icon = document.createElement("div");
  icon.className = "empty-icon";
  icon.innerHTML = CATEGORY_ICONS[activeCategoryId] || CATEGORY_ICONS.unclassified;
  const heading = document.createElement("strong");
  heading.textContent = title;
  const copy = document.createElement("p");
  copy.textContent = description;
  empty.append(icon, heading, copy);
  els.presetList.replaceChildren(empty);
}

function createPresetRow(preset) {
  const row = document.createElement("div");
  row.className = `preset-row${selectedNames.has(preset.name) ? " selected" : ""}`;
  row.dataset.presetName = preset.name;
  row.setAttribute("role", "row");

  const selectCell = document.createElement("div");
  selectCell.className = "select-cell";
  selectCell.setAttribute("role", "cell");
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = selectedNames.has(preset.name);
  checkbox.setAttribute("aria-label", `选择预设 ${preset.name}`);
  checkbox.addEventListener("change", () => {
    if (checkbox.checked) selectedNames.add(preset.name);
    else selectedNames.delete(preset.name);
    renderPresetList();
    renderBulkBar();
  });
  selectCell.appendChild(checkbox);
  row.appendChild(selectCell);

  const nameCell = document.createElement("div");
  nameCell.className = "name-cell";
  nameCell.setAttribute("role", "cell");
  const dragHandle = document.createElement("span");
  dragHandle.className = "drag-handle";
  dragHandle.draggable = true;
  dragHandle.title = "拖动以移动或排序；也可以勾选后使用批量移动";
  dragHandle.setAttribute("aria-hidden", "true");
  dragHandle.innerHTML = ACTION_ICONS.grip;
  dragHandle.addEventListener("dragstart", (event) => startDragging(event, preset.name, row));
  dragHandle.addEventListener("dragend", finishDragging);
  nameCell.appendChild(dragHandle);

  const nameButton = document.createElement("button");
  nameButton.type = "button";
  nameButton.className = "preset-name-button";
  nameButton.textContent = preset.name;
  nameButton.title = `编辑 ${preset.name}`;
  nameButton.addEventListener("click", () => openEditor(preset));
  nameCell.appendChild(nameButton);
  row.appendChild(nameCell);

  const promptCell = document.createElement("div");
  promptCell.className = "prompt-cell";
  promptCell.setAttribute("role", "cell");
  const prompt = document.createElement("code");
  prompt.textContent = preset.prompt;
  prompt.title = preset.prompt;
  promptCell.appendChild(prompt);
  row.appendChild(promptCell);

  const actionCell = document.createElement("div");
  actionCell.className = "row-actions";
  actionCell.setAttribute("role", "cell");
  actionCell.appendChild(
    createRowAction("编辑", "edit", () => openEditor(preset)),
  );
  actionCell.appendChild(
    createRowAction("删除", "delete", () => requestDelete([preset.name]), true),
  );
  row.appendChild(actionCell);

  row.addEventListener("dragover", (event) => handleRowDragOver(event, row, preset.name));
  row.addEventListener("dragleave", () => clearRowDropState(row));
  row.addEventListener("drop", (event) => handleRowDrop(event, row, preset.name));
  return row;
}

function createRowAction(label, iconName, handler, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `row-action${danger ? " danger" : ""}`;
  button.title = label;
  button.setAttribute("aria-label", label);
  button.innerHTML = ACTION_ICONS[iconName];
  button.addEventListener("click", handler);
  return button;
}

function updateSelectVisible(visible) {
  const selectedVisible = visible.filter((preset) => selectedNames.has(preset.name));
  els.selectVisible.disabled = visible.length === 0 || operationPending;
  els.selectVisible.checked = visible.length > 0 && selectedVisible.length === visible.length;
  els.selectVisible.indeterminate =
    selectedVisible.length > 0 && selectedVisible.length < visible.length;
}

function renderBulkBar() {
  const selectedCount = selectedNames.size;
  els.bulkBar.hidden = selectedCount === 0;
  els.selectedCount.textContent = `已选择 ${selectedCount} 项`;
  renderCategoryOptions(els.bulkCategory, els.bulkCategory.value, {
    disableCategoryId: activeCategoryId,
  });
  els.bulkMoveButton.disabled = operationPending || !selectedCount || !els.bulkCategory.value;
  els.bulkDeleteButton.disabled = operationPending || !selectedCount;
}

function renderCategoryOptions(select, selectedId, { disableCategoryId = null } = {}) {
  const fragment = document.createDocumentFragment();
  let fallback = "";
  for (const category of managerState.categories) {
    const option = document.createElement("option");
    option.value = category.id;
    option.textContent = category.label;
    option.disabled = category.id === disableCategoryId;
    if (!option.disabled && !fallback) fallback = category.id;
    fragment.appendChild(option);
  }
  select.replaceChildren(fragment);
  const selectedOption = Array.from(select.options).find(
    (option) => option.value === selectedId && !option.disabled,
  );
  select.value = selectedOption ? selectedId : fallback;
}

function startDragging(event, presetName, row) {
  const category = getCategory();
  if (!category || operationPending) {
    event.preventDefault();
    return;
  }
  draggingNames = selectedNames.has(presetName)
    ? category.presets
        .map((preset) => preset.name)
        .filter((name) => selectedNames.has(name))
    : [presetName];
  event.dataTransfer.effectAllowed = "move";
  event.dataTransfer.setData("text/plain", draggingNames.join("\n"));
  row.classList.add("dragging");
  document.body.classList.add("is-dragging");
}

function finishDragging() {
  draggingNames = [];
  document.body.classList.remove("is-dragging");
  viewRoot.querySelectorAll(".dragging, .drop-before, .drop-after, .drop-target").forEach((node) => {
    node.classList.remove("dragging", "drop-before", "drop-after", "drop-target");
  });
}

function handleRowDragOver(event, row, targetName) {
  if (!draggingNames.length || draggingNames.includes(targetName)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = "move";
  const after = event.clientY > row.getBoundingClientRect().top + row.offsetHeight / 2;
  row.classList.toggle("drop-before", !after);
  row.classList.toggle("drop-after", after);
}

function clearRowDropState(row) {
  row.classList.remove("drop-before", "drop-after");
}

async function handleRowDrop(event, row, targetName) {
  event.preventDefault();
  if (!draggingNames.length || draggingNames.includes(targetName)) {
    clearRowDropState(row);
    return;
  }
  const after = row.classList.contains("drop-after");
  clearRowDropState(row);
  const category = getCategory();
  const movingSet = new Set(draggingNames);
  const remaining = category.presets
    .map((preset) => preset.name)
    .filter((name) => !movingSet.has(name));
  const targetIndex = remaining.indexOf(targetName);
  await movePresets(
    draggingNames,
    activeCategoryId,
    Math.max(0, targetIndex + (after ? 1 : 0)),
  );
}

async function movePresets(names, targetCategory, targetIndex = null) {
  if (!names.length || operationPending) return;
  setOperationPending(true);
  try {
    const payload = { names, target_category: targetCategory };
    if (Number.isInteger(targetIndex)) payload.target_index = targetIndex;
    const state = await api.move(payload);
    applyState(state);
    setSyncStatus("synced", "已同步");
    showToast(names.length > 1 ? `已移动 ${names.length} 个预设` : "预设已移动");
  } catch (error) {
    setSyncStatus("error", "保存失败");
    showToast(error.message || "预设移动失败", "error");
  } finally {
    setOperationPending(false);
    finishDragging();
    renderBulkBar();
  }
}

function openEditor(preset = null) {
  if (operationPending) return;
  editingOriginalName = preset?.name || null;
  els.editorTitle.textContent = preset ? "编辑预设" : "新建预设";
  els.presetName.value = preset?.name || "";
  els.presetPrompt.value = preset?.prompt || "";
  els.promptLength.textContent = String(els.presetPrompt.value.length);
  renderCategoryOptions(els.presetCategory, activeCategoryId);
  els.presetCategory.value = activeCategoryId;
  els.editorError.hidden = true;
  els.editorError.textContent = "";
  els.editorDialog.showModal();
  window.setTimeout(() => els.presetName.focus(), 0);
}

function closeEditor() {
  if (!operationPending && els.editorDialog.open) els.editorDialog.close();
}

function setEditorSaving(saving) {
  els.savePresetButton.disabled = saving;
  els.closeEditorButton.disabled = saving;
  els.cancelEditorButton.disabled = saving;
  els.savePresetButton.textContent = saving ? "正在保存…" : "保存预设";
}

async function saveEditor(event) {
  event.preventDefault();
  if (operationPending) return;
  const name = els.presetName.value.trim();
  const prompt = els.presetPrompt.value.trim();
  const category = els.presetCategory.value;
  if (!name || !prompt || !category) {
    els.editorError.textContent = "请填写名称、分类和提示词内容。";
    els.editorError.hidden = false;
    return;
  }

  setOperationPending(true);
  setEditorSaving(true);
  els.editorError.hidden = true;
  try {
    const state = editingOriginalName
      ? await api.update({ original_name: editingOriginalName, name, prompt, category })
      : await api.create({ name, prompt, category });
    applyState(state);
    els.editorDialog.close();
    setSyncStatus("synced", "已同步");
    showToast(editingOriginalName ? "预设修改已保存" : "预设已创建");
  } catch (error) {
    els.editorError.textContent = error.message || "预设保存失败，请重试。";
    els.editorError.hidden = false;
    setSyncStatus("error", "保存失败");
  } finally {
    setOperationPending(false);
    setEditorSaving(false);
    renderBulkBar();
  }
}

function requestDelete(names) {
  if (!names.length || operationPending) return;
  pendingDeleteNames = [...names];
  els.confirmMessage.textContent =
    names.length === 1
      ? `将永久删除“${names[0]}”及其分类引用。`
      : `将永久删除选中的 ${names.length} 个预设及其分类引用。`;
  els.confirmDialog.showModal();
  window.setTimeout(() => els.cancelDeleteButton.focus(), 0);
}

async function confirmDelete() {
  if (!pendingDeleteNames.length || operationPending) return;
  const names = [...pendingDeleteNames];
  setOperationPending(true);
  els.confirmDeleteButton.disabled = true;
  els.cancelDeleteButton.disabled = true;
  els.confirmDeleteButton.textContent = "正在删除…";
  try {
    const state = await api.delete({ names });
    applyState(state);
    els.confirmDialog.close();
    setSyncStatus("synced", "已同步");
    showToast(names.length > 1 ? `已删除 ${names.length} 个预设` : "预设已删除");
  } catch (error) {
    setSyncStatus("error", "删除失败");
    showToast(error.message || "预设删除失败", "error");
  } finally {
    pendingDeleteNames = [];
    setOperationPending(false);
    els.confirmDeleteButton.disabled = false;
    els.cancelDeleteButton.disabled = false;
    els.confirmDeleteButton.textContent = "确认删除";
    renderBulkBar();
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
    if (!initial) showToast("预设列表已刷新");
  } catch (error) {
    els.loadingState.hidden = true;
    setSyncStatus("error", "连接失败");
    if (initial || !managerState.categories.length) {
      renderFatalError(error.message || "预设列表加载失败");
    }
    showToast(error.message || "预设列表加载失败", "error");
  } finally {
    if (!initial) setOperationPending(false);
  }
}

function renderFatalError(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "empty-state error-state fatal-error";
  const heading = document.createElement("strong");
  heading.textContent = "无法加载预设管理器";
  const copy = document.createElement("p");
  copy.textContent = message;
  const retry = document.createElement("button");
  retry.type = "button";
  retry.className = "button button-secondary";
  retry.textContent = "重新加载";
  retry.addEventListener("click", () => {
    if (api) loadState();
    else window.location.reload();
  });
  wrapper.append(heading, copy, retry);
  els.presetList.replaceChildren(wrapper);
}

function bindEvents() {
  els.refreshButton.addEventListener("click", () => loadState());
  els.newPresetButton.addEventListener("click", () => openEditor());
  els.searchInput.addEventListener("input", () => {
    window.clearTimeout(searchRenderTimer);
    searchRenderTimer = window.setTimeout(() => {
      renderCategorySummary();
      renderPresetList();
      renderBulkBar();
    }, 100);
  });
  els.selectVisible.addEventListener("change", () => {
    for (const preset of getVisiblePresets()) {
      if (els.selectVisible.checked) selectedNames.add(preset.name);
      else selectedNames.delete(preset.name);
    }
    renderPresetList();
    renderBulkBar();
  });
  els.clearSelectionButton.addEventListener("click", () => {
    selectedNames.clear();
    renderPresetList();
    renderBulkBar();
  });
  els.bulkMoveButton.addEventListener("click", () => {
    movePresets([...selectedNames], els.bulkCategory.value);
  });
  els.bulkDeleteButton.addEventListener("click", () => requestDelete([...selectedNames]));
  els.closeEditorButton.addEventListener("click", closeEditor);
  els.cancelEditorButton.addEventListener("click", closeEditor);
  els.editorForm.addEventListener("submit", saveEditor);
  els.presetPrompt.addEventListener("input", () => {
    els.promptLength.textContent = String(els.presetPrompt.value.length);
  });
  els.cancelDeleteButton.addEventListener("click", () => {
    pendingDeleteNames = [];
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
    api = createPresetApi(bridge);
    await loadState({ initial: true });
  } catch (error) {
    els.loadingState.hidden = true;
    setSyncStatus("error", "连接失败");
    renderFatalError(error.message || "无法连接 AstrBot 插件页面 Bridge。");
  }
}

window.addEventListener("beforeunload", () => {
  if (typeof detachContextHandler === "function") detachContextHandler();
});

export function initializePresetManager(root) {
  if (!(root instanceof HTMLElement)) {
    throw new Error("预设管理视图挂载点不可用。");
  }
  viewRoot = root;
  els = collectElements(root);
  return bootstrap();
}
