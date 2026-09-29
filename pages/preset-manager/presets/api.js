/** Build the preset API client for the shared NovelAI management page. */
export function createPresetApi(bridge) {
  const apiGet = bridge?.apiGet?.bind(bridge);
  const apiPost = bridge?.apiPost?.bind(bridge);

  if (!apiGet || !apiPost) {
    throw new Error("AstrBot 插件页面 Bridge 不可用。");
  }

  return {
    loadState() {
      return apiGet("presets/state");
    },
    create(payload) {
      return apiPost("presets/create", payload);
    },
    update(payload) {
      return apiPost("presets/update", payload);
    },
    move(payload) {
      return apiPost("presets/move", payload);
    },
    delete(payload) {
      return apiPost("presets/delete", payload);
    },
  };
}
