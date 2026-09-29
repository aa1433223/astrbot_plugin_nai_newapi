/** Build the authenticated style-library API client for the shared management page. */
export function createStyleApi(bridge) {
  const apiGet = bridge?.apiGet?.bind(bridge);
  const apiPost = bridge?.apiPost?.bind(bridge);
  const upload = bridge?.upload?.bind(bridge);

  if (!apiGet || !apiPost || !upload) {
    throw new Error("AstrBot 插件页面 Bridge 不可用。");
  }

  return {
    loadState() {
      return apiGet("styles/state");
    },
    create(payload) {
      return apiPost("styles/create", payload);
    },
    update(payload) {
      return apiPost("styles/update", payload);
    },
    delete(payload) {
      return apiPost("styles/delete", payload);
    },
    uploadPreview(file) {
      return upload("styles/upload", file);
    },
  };
}
