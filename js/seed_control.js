export function seedControl(node, field) {
  node.properties ??= {};
  if (typeof node.properties.liao_fixed_seed !== "boolean") node.properties.liao_fixed_seed = false;
  const seed = node.widgets?.find(widget => widget.name === "随机种子");
  const input = field.querySelector("input");
  const box = document.createElement("div");
  box.style.cssText = "position:relative;min-width:0";
  field.style.display = "block";
  if (input) input.style.paddingRight = "36px";
  box.append(field);
  const label = document.createElement("label");
  label.style.cssText = "position:absolute;right:9px;bottom:8px;display:flex;align-items:center;justify-content:center;cursor:pointer;margin:0";
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.style.cssText = "width:16px!important;height:16px;margin:0;accent-color:#16a6a8;cursor:pointer";
  toggle.setAttribute("aria-label", "固定种子");
  label.append(toggle);
  label.title = "固定种子：开启后保持当前种子；关闭时每次生成随机种子。";
  toggle.title = label.title;
  box.append(label);
  const sync = () => {
    toggle.checked = node.properties.liao_fixed_seed === true;
    // This switch owns the policy; the native after-generation control must not change a locked seed.
    const control = node.widgets?.find(widget => widget.name === "control_after_generate");
    if (control) control.value = "fixed";
    if (seed && input) input.value = String(seed.value);
  };
  toggle.addEventListener("change", () => {
    node.properties.liao_fixed_seed = toggle.checked;
    sync();
    node.graph?.setDirtyCanvas?.(true, true);
  });
  if (seed) {
    const beforeQueued = seed.beforeQueued;
    seed.beforeQueued = function (...args) {
      const result = beforeQueued?.apply(this, args);
      if (!node.properties.liao_fixed_seed) {
        let value;
        do {
          const words = new Uint32Array(2);
          crypto.getRandomValues(words);
          value = (words[0] & 0x1fffff) * 4294967296 + words[1];
        } while (value === seed.value);
        seed.value = value;
      }
      sync();
      return result;
    };
  }
  const configured = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = configured?.apply(this, args);
    sync();
    return result;
  };
  sync();
  return box;
}
