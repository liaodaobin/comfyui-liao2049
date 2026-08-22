import { app } from "../../../scripts/app.js";

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function sanitizeLegacyInputs(node) {
    for (const name of ["images", "parameters"]) {
        const index = (node.inputs || []).findIndex((input) => input.name === name);
        if (index >= 0) {
            node.removeInput(index);
        }
    }
}

function normalizeNumberWidget(node, name, fallback, min = null, max = null) {
    const widget = findWidget(node, name);
    if (!widget) return;
    let value = Number(widget.value);
    if (!Number.isFinite(value)) value = fallback;
    if (min != null) value = Math.max(min, value);
    if (max != null) value = Math.min(max, value);
    if (widget.value !== value) {
        widget.value = value;
        widget.callback?.(value);
    }
}

function sanitizeLegacyWidgetValues(node) {
    normalizeNumberWidget(node, "max_frames", 24, 2, 1024);
    normalizeNumberWidget(node, "max_size", 768, 128, 16384);
    normalizeNumberWidget(node, "seed", 0, 0);
    normalizeStyleWidget(node);
    normalizeImageWashDefaults(node);
    restoreImageWidget(node);
}

function normalizeStyleWidget(node) {
    const widget = findWidget(node, "style");
    if (!widget) return;

    const aliases = {
        "": "文生图",
        "false": "文生图",
        "text": "文生图",
        "text-to-image": "文生图",
        "t2i": "文生图",
        "true": "风格参考",
        "style": "风格参考",
        "style_reference": "风格参考",
        "style reference": "风格参考",
        "image wash": "洗图",
        "wash": "洗图",
        "rewrite image": "洗图",
    };

    let value = widget.value;
    if (typeof value === "boolean") {
        value = value ? "风格参考" : "文生图";
    } else {
        const text = String(value ?? "").trim();
        value = aliases[text.toLowerCase()] || text;
    }

    if (!["文生图", "风格参考", "洗图"].includes(value)) {
        value = "文生图";
    }

    if (widget.value !== value) {
        widget.value = value;
        widget.callback?.(value);
    }
}

function normalizeImageWashDefaults(node) {
    const style = findWidget(node, "style");
    const maxSize = findWidget(node, "max_size");
    if (!style || !maxSize || style.value !== "洗图") return;

    const value = Number(maxSize.value);
    if (!Number.isFinite(value) || value < 768) {
        maxSize.value = 768;
        maxSize.callback?.(768);
    }
}

function restoreImageWidget(node) {
    const widget = findWidget(node, "style_image");
    if (!widget) return;

    widget.hidden = false;
    widget.options = widget.options || {};
    widget.options.hidden = false;
    if (widget._wenwuOriginalComputeSize) {
        widget.computeSize = widget._wenwuOriginalComputeSize;
    }
    if (widget.element) {
        widget.element.style.display = "";
    }
}

app.registerExtension({
    name: "Liao2049.Krea2PromptInstruct",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WenWuKrea2PromptInstruct") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            sanitizeLegacyInputs(this);
            sanitizeLegacyWidgetValues(this);

            const style = findWidget(this, "style");
            if (style && !style._wenwuKrea2Wrapped) {
                const originalCallback = style.callback;
                style.callback = (...args) => {
                    originalCallback?.apply(style, args);
                    normalizeImageWashDefaults(this);
                };
                style._wenwuKrea2Wrapped = true;
            }
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            sanitizeLegacyInputs(this);
            sanitizeLegacyWidgetValues(this);
        };
    },
});
