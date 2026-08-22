import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const PLANNER_MIN_WIDTH = 500;
const PLANNER_MIN_HEIGHT = 500;
const PLANNER_MAX_HEIGHT = 620;
const DESCRIPTION_HEIGHT = 54;
const APPROVED_MIN_HEIGHT = 240;
const APPROVED_MAX_HEIGHT = 355;

function findWidget(node, name) {
    return (node.widgets || []).find((widget) => widget.name === name);
}

function widgetValue(node, name, fallback = null) {
    const widget = findWidget(node, name);
    return widget?.value ?? fallback;
}

function isLegacyBooleanText(value) {
    return typeof value === "string" && ["true", "false"].includes(value.trim().toLowerCase());
}

function linkedNode(node, inputName) {
    const input = (node.inputs || []).find((item) => item.name === inputName);
    const link = input?.link != null ? app.graph.links?.[input.link] : null;
    return link ? app.graph.getNodeById(link.origin_id) : null;
}

function llamaConfigFromLoader(node) {
    return {
        model: widgetValue(node, "model"),
        mmproj: widgetValue(node, "mmproj", "None"),
        chat_handler: widgetValue(node, "chat_handler", "None"),
        n_ctx: Number(widgetValue(node, "n_ctx", 8192)),
        vram_limit: Number(widgetValue(node, "vram_limit", -1)),
        image_min_tokens: Number(widgetValue(node, "image_min_tokens", 0)),
        image_max_tokens: Number(widgetValue(node, "image_max_tokens", 0)),
    };
}

function setTextareaHeight(widget, height) {
    if (!widget?.inputEl) return;
    const clamped = Math.min(APPROVED_MAX_HEIGHT, Math.max(APPROVED_MIN_HEIGHT, height));
    widget.inputEl.style.height = `${clamped}px`;
    widget.inputEl.style.minHeight = `${clamped}px`;
    widget.inputEl.style.maxHeight = `${clamped}px`;
}

function setWidgetHeight(widget, height) {
    if (!widget) return;
    widget.computeSize = () => [0, height];
}

async function previewPromptOnly(node) {
    const loader = linkedNode(node, "llama_model");
    if (!loader) {
        throw new Error("llama_model input is not connected.");
    }
    const sourceNode = linkedNode(node, "source_lyrics");
    const sourceLyrics = sourceNode?.aceSourceLyrics
        || sourceNode?.properties?.aceSourceLyrics
        || "";
    if (sourceNode && !sourceLyrics.trim()) {
        throw new Error("请先运行一次工作流完成源歌曲歌词识别，再点击预览/刷新。");
    }
    const response = await fetch("/liao2049/ace_step_llm_planner/preview", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            llama_model: llamaConfigFromLoader(loader),
            description: widgetValue(node, "description", ""),
            duration: widgetValue(node, "duration", 120.0),
            language: widgetValue(node, "language", "auto"),
            seed: Math.floor(Date.now() % 0xFFFFFFFF),
            source_lyrics: sourceLyrics,
        }),
    });
    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || `Preview failed: ${response.status}`);
    }
    return data.text || "";
}

function clampPlannerSize(node) {
    const width = Math.max(node.size?.[0] || 0, PLANNER_MIN_WIDTH);
    const height = Math.min(PLANNER_MAX_HEIGHT, Math.max(node.size?.[1] || 0, PLANNER_MIN_HEIGHT));
    if (node.size?.[0] !== width || node.size?.[1] !== height) {
        node.setSize([width, height]);
    }
}

function applyPlannerLayout(node) {
    const description = findWidget(node, "description");
    const approved = findWidget(node, "approved_plan");
    const duration = findWidget(node, "duration");
    const language = findWidget(node, "language");
    const editMode = findWidget(node, "use_confirmed_prompt");

    if (duration) {
        duration.label = "\u65f6\u957f";
    }
    if (language) {
        language.label = "\u8bed\u8a00";
    }
    if (editMode) {
        editMode.label = "\u7f16\u8f91";
    }

    if (description?.inputEl) {
        description.inputEl.style.height = `${DESCRIPTION_HEIGHT}px`;
        description.inputEl.style.minHeight = `${DESCRIPTION_HEIGHT}px`;
        description.inputEl.style.maxHeight = `${DESCRIPTION_HEIGHT}px`;
        description.inputEl.style.resize = "none";
        description.inputEl.style.overflowY = "auto";
        description.inputEl.placeholder = "\u8f93\u5165\u81ea\u7136\u8bed\u8a00\uff0c\u4f8b\u5982\uff1a\u5199\u4e00\u9996\u4e2d\u6587\u56fd\u98ce\u53d9\u4e8b\u6b4c\uff0c\u5973\u58f0\uff0c\u60c5\u7eea\u4ece\u9690\u5fcd\u5230\u7206\u53d1\u3002";
        setWidgetHeight(description, DESCRIPTION_HEIGHT + 8);
    }

    if (approved?.inputEl) {
        approved.label = "\u786e\u8ba4\u7a3f";
        if (isLegacyBooleanText(approved.value)) {
            approved.value = "";
            approved.callback?.("");
        }
        const editable = Boolean(editMode?.value);
        approved.inputEl.readOnly = !editable;
        approved.inputEl.placeholder = editable
            ? "\u7f16\u8f91\u6a21\u5f0f\u5df2\u5f00\u542f\uff1a\u6700\u7ec8\u8fd0\u884c\u4f1a\u4f7f\u7528\u8fd9\u91cc\u7684\u5185\u5bb9\u3002"
            : "\u53ea\u8bfb\u9884\u89c8\uff1a\u6700\u7ec8\u8fd0\u884c\u4f1a\u5ffd\u7565\u8fd9\u91cc\uff0c\u6309\u81ea\u7136\u8bed\u8a00\u91cd\u65b0\u751f\u6210\u3002";
        approved.inputEl.style.opacity = editable ? "1" : "0.58";
        approved.inputEl.style.backgroundColor = editable ? "#1f1f1f" : "#303030";
        approved.inputEl.style.color = editable ? "#f2f2f2" : "#b8b8b8";
        approved.inputEl.style.resize = "none";
        approved.inputEl.style.overflowY = "auto";
        const reservedHeight = 280;
        const nodeHeight = Math.min(node.size?.[1] || PLANNER_MIN_HEIGHT, PLANNER_MAX_HEIGHT);
        const approvedHeight = Math.min(APPROVED_MAX_HEIGHT, Math.max(APPROVED_MIN_HEIGHT, nodeHeight - reservedHeight));
        setTextareaHeight(approved, approvedHeight);
        setWidgetHeight(approved, approvedHeight + 10);
    }
}

app.registerExtension({
    name: "AceStepLLM.PlanPreview",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AceStepSourceLyrics") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const displayed = message?.text?.[0] ?? "";
                const lyrics = String(displayed).replace(/^\[[^\]\r\n]+\]\s*\r?\n?/, "").trim();
                this.aceSourceLyrics = lyrics;
                this.properties = this.properties || {};
                this.properties.aceSourceLyrics = lyrics;
            };
            return;
        }

        const isPreviewOnly = nodeData.name === "AceStepSongPlanPreview";
        const isPlanner = nodeData.name === "AceStepLLMSongPlanner";
        if (!["AceStepSongPlanPreview", "AceStepLLMSongPlanner"].includes(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            if (isPlanner) {
                this.acePreviewWidget = findWidget(this, "approved_plan");
                this.aceEditModeWidget = findWidget(this, "use_confirmed_prompt");
                clampPlannerSize(this);
                applyPlannerLayout(this);

                const previewButton = this.addWidget("button", "\u9884\u89c8/\u5237\u65b0\u63d0\u793a\u8bcd", null, async () => {
                    if (this.acePreviewBusy) return;
                    this.acePreviewBusy = true;
                    previewButton.name = "\u751f\u6210\u4e2d...";
                    previewButton.disabled = true;
                    app.graph.setDirtyCanvas(true, true);
                    if (this.acePreviewWidget) {
                        this.acePreviewWidget.value = "";
                        this.acePreviewWidget.callback?.("");
                    }
                    try {
                        const text = await previewPromptOnly(this);
                        if (this.acePreviewWidget) {
                            this.acePreviewWidget.value = text;
                            this.acePreviewWidget.callback?.(text);
                        }
                        app.graph.setDirtyCanvas(true, true);
                    } catch (error) {
                        if (this.acePreviewWidget) {
                            this.acePreviewWidget.inputEl.placeholder = `预览失败：${error.message || error}`;
                        }
                        console.error(error);
                    } finally {
                        this.acePreviewBusy = false;
                        previewButton.name = "\u9884\u89c8/\u5237\u65b0\u63d0\u793a\u8bcd";
                        previewButton.disabled = false;
                        app.graph.setDirtyCanvas(true, true);
                    }
                });
                previewButton.serialize = false;

                const clearButton = this.addWidget("button", "\u6e05\u7a7a\u786e\u8ba4\u7a3f", null, () => {
                    if (this.acePreviewBusy) return;
                    const editMode = findWidget(this, "use_confirmed_prompt");
                    if (editMode) {
                        editMode.value = false;
                        editMode.callback?.(false);
                    }
                    if (this.acePreviewWidget) {
                        this.acePreviewWidget.value = "";
                        this.acePreviewWidget.callback?.("");
                    }
                    applyPlannerLayout(this);
                    app.graph.setDirtyCanvas(true, true);
                });
                clearButton.serialize = false;

                previewButton.computeSize = () => [0, 26];
                clearButton.computeSize = () => [0, 26];
                if (this.aceEditModeWidget) {
                    const onEditModeChanged = this.aceEditModeWidget.callback;
                    this.aceEditModeWidget.callback = (value) => {
                        onEditModeChanged?.call(this.aceEditModeWidget, value);
                        applyPlannerLayout(this);
                        app.graph.setDirtyCanvas(true, true);
                    };
                }
                clampPlannerSize(this);
                applyPlannerLayout(this);
            } else {
                this.acePreviewWidget = ComfyWidgets["STRING"](
                    this,
                    "preview_output",
                    ["STRING", { multiline: true }],
                    app
                ).widget;
                this.acePreviewWidget.inputEl.readOnly = true;
                this.acePreviewWidget.inputEl.placeholder = "Run the workflow to preview the generated song plan here.";
                this.acePreviewWidget.serializeValue = async () => "";
                this.setSize(isPreviewOnly ? [520, 460] : [Math.max(this.size[0], 520), Math.max(this.size[1], 620)]);
            }
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            if (isPlanner) {
                clampPlannerSize(this);
                applyPlannerLayout(this);
            }
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const text = message?.text?.[0] ?? "";
            if (this.acePreviewWidget) {
                this.acePreviewWidget.value = text;
                this.acePreviewWidget.callback?.(text);
                requestAnimationFrame(() => {
                    if (isPlanner) applyPlannerLayout(this);
                    app.graph.setDirtyCanvas(true, false);
                });
            }
        };
    },
});

