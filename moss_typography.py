"""Local font rendering for exact text in MOSS layout boxes."""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont


AUTO_FONT = "按描述自动匹配"
SYSTEM_FONTS = {"楷体": "simkai.ttf", "宋体": "simsun.ttc", "黑体": "simhei.ttf", "微软雅黑": "msyh.ttc"}
COLORS = {"绿色": "#20843d", "黄色": "#ffd633", "红色": "#d62c27", "蓝色": "#2863c9",
          "白色": "#ffffff", "黑色": "#161616", "淡蓝色": "#82c9ed", "橙色": "#f39029", "紫色": "#8a43b7"}


def font_files(models_dir, node_dir):
    fonts = {}
    system = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for label, filename in SYSTEM_FONTS.items():
        path = system / filename
        if path.is_file():
            fonts[label] = str(path)
    for label, root in [("模型字体", Path(models_dir)/"fonts"), ("节点字体", Path(node_dir)/"fonts")]:
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".ttf", ".otf", ".ttc"}:
                    fonts[f"{label}/{path.relative_to(root).as_posix()}"] = str(path)
    return fonts


def resolve_font(choice, description, fonts):
    if choice != AUTO_FONT:
        if choice not in fonts:
            raise ValueError("所选字体不存在，请重新选择字体文件。")
        return fonts[choice]
    requested = next((name for name in ["微软雅黑", "楷体", "楷书", "宋体", "黑体"] if name in description), None)
    if requested == "楷书":
        requested = "楷体"
    if requested:
        if requested not in fonts:
            raise ValueError(f"未找到{requested}字体，请在字体选择中指定本地字体，或放入 models/fonts。")
        return fonts[requested]
    if not fonts:
        raise ValueError("精确排字需要本地字体，请把中文 TTF/OTF/TTC 字体放入 models/fonts。")
    return fonts.get("宋体", next(iter(fonts.values())))


def text_color(box):
    palette = box.get("palette") or []
    candidates = list(palette) + [color for color in COLORS if color in box.get("desc", "")]
    for color in candidates:
        try:
            return ImageColor.getrgb(COLORS.get(str(color), str(color)))[:3]
        except ValueError:
            continue
    return (25, 25, 25)


def glyph_layout(text, font, vertical):
    lines = text.split("\n")
    heights = [font.getbbox(ch)[3]-font.getbbox(ch)[1] for ch in text if ch != "\n"]
    cell = max(font.size, max(heights, default=font.size))
    if vertical:
        widths = [max((font.getbbox(ch)[2]-font.getbbox(ch)[0] for ch in line), default=cell) for line in lines]
        gap = max(1, round(cell*.15))
        width = sum(widths)+gap*(len(lines)-1)
        height = max((len(line)*cell for line in lines), default=0)
        glyphs, right = [], width
        for line, col_width in zip(lines, widths):
            right -= col_width
            for i, ch in enumerate(line):
                l, t, r, b = font.getbbox(ch)
                glyphs.append((ch, right+(col_width-r+l)/2-l, i*cell+(cell-b+t)/2-t))
            right -= gap
        return width, height, glyphs
    widths = [font.getlength(line) for line in lines]
    width, glyphs = max(widths, default=0), []
    for i, (line, line_width) in enumerate(zip(lines, widths)):
        _, top, _, _ = font.getbbox(line or " ")
        glyphs.append((line, (width-line_width)/2, i*cell-top))
    return width, len(lines)*cell, glyphs


def render_text(image, raw_layout, fonts, choice=AUTO_FONT, clear=True):
    """Return an RGB PIL image. Only explicit text fields are drawn, never descriptions."""
    boxes = json.loads(raw_layout) if isinstance(raw_layout, str) else raw_layout
    result = image.convert("RGB").copy()
    for box in boxes:
        if box.get("type") != "text":
            continue
        text = str(box.get("text", ""))
        if not text.strip():
            continue
        x0, y0 = round(box["x"]*result.width), round(box["y"]*result.height)
        x1, y1 = round((box["x"]+box["w"])*result.width), round((box["y"]+box["h"])*result.height)
        width, height = max(1, x1-x0), max(1, y1-y0)
        region = result.crop((x0, y0, x1, y1))
        if clear:
            pixels = np.asarray(region)
            edge = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
            background = tuple(int(v) for v in np.median(edge, axis=0))
            region = Image.new("RGB", (width, height), background)
        description = str(box.get("desc", ""))
        path = resolve_font(choice, description, fonts)
        vertical = box.get("direction") == "vertical" or ("direction" not in box and "竖排" in description)
        margin = max(2, round(min(width, height)*.04))
        low, high, best = 1, max(width, height), 1
        while low <= high:
            size = (low+high)//2
            font = ImageFont.truetype(path, size)
            tw, th, _ = glyph_layout(text, font, vertical)
            if tw <= width-2*margin and th <= height-2*margin:
                best, low = size, size+1
            else:
                high = size-1
        font = ImageFont.truetype(path, best)
        tw, th, glyphs = glyph_layout(text, font, vertical)
        draw = ImageDraw.Draw(region)
        for chars, x, y in glyphs:
            draw.text(((width-tw)/2+x, (height-th)/2+y), chars, font=font, fill=text_color(box))
        result.paste(region, (x0, y0))
    return result
