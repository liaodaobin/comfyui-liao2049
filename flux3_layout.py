"""Flux3 layout captions with coordinates owned by the user's canvas."""
import json
import math


def layout_elements(raw):
    try:
        elements = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise ValueError("构图数据无效，请重新打开构图画布。") from exc
    if not isinstance(elements, list) or not 1 <= len(elements) <= 40:
        raise ValueError("请在构图画布上添加 1–40 个物体或文字框。")
    rows = []
    for i, item in enumerate(elements, 1):
        if not isinstance(item, dict):
            raise ValueError(f"构图第 {i} 项不是有效元素。")
        try:
            x, y, w, h = (float(item[k]) for k in ("x", "y", "w", "h"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"构图第 {i} 项的坐标无效。") from exc
        if not all(math.isfinite(v) for v in (x, y, w, h)) or min(x, y) < 0 or min(w, h) <= 0 or x+w > 1.00001 or y+h > 1.00001:
            raise ValueError(f"构图第 {i} 项超出画布，请调整位置或大小。")
        text = str(item.get("text", "")).strip()
        desc = str(item.get("desc", "")).strip()
        is_text = item.get("type") == "text"
        if not (text if is_text else desc):
            raise ValueError(f"请填写第 {i} 个{'文字框的文字' if is_text else '物体框的描述'}。")
        coords = [round(y*1000), round(x*1000), round((y+h)*1000), round((x+w)*1000)]
        if coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError(f"构图第 {i} 项太小，请放大方框。")
        if is_text:
            vertical = item.get("direction") == "vertical" or ("direction" not in item and "竖排" in desc)
            orientation = "vertical text, top to bottom" if vertical else "horizontal text"
            description = f'Text reading {json.dumps(text, ensure_ascii=False)}, {orientation}.'
            if desc:
                description += " " + desc
        else:
            description = desc
        rows.append({"id": f"Zh_Text_{i}" if is_text else f"object_{i}", "bbox": coords, "desc": description})
    return rows


def layout_prompt(prompt, raw, expanded=""):
    rows = layout_elements(raw)
    caption = str(prompt).strip()
    if expanded:
        # The LLM may enrich object descriptions, but never owns boxes or text.
        try:
            body = expanded.strip()
            start, end = body.find("{"), body.rfind("}")
            if start < 0 or end < start:
                raise ValueError("missing JSON")
            data = json.loads(body[start:end+1])
            descriptions = data.get("descriptions", {})
            if not isinstance(descriptions, dict):
                raise ValueError("invalid descriptions")
            for row in rows:
                description = descriptions.get(row["id"])
                if row["id"].startswith("object_") and isinstance(description, str) and description.strip():
                    row["desc"] += " " + description.strip()
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("构图智能扩写返回格式无效；请重试或关闭智能扩写，原构图不会丢失。") from exc
    citations = " ".join(f"{'A text block' if r['id'].startswith('Zh_Text_') else 'A subject'} <{r['id']}>." for r in rows)
    return f"{caption} {citations} " + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


LAYOUT_EXPANSION_SYSTEM = '''Enrich the object descriptions in the supplied image layout.
Return exactly one JSON object: {"descriptions":{"object_1":"English visual description"}}.
Use only the supplied object IDs. Keep the user's identities, counts, attributes and style.
Do not add subjects. Do not output or change coordinates, text blocks, or text wording.
Describe appearance, materials, clothing, pose and expression without spatial instructions.
No markdown or commentary.'''
