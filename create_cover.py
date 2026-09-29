# -*- coding: utf-8 -*-
"""Обложки для ролика: 16:9 (YouTube) и 9:16 (Shorts/Клипы) из одного кадра.

Почему переписано:
  - кадр клипа 640x640 мал для превью и при загрузке кропается платформой;
  - старая версия ставила каждое слово на отдельную строку по центру кадра,
    без полей -> длинный заголовок физически не влезал и резался за кадром.

Что делает сейчас:
  1. Апскейл кадра до целевого размера с cover-кропом под нужное
     соотношение сторон (лицо не срезается: вертикальный якорь сверху).
  2. Текст верстается ТОЛЬКО внутри safe-zone (поля 8% с каждой стороны),
     переносом по ширине (не по одному слову), с авто-подбором кегля —
     вылезти за кадр физически невозможно.
  3. Текст в нижней (или верхней) трети, а не поверх лица, на
     полупрозрачной плашке вместо толстой обводки.
  4. Номер части — маленькой плашкой в углу (part_label).

Использование:
    from create_cover import create_covers
    create_covers("cover_frame.png", "ПАРИ НА ГОД", out_dir, part_label="Часть 5")
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Целевые размеры обложек. Делаем сразу оба варианта — лишний удаляется руками.
ASPECTS = {
    "16x9": (1280, 720),   # YouTube / VK видео
    "9x16": (720, 1280),   # Shorts / TikTok / VK Клипы
}

# Вертикальный якорь кропа: 0.0 = верх кадра, 0.5 = центр. Лица в аниме-кадре
# обычно в верхней половине, поэтому при кропе 16:9 режем снизу.
CROP_ANCHOR_Y = 0.32

FONT_CANDIDATES = [
    r"C:\Users\Loopy\Desktop\comfyui\ComfyUI\output\AI_VIDEO\Montserrat-VariableFont_wght.ttf",
    os.path.join(BASE_DIR, "Montserrat-VariableFont_wght.ttf"),
    "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
]

SAFE_MARGIN = 0.08          # поля safe-zone (доля от стороны)
MAX_TEXT_BLOCK_H = 0.34     # текстовый блок не выше 34% высоты кадра
MAX_LINES = 3
LINE_SPACING = 0.18         # межстрочный интервал (доля от кегля)
PLATE_ALPHA = 150           # прозрачность плашки под текстом (0..255)


def _resolve_font_path(font_path: str | None) -> str | None:
    for p in ([font_path] if font_path else []) + FONT_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


def _load_font(font_path: str | None, size: int) -> ImageFont.FreeTypeFont:
    p = _resolve_font_path(font_path)
    if not p:
        return ImageFont.load_default()
    font = ImageFont.truetype(p, size)
    # Montserrat — variable font: без выбора инстанса получим Regular (тонкий).
    for name in (b"ExtraBold", b"Bold", b"SemiBold"):
        try:
            font.set_variation_by_name(name)
            break
        except Exception:
            continue
    return font


def _fit_frame(img: Image.Image, target: tuple[int, int]) -> Image.Image:
    """Cover-кроп + апскейл кадра под целевое соотношение сторон."""
    tw, th = target
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    new = (max(tw, int(round(sw * scale))), max(th, int(round(sh * scale))))
    img = img.resize(new, Image.LANCZOS)
    nw, nh = img.size
    left = int(round((nw - tw) / 2))
    top = int(round((nh - th) * CROP_ANCHOR_Y))
    return img.crop((left, top, left + tw, top + th))


def _wrap(draw, text: str, font, max_w: int) -> list[str] | None:
    """Жадный перенос по ПИКСЕЛЬНОЙ ширине. None — слово шире строки."""
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    cur = ""
    for w in words:
        if draw.textlength(w, font=font) > max_w:
            return None
        probe = f"{cur} {w}".strip()
        if draw.textlength(probe, font=font) <= max_w:
            cur = probe
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _block_metrics(draw, lines: list[str], font) -> tuple[int, int, int]:
    """(ширина блока, высота блока, шаг строки)."""
    asc, desc = font.getmetrics()
    line_h = asc + desc
    step = int(round(line_h * (1 + LINE_SPACING)))
    width = max((int(draw.textlength(l, font=font)) for l in lines), default=0)
    height = step * (len(lines) - 1) + line_h if lines else 0
    return width, height, step


def _fit_text(
    draw, text: str, font_path: str | None, max_w: int, max_h: int,
    size_hi: int, size_lo: int = 18, max_lines: int = MAX_LINES,
):
    """Подбор максимального кегля, при котором текст влезает в safe-zone."""
    best = None
    for size in range(size_hi, size_lo - 1, -2):
        font = _load_font(font_path, size)
        lines = _wrap(draw, text, font, max_w)
        if not lines or len(lines) > max_lines:
            continue
        bw, bh, step = _block_metrics(draw, lines, font)
        if bw <= max_w and bh <= max_h:
            best = (font, lines, bw, bh, step)
            break
    if best is None:
        font = _load_font(font_path, size_lo)
        lines = _wrap(draw, text, font, max_w) or [text]
        lines = lines[:max_lines]
        bw, bh, step = _block_metrics(draw, lines, font)
        best = (font, lines, bw, bh, step)
    return best


def _draw_plate(img: Image.Image, box, radius: int, alpha: int = PLATE_ALPHA):
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(box, radius=radius, fill=(0, 0, 0, alpha))
    return Image.alpha_composite(img.convert("RGBA"), overlay)


def create_cover(
    input_image_path: str,
    text: str,
    output_path: str,
    font_path: str | None = None,
    font_size: int | None = None,
    aspect: str = "16x9",
    part_label: str | None = None,
    text_position: str = "bottom",
    uppercase: bool = True,
    accent_color: tuple[int, int, int] = (255, 255, 255),
) -> str:
    """Одна обложка нужного формата. font_size — ПОТОЛОК кегля, не фикс."""
    target = ASPECTS.get(aspect)
    if target is None:
        raise ValueError(f"aspect {aspect!r} не из {sorted(ASPECTS)}")

    img = _fit_frame(Image.open(input_image_path).convert("RGB"), target)
    W, H = img.size
    mx, my = int(W * SAFE_MARGIN), int(H * SAFE_MARGIN)
    safe_w, safe_h = W - 2 * mx, int(H * MAX_TEXT_BLOCK_H)

    text = (text or "").strip()
    if uppercase:
        text = text.upper()

    draw = ImageDraw.Draw(img)
    size_hi = font_size or int(min(W, H) * 0.17)
    font, lines, bw, bh, step = _fit_text(draw, text, font_path, safe_w, safe_h, size_hi)

    pad = int(font.size * 0.35)
    if text_position == "top":
        block_top = my
    else:
        block_top = H - my - bh
    block_left = (W - bw) // 2

    img = _draw_plate(
        img,
        (block_left - pad, block_top - pad, block_left + bw + pad, block_top + bh + pad),
        radius=int(pad * 0.9),
    )
    draw = ImageDraw.Draw(img)

    stroke = max(2, font.size // 22)
    y = block_top
    for line in lines:
        lw = draw.textlength(line, font=font)
        draw.text(
            ((W - lw) / 2, y), line, font=font, fill=accent_color,
            stroke_width=stroke, stroke_fill=(0, 0, 0),
        )
        y += step

    # номер части — мелко, в углу напротив текстового блока
    if part_label:
        pf = _load_font(font_path, max(16, int(min(W, H) * 0.045)))
        pl = part_label.strip()
        pw = int(draw.textlength(pl, font=pf))
        asc, desc = pf.getmetrics()
        ph = asc + desc
        ppad = int(pf.size * 0.35)
        px = W - mx - pw
        py = my if text_position != "top" else H - my - ph
        img = _draw_plate(
            img, (px - ppad, py - ppad, px + pw + ppad, py + ph + ppad),
            radius=int(ppad * 0.8), alpha=120,
        )
        draw = ImageDraw.Draw(img)
        draw.text((px, py), pl, font=pf, fill=(255, 255, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    img.convert("RGB").save(output_path, quality=95)
    print(f"Обложка {aspect} сохранена: {output_path} "
          f"({W}x{H}, кегль {font.size}, строк {len(lines)})")
    return output_path


def create_covers(
    input_image_path: str,
    text: str,
    output_dir: str,
    base_name: str = "result_cover",
    **kwargs,
) -> dict[str, str]:
    """Сразу оба формата из одного кадра: 16x9 и 9x16."""
    out: dict[str, str] = {}
    for aspect in ASPECTS:
        out[aspect] = create_cover(
            input_image_path=input_image_path,
            text=text,
            output_path=os.path.join(output_dir, f"{base_name}_{aspect}.jpg"),
            aspect=aspect,
            **kwargs,
        )
    return out


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "cover_frame.png"
    txt = sys.argv[2] if len(sys.argv) > 2 else "ПАРИ НА ГОД"
    create_covers(src, txt, ".", part_label="Часть 5")
