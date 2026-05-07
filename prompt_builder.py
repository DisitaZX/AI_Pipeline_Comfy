"""Детерминистический prompt-builder для Qwen Image Edit 2511 (без LLM в loop'е).

Эмпирически подтверждено на сборке justus16:
  * `Picture N (descriptive alias)` сильно повышает adherence к каждой entity
    (модель перестаёт "терять" второстепенных персонажей, см. кейс Picture 2 =
    девушка в side-profile сцене).
  * Развёрнутые CHAR-anchored camera-фразы (где стоит камера, на какой высоте,
    куда направлена) надёжно переопределяют композицию reference'а — короткое
    "side profile medium shot." теряется на vision-токенах.
  * Перестановка image1/image2/image3 в энкодере НЕ помогает заметно — Qwen-VL
    в Qwen Image Edit Plus не сильно чувствителен к позиции vision-токена,
    но очень чувствителен к количеству и качеству TEXT-токенов вокруг
    каждого упоминания Picture N.

Этот модуль:
  - Развёртывает короткие camera-presets в CHAR-anchored фразы (`CAMERA_HINTS_RICH`).
  - Для каждой entity извлекает short alias из её `base_prompt` (или берёт
    из явного override `bp["short_alias"]`).
  - Подменяет `[entity]` в `scene["image_prompt"]` на `Picture N (alias)`,
    дедупит entity, лимитит до 3 (hard cap у `TextEncodeQwenImageEditPlus`),
    режет неизвестные `[bracketed]` токены.
  - Аппендит общий style stack для Pixar 3D CGI Toon look.

Никаких сетевых вызовов, никаких LLM. Полностью детерминистический.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ============================================================ camera presets

# Развёрнутые camera-only фразы для каждого пресета из CAMERA_PRESETS
# (chunk_prompts_2.py:CAMERA_PRESETS).
#
# ВАЖНО: фразы НЕ упоминают что находится в кадре (никаких "subjects", "both",
# "viewer", "the subject", "two-shot", "their faces", "head and shoulders" и
# т.п.). Это сделано намеренно — иначе Qwen Edit 2511 трактует эти токены как
# content-signal и дорисовывает лишних людей/объектов в кадр (в т.ч. на сценах
# где должен быть один объект). Каждая фраза говорит ТОЛЬКО про камеру:
# где она стоит, как наклонена, какое framing/lens/perspective.
CAMERA_HINTS_RICH: Dict[str, str] = {
    "low angle wide shot": (
        "Low angle shot, camera placed near floor level and tilted upward, "
        "wide framing with a lot of headroom, dramatic upward perspective"
    ),
    "high angle medium shot": (
        "High angle shot, camera elevated above the action and tilted "
        "downward at roughly 45 degrees, medium framing, foreshortened "
        "top-down perspective"
    ),
    "eye-level close-up": (
        "Eye-level shot, tight close-up framing, direct horizontal "
        "perspective, shallow depth of field"
    ),
    "over-the-shoulder medium shot": (
        "Over-the-shoulder camera angle, medium framing with a soft "
        "out-of-focus foreground edge in the bottom corner of frame, "
        "classic shoulder-anchored perspective"
    ),
    "extreme wide establishing shot": (
        "Extreme wide establishing shot, camera pulled far back, very "
        "wide framing, deep field of view, epic environmental scale"
    ),
    "tight close-up on face": (
        "Tight close-up shot, camera placed extremely close, very shallow "
        "depth of field, intimate macro framing"
    ),
    "three-quarter view medium shot": (
        "Three-quarter angle, camera roughly 45 degrees off direct front, "
        "medium framing, classic portrait-style composition"
    ),
    "front-on medium-wide shot": (
        "Front-on view, camera placed directly facing forward at chest "
        "level, medium-wide framing, symmetric centered composition"
    ),
    "side profile medium shot": (
        "Side profile shot, camera positioned perpendicular to the action "
        "axis at hip level, medium framing, classic profile silhouette "
        "composition"
    ),
    "dutch angle medium shot": (
        "Dutch angle, camera tilted roughly 15 to 20 degrees off horizontal, "
        "medium framing, dramatic skewed composition with tilted horizon"
    ),
}


def rich_angle_phrase(preset: str) -> str:
    """CHAR-anchored cinematic фраза для пресета. Если пресет не в словаре —
    возвращаем сам пресет как fallback (не валим pipeline)."""
    return CAMERA_HINTS_RICH.get(preset, preset)


# ============================================================ style stack

# Style stack который аппендится в конец каждого финального prompt'а.
# Вокабуляр ровно тот же что в SYSTEM_PROMPT_IMAGE из chunk_prompts_2.py
# (Pixar 3D CGI Toon look), плюс стандартные cinematic-токены освещения.
# P1x4r LoRA-триггер НЕ включён намеренно — в текущем qwen_image.json нет
# LoraLoaderModelOnly для P1x4r чекпойнта (см. node 179 — там только
# Qwen-Image-Edit-2511-Lightning-8steps-V1.0). Если ты в будущем подключишь
# P1x4r LoRA, добавь её триггер в начало STYLE_STACK или префиксом angle-фразы.
STYLE_STACK = (
    "Pixar CGI Toon Style, stylized CGI character, pixar-like 3D render, "
    "cinematic animated character, high-end animated film style, "
    "soft volumetric lighting, PBR textures, subsurface scattering on skin, "
    "smooth highlights"
)


# ============================================================ short alias

# Регексп для отрезания style-фразы вида "Pixar CGI Toon Style," (или вариаций)
# В ЛЮБОМ месте текста — иногда GROK дописывает её и в конце base_prompt'а.
# Не схлопываем сразу, а вырезаем все вхождения.
_STYLE_PHRASE_RE = re.compile(
    r"\bpixar(?:\s+(?:cgi|3d|cg|toon|cartoon))*(?:\s+style)\b\s*[,.;:\-]?\s*",
    re.IGNORECASE,
)

# Регексп для нормализации множественных пробелов / переводов строк.
_WS_RE = re.compile(r"\s+")


def short_alias_from_base_prompt(
    base_prompt: str,
    max_words: int = 18,
    max_clauses: int = 4,
) -> str:
    """Эвристика: получить descriptive alias из base_prompt'а.

    Шаги:
      1. Удаляем все вхождения style-фразы ("Pixar CGI Toon Style") в любом
         месте — стилистика и так аппендится через STYLE_STACK.
      2. Режем до первой точки (точки внутри чисел вроде "1.5m" не считаем —
         ищем точку с пробелом или концом строки после).
      3. Чиним whitespace.
      4. Берём не более max_clauses запятых-разделённых клауз и не более
         max_words слов в сумме. Это даёт компактный alias на 1-2 строки,
         без оборванных "deep dark eyes with" хвостов.

    Примеры (из main.json):
      "Pixar CGI Toon Style, A 185cm tall, slender, strikingly handsome young
       man with sharp aristocratic features, jet black hair styled in a neat
       modern fade, deep dark eyes with..."
      → "A 185cm tall, slender, strikingly handsome young man with sharp
         aristocratic features, jet black hair styled in a neat modern fade"
         (4 клаузы, ~22 слова, обрезаем до 18 → последняя клауза сократится)

      "A dark, damp drainage channel at night, concrete walls with visible
       stains and shadows, Pixar CGI Toon Style"
      → "A dark, damp drainage channel at night, concrete walls with visible
         stains and shadows" (style-фраза вырезана из конца)
    """
    s = (base_prompt or "").strip()
    if not s:
        return ""

    # 1. Удаляем стиль-фразы везде
    s = _STYLE_PHRASE_RE.sub("", s)

    # 2. До первой точки (с пробелом или концом строки)
    m = re.search(r"\.(\s|$)", s)
    if m:
        s = s[: m.start()]

    # 3. Cleanup
    s = _WS_RE.sub(" ", s).strip(" ,.;:-")

    # 4. Лимит по клаузам и словам.
    # Берём не больше max_clauses запятых-разделённых частей. Это сохраняет
    # естественные семантические границы и не оставляет оборванных хвостов.
    clauses = [c.strip() for c in s.split(",")]
    clauses = [c for c in clauses if c]
    if max_clauses and len(clauses) > max_clauses:
        clauses = clauses[:max_clauses]
    s = ", ".join(clauses)

    # Если суммарно всё ещё слишком длинно — режем по словам, но при этом
    # стараемся остановиться на запятой чтобы не оставлять хвост типа
    # "deep dark eyes with".
    words = s.split()
    if len(words) > max_words:
        truncated = " ".join(words[:max_words])
        last_comma = truncated.rfind(",")
        if last_comma > len(truncated) // 2:
            s = truncated[:last_comma].strip()
        else:
            s = truncated.strip()

    return s.strip(" ,.;:-")


# ============================================================ prompt builder

_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")


def build_qwen_edit_prompt(
    image_prompt: str,
    base_prompts_by_name: Dict[str, dict],
    camera_preset: str,
    max_pictures: int = 5,
    style_stack: str = STYLE_STACK,
    *,
    continuity_mode: str = "none",  # "none" | "soft" | "hard"
) -> Tuple[str, List[str]]:
    """Собрать финальный prompt для TextEncodeQwenImageEditPlus + список
    entity-имён в порядке Picture 1..N.

    Параметры:
      image_prompt          - сырой `scene["image_prompt"]` с маркерами
                              `[entity_name]`. Например:
                              "[shen_fei] crouching before [lin_shuang] in
                              [drainage_channel]. Pixar CGI Toon Style."
      base_prompts_by_name  - dict {base_name: base_prompt_dict}, обычно
                              `{bp["base_name"]: bp for bp in plan["base_prompts"]}`.
                              Каждый bp_dict имеет ключ "base_prompt" (string)
                              и опциональный "short_alias" (string, override
                              для эвристики).
      camera_preset         - выходной токен `pick_camera_preset(idx)` из
                              chunk_prompts_2.py. Например "side profile
                              medium shot".
      max_pictures          - hard cap. Qwen Edit поддерживает максимум 3
                              reference картинки.
      style_stack           - что аппендить в конец prompt'а. По умолчанию —
                              STYLE_STACK (Pixar 3D CGI Toon).

    Возвращает:
      (final_prompt, ordered_entities) — entity-имена в порядке появления,
      без дубликатов, отфильтрованные по `base_prompts_by_name`. Caller
      строит `image_paths` по этому списку.

    Логика:
      1. Резолвим entity из `[name]` маркеров image_prompt'а, фильтруя
         только те что есть в `base_prompts_by_name` (отсеиваем мусорные
         скобки типа `[wide shot]`), дедупаем, режем до `max_pictures`.
      2. Каждое `[name]` подменяем на `Picture N (alias)` со всеми
         occurrences. Alias берётся из `bp["short_alias"]` если задано,
         иначе через `short_alias_from_base_prompt(bp["base_prompt"])`.
      3. Любые оставшиеся `[unknown]` токены превращаем в plain text
         (убираем скобки, оставляем содержимое).
      4. Префиксим CHAR-anchored angle-фразой, аппендим style stack.

    Пример:
      image_prompt = "[shen_fei] crouching before [lin_shuang] in [drainage_channel]."
      camera_preset = "side profile medium shot"
      base_prompts_by_name = {
          "shen_fei": {"base_prompt": "Pixar CGI Toon Style, A 185cm tall slender handsome young man with short black hair, wearing dark casual clothing", ...},
          "lin_shuang": {"base_prompt": "Pixar CGI Toon Style, slender elegant young woman with sharp angular features, dark robes", ...},
          "drainage_channel": {"base_prompt": "A dark, damp drainage channel at night, concrete walls with visible stains", ...},
      }

      final_prompt = (
          "Side profile two-shot, camera at hip level perpendicular to the line "
          "connecting both characters, framing them both from waist-up against "
          "the background, classic side-view composition where the viewer sees "
          "their faces in pure profile silhouettes. "
          "Picture 1 (A 185cm tall slender handsome young man with short black "
          "hair, wearing dark casual clothing) crouching before "
          "Picture 2 (slender elegant young woman with sharp angular features, "
          "dark robes) in "
          "Picture 3 (A dark, damp drainage channel at night, concrete walls "
          "with visible stains). "
          "Pixar CGI Toon Style, stylized CGI character, ..."
      )
      ordered_entities = ["shen_fei", "lin_shuang", "drainage_channel"]
    """
    if not isinstance(image_prompt, str) or not image_prompt.strip():
        raise ValueError("image_prompt must be a non-empty string")

    # 1. Резолвим entities в порядке появления, фильтруя по base_prompts_by_name
    entities: List[str] = []
    seen: set[str] = set()

    # Если continuity_mode != "none", Picture 1 зарезервирована под previous-frame
    # (caller пихает prev_image_path в начало image_paths). Все base entities
    # сдвигаются на Picture 2..N, и effective max_pictures для них снижается на 1.
    picture_offset = 1 if continuity_mode in ("soft", "hard") else 0
    effective_max = max_pictures - picture_offset

    for m in _BRACKET_TOKEN_RE.finditer(image_prompt):
        name = m.group(1).strip()
        if name in base_prompts_by_name and name not in seen:
            seen.add(name)
            entities.append(name)
            if len(entities) >= effective_max:
                break

    # 2. Подмена [entity] -> Picture N (alias), с учётом offset
    out = image_prompt
    for i, name in enumerate(entities):
        bp = base_prompts_by_name.get(name, {})
        alias = (bp.get("short_alias") or "").strip()
        if not alias:
            alias = short_alias_from_base_prompt(bp.get("base_prompt", ""))
        if not alias:
            alias = name.replace("_", " ")

        replacement = f"Picture {i + 1 + picture_offset} ({alias})"
        out = re.sub(
            r"\[\s*" + re.escape(name) + r"\s*\]",
            replacement,
            out,
        )

    # 3. Стрипаем оставшиеся неизвестные [bracketed] → plain text
    out = _BRACKET_TOKEN_RE.sub(lambda m: m.group(1).strip(), out)

    # 3.5. Удаляем style-фразы из body — иначе будет дублирование с STYLE_STACK
    # в финальном prompt'е (image_prompt'ы из GROK plan'а часто оканчиваются
    # "Pixar CGI Toon Style, cinematic lighting." или похоже).
    out = _STYLE_PHRASE_RE.sub("", out)
    # Подчищаем оставшиеся артефакты типа ", cinematic lighting." в конце
    # (style вырезан, но повисший хвост со стилистикой остаётся).
    # Снимаем хвостовые orphaned-фрагменты "., cinematic lighting" / ", glowing
    # interface" и т.п. — но только из конца body, чтобы не ломать действие.
    out = re.sub(r"[,\s]+(?:cinematic|glowing|detailed|intense|atmospheric|moody)\s+\w+\s*\.?\s*$", "", out, flags=re.IGNORECASE)
    # Финальный cleanup: лишние " , " и пробелы
    out = re.sub(r"\s*,\s*,\s*", ", ", out)
    out = re.sub(r"\s*,\s*\.", ".", out)
    out = re.sub(r"\.\s*\.+", ".", out)

    # 4. Финальная сборка: angle + continuity prefix + body + style stack
    body = _WS_RE.sub(" ", out).strip(" ,.;:-")
    angle = rich_angle_phrase(camera_preset).rstrip(". ").strip()
    style = style_stack.strip().rstrip(".")

    if continuity_mode == "soft":
        continuity_prefix = (
            "Picture 1 shows the immediately preceding moment of this same scene. "
            "The action below is a direct continuation: characters keep the same "
            "pose, expression, clothing, body position, and any props they were "
            "holding from Picture 1. Camera angle may shift to the new framing "
            "described below, but character state is preserved."
        )
    elif continuity_mode == "hard":
        continuity_prefix = (
            "Picture 1 shows the exact previous frame of this same scene. "
            "This image is the very next moment, with only minor incremental "
            "motion: characters in identical pose, expression, clothing, and "
            "position; props in identical placement. The camera does NOT cut — "
            "framing and angle remain the same as Picture 1."
        )
    else:
        continuity_prefix = ""

    if continuity_prefix:
        final = f"{angle}. {continuity_prefix} {body}. {style}.".strip()
    else:
        final = f"{angle}. {body}. {style}.".strip()
    final = _WS_RE.sub(" ", final)
    return final, entities


# ============================================================ self-test

if __name__ == "__main__":
    # Быстрый sanity-check на base_prompts из реального main.json
    import json
    import sys

    plan_path = sys.argv[1] if len(sys.argv) > 1 else "main.json"
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    base_prompts_by_name = {bp["base_name"]: bp for bp in plan.get("base_prompts", [])}

    print("=" * 70)
    print("base_prompts → short_alias")
    print("=" * 70)
    for name, bp in base_prompts_by_name.items():
        alias = short_alias_from_base_prompt(bp.get("base_prompt", ""))
        print(f"  {name:25s} -> {alias}")

    print()
    print("=" * 70)
    print("Sample built prompts (first 5 scenes)")
    print("=" * 70)

    # CAMERA_PRESETS из chunk_prompts_2.py
    camera_presets = [
        "low angle wide shot",
        "high angle medium shot",
        "eye-level close-up",
        "over-the-shoulder medium shot",
        "extreme wide establishing shot",
        "tight close-up on face",
        "three-quarter view medium shot",
        "front-on medium-wide shot",
        "side profile medium shot",
        "dutch angle medium shot",
    ]

    for i, scene in enumerate(plan.get("scenes", [])[:5]):
        camera = camera_presets[i % len(camera_presets)]
        try:
            final, ents = build_qwen_edit_prompt(
                image_prompt=scene["image_prompt"],
                base_prompts_by_name=base_prompts_by_name,
                camera_preset=camera,
            )
        except Exception as e:
            print(f"\n--- scene {scene.get('scene_id')} FAILED: {e} ---")
            continue
        print(f"\n--- scene {scene.get('scene_id')} ({camera}) ---")
        print(f"  entities: {ents}")
        print(f"  prompt:   {final}")
