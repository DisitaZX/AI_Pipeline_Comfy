"""Детерминистический prompt-builder для FLUX.2 Klein multi-reference image edit.

Задача модуля:
  - Для каждой entity извлекает short alias из её `base_prompt`.
  - Подменяет `[base_name]` в `scene["image_prompt"]` на `{alias} from image N`,
    дедупит entity, лимитит до 5 (hard cap у Klein multi-ref).
  - Reorder: characters/objects первыми (identity-critical), location последней.
  - Добавляет `high_budget_anime_movie_style` prefix.

Никаких сетевых вызовов, никаких LLM. Полностью детерминистический.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ============================================================ style stack

# STYLE_STACK — РАЗВЁРНУТЫЙ style descriptor, используется ТОЛЬКО для
# генерации базовых entity-refs через `qwen_variant_instruction` (character
# face / location side-views и т.п.). Туда нужно много текста — ref-картинки
# рисуются с нуля, без визуального contextа, поэтому стиль приходится
# проговаривать.
#
# Для финальных keyframe-prompt'ов (`build_qwen_edit_prompt` для Klein c1)
# используется только компактный "high_budget_anime_movie_style" prefix —
# стиль уже зашит в латентах reference'ов, повторение длинной style-фразы
# съедает text-attention budget впустую.
STYLE_STACK = (
    "high_budget_anime_movie_style, 3D cinematic anime, detailed sculptural character design "
    "with realistic skin and fabric shaders, dramatic volumetric cinematic lighting, "
    "shallow depth of field, filmic color grading, stylized anime proportions with "
    "physically-based rendering, soft ambient occlusion, clean sharp edges, "
    "high production value environment, cinematic atmosphere."
)


# ============================================================ entity types

# Базовые роли в plan'е (поле base_prompts[].entity_type). GROK обязан
# выставлять одно из трёх значений на каждой переменной. Любое другое —
# strict fail в build_qwen_edit_prompt и в стадии генерации рефов.
ENTITY_TYPES: tuple[str, ...] = ("character", "location", "object")


# Ref-варианты на каждую entity_type. Index 0 — "seed", который рисуется
# text-to-image (Z-Anime для старого Z-Anime.json workflow или FLUX.2 для
# нового image_flux2_klein_base_refs.json) с подсказкой про каноническую
# позу/ракурс. Все последующие индексы (если они есть) — Qwen Image-Edit
# от seed'а (тот же subject, другая сторона/кадрирование). Все варианты
# находятся в eye-level horizontal perspective, чтобы не конфликтовать с
# произвольным camera_preset сцены.
#
# locations: 4 поворота вокруг центра (front/right/left/back), все при
#   камере ~1.6m над землёй. Front — seed (text-to-image), остальные
#   три — Qwen-edit от seed'а.
# characters: ОДИН реф — полный рост (fullbody) text-to-image. Никаких
#   Qwen-edit вариантов: face-портрет был удалён, во всех сценах
#   character'а Klein получает только fullbody-реф. Это даёт стабильную
#   identity-preservation (один и тот же latent на всех сценах), а
#   композировать close-up портрет на сцене Klein умеет и сам по
#   image_prompt'у.
# objects: один реф без variant-суффикса (как было раньше).
_ENTITY_VARIANTS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "character": (),
    "location":  (),
    "object":    (),
}


def entity_variants(entity_type: str) -> tuple[str, ...]:

    if entity_type not in _ENTITY_VARIANTS_BY_TYPE:
        raise ValueError(
            f"unknown entity_type {entity_type!r}; expected one of {ENTITY_TYPES}"
        )
    return _ENTITY_VARIANTS_BY_TYPE[entity_type]


def entity_seed_variant(entity_type: str) -> str:
    """Имя seed-варианта (генерится text-to-image). Пусто для object."""
    variants = entity_variants(entity_type)
    return variants[0] if variants else ""


def entity_klein_variants(entity_type: str) -> tuple[str, ...]:
    """Имена variant'ов которые надо догенерить через Klein-edit из seed'а."""
    variants = entity_variants(entity_type)
    return variants[1:] if len(variants) > 1 else ()


def entity_ref_filename(base_name: str, variant: str | None) -> str:
    """Имя файла ref-картинки (как пишет ComfyUI SaveImage).

    SaveImage аппендит "_00001_.png" к filename_prefix. Соглашение:
      - object  (variant пуст/None): "{base_name}_00001_.png"
      - char/loc (variant задан):     "{base_name}_{variant}_00001_.png"
    """
    if variant:
        return f"{base_name}_{variant}_00001_.png"
    return f"{base_name}_00001_.png"


# ============================================================ ref-gen prompts


def seed_prompt_for_entity(entity_type: str, base_prompt: str) -> str:
    """Text-to-image prompt для seed-рефа (через Z-Anime).

    Аппендит к base_prompt'у entity-type-specific framing: канонический
    eye-level ракурс, plain background, чтобы Klein потом от этого seed'а
    мог рисовать варианты без артефактов.

    Strict-fail если entity_type не в ENTITY_TYPES.
    """
    if entity_type == "character":
        return (
            f"{base_prompt} "

            f"plain neutral background, no environmental "
            f"props, no cropping"
        )
    if entity_type == "location":
        return (
            f"{base_prompt}, wide establishing shot seen from the front "
            f"side at mid-level view (camera height roughly 1.6 meters above "
            f"the ground), level horizontal perspective with the horizon "
            f"line across the middle of the frame, facing into the "
            f"location with all main features visible, no characters or "
            f"people in the frame"
        )
    if entity_type == "object":
        return f"{base_prompt}"
    raise ValueError(
        f"unknown entity_type {entity_type!r}; expected one of {ENTITY_TYPES}"
    )


def qwen_variant_instruction(
    entity_type: str,
    variant: str,
    base_prompt: str,
    style_stack: str = STYLE_STACK,
) -> str:
    """Prompt для Qwen-Image-Edit, который от seed-рефа рисует variant.

    Qwen Edit отзывчив на императивные инструкции и явное описание
    целевого state'а. В отличие от Klein-edit (preservation-biased)
    Qwen Edit реально умеет в умеренный 3D-поворот для архитектурных
    объектов, если ему сказать что должно появиться в кадре после
    поворота и что должно исчезнуть.

    Picture 1 = seed-реф (передаётся в Qwen как единственный input image,
    подключённый ко всем трём слотам image1/image2/image3 либо только
    к image1 — patch функция решает).

    Strict-fail если variant не из entity_klein_variants(entity_type).
    Имя функции по историческим причинам сохраняет namespace
    klein_*; см. ниже alias `klein_variant_instruction` для обратной
    совместимости с импортом.
    """
    if entity_type == "character":
        # У character больше нет Qwen-edit вариантов: единственный реф —
        # fullbody seed (text-to-image), face-портрет был удалён. Если
        # сюда попали — это баг вызывающей стороны (entity_klein_variants
        # пуст для character'а и цикл должен быть пропущен).
        raise ValueError(
            f"character has no qwen-edit variants; got variant={variant!r}. "
            f"entity_klein_variants('character') = "
            f"{entity_klein_variants('character')!r}. The caller should "
            f"skip the qwen-edit loop entirely for characters."
        )

    if entity_type == "location":

        raise ValueError(
            f"unknown location variant {variant!r}; "
            f"expected one of {entity_klein_variants('location')}"
        )



    raise ValueError(
        f"unknown entity_type {entity_type!r}; expected one of {ENTITY_TYPES}"
    )


# Backward-compat alias: старые импорты `klein_variant_instruction`
# теперь резолвятся в qwen-версию (variant generation переехала на
# Qwen Image Edit; Klein остался только для chunk-keyframe'ов сцен).
klein_variant_instruction = qwen_variant_instruction


# ============================================================ variant pickers

# ============================================================ short alias


# Регексп для нормализации множественных пробелов / переводов строк.
_WS_RE = re.compile(r"\s+")


def short_alias_from_base_prompt(
    base_prompt: str,
    max_words: int = 10,
    max_clauses: int = 1,
) -> str:
    """Эвристика: получить КОМПАКТНЫЙ descriptive alias из base_prompt'а.

    Цель — выдать identity-hint на 5-10 слов: достаточно чтобы Klein/Qwen
    привязали имя к нужной entity, но не так много чтобы alias начал
    конкурировать с reference-латентом за text-attention. Identity и так
    в латентах ref'ов; тексту нужен только short noun phrase, не
    полное описание персонажа на 20+ слов.

    Обычный формат base_prompt'а (из GROK): "Description... Japanese anime
    style, [extra style stuff]." — т.е. стиль в КОНЦЕ после точки.
    Первый рез по точке уже выбрасывает style-блок. max_clauses=1 и
    max_words=10 затем обрезают до одной короткой субъектной фразы.

    Шаги:
      1. Удаляем все вхождения style-фразы ("Pixar CGI Toon Style") в любом
         месте — стилистика и так в reference-латентах.
      2. Режем до первой точки (точки внутри чисел вроде "1.5m" не считаем —
         ищем точку с пробелом или концом строки после).
      3. Чиним whitespace.
      4. Берём не более max_clauses запятых-разделённых клауз и не более
         max_words слов в сумме. Это даёт alias на одну короткую фразу,
         без оборванных "deep dark eyes with" хвостов.

    Примеры (из main.json) с дефолтами max_words=10, max_clauses=1:
      "A thirteen-year-old Chinese youth with an aura of profound calmness,
       deep as an abyss. He has a mortal fate. Japanese anime style."
      → "A thirteen-year-old Chinese youth with an aura" (1 клауза, 8 слов
         после strip trailing "of")

      "The terrifying interior of the Nine Saints Demon Gates, dark ominous
       stone architecture. Japanese anime style."
      → "The terrifying interior of the Nine Saints Demon Gates" (1 клауза
         срезана по запятой, 9 слов)
    """
    s = (base_prompt or "").strip()
    if not s:
        return ""

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

    # Срезаем хвостовые предлоги/артикли/связки, чтобы не оставалось
    # "with an aura of" / "of the Nine Saints Demon" / "skilled messenger of"
    # как финал alias'а. Также чистим пунктуацию.
    _TRAILING_FILLER = {
        "a", "an", "the", "of", "with", "and", "or", "but", "to", "for",
        "in", "on", "at", "by", "from", "into", "onto", "as", "is", "are",
        "was", "were", "has", "have", "had", "be", "been", "being",
    }
    s = s.strip(" ,.;:-")
    parts = s.split()
    while parts and parts[-1].lower() in _TRAILING_FILLER:
        parts.pop()
    s = " ".join(parts).strip(" ,.;:-")

    return s


# ============================================================ prompt builder

_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")

def build_qwen_edit_prompt(
    image_prompt: str,
    base_prompts_by_name: Dict[str, dict],
    max_pictures: int = 5,
    *,
    image_offset: int = 0,
) -> Tuple[str, List[str]]:

    if not isinstance(image_prompt, str) or not image_prompt.strip():
        raise ValueError("image_prompt must be a non-empty string")
    if image_offset < 0:
        raise ValueError(f"image_offset must be >= 0, got {image_offset}")

    def _etype(name: str) -> str:
        bp = base_prompts_by_name[name]
        et = bp.get("entity_type")
        if et not in ENTITY_TYPES:
            raise ValueError(
                f"base_prompts[{name!r}].entity_type is {et!r}; "
                f"expected one of {ENTITY_TYPES}. Re-run GROK plan stage "
                f"with the updated schema (each base_prompt must have "
                f"entity_type)."
            )
        return et

    # 1. Резолвим entities в порядке появления, фильтруя по base_prompts_by_name
    raw_entities: List[str] = []
    seen: set[str] = set()
    for m in _BRACKET_TOKEN_RE.finditer(image_prompt):
        name = m.group(1).strip()
        if name in base_prompts_by_name and name not in seen:
            seen.add(name)
            raw_entities.append(name)

    # Проверяем, является ли промпт состоящим ровно из одной сущности, и это локация
    is_single_location_only = (
        len(raw_entities) == 1 and _etype(raw_entities[0]) == "location"
    )

    # 2. Обработка подстановок (без перестановки, в оригинальном порядке)
    image_entities: List[str] = []
    out = image_prompt

    for name in raw_entities:

        # применяем Picture N
        if len(image_entities) < max_pictures:
            image_entities.append(name)
            picture_idx = len(image_entities) + image_offset
            replacement = f"<Picture {picture_idx}>"
            out = re.sub(
                r"\[\s*" + re.escape(name) + r"\s*\]",
                replacement,
                out,
            )

    # 3. Стрипаем оставшиеся неизвестные или превысившие лимит [bracketed] → plain text
    out = _BRACKET_TOKEN_RE.sub(lambda m: m.group(1).strip(), out)

    # Финальный cleanup: лишние " , " и пробелы
    out = re.sub(r"\s*,\s*,\s*", ", ", out)
    out = re.sub(r"\s*,\s*\.", ".", out)
    out = re.sub(r"\.\s*\.+", ".", out)

    # 4. Финальная сборка: только style flag + body.
    body = _WS_RE.sub(" ", out).strip(" ,.;:-")

    final = f"{body}.".strip()
    final = _WS_RE.sub(" ", final)
    
    # Возвращаем final и только те сущности, которые реально стали "image N"
    return final, image_entities