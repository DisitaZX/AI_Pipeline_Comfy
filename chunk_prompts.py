"""
Промпты для пайплайна:

  - Wan 2.2 i2v (lightx2v / SVI v2 продолжение): get_chunk_prompts_from_ollama
    Сцена длиной N кадров режется на M чанков по ~81 кадру (5с при 15fps).
    Ollama за ОДИН вызов отдаёт массив из M chunk-промптов.

  - Qwen Image Edit 2511 (multi-image composer для сцен):
      * substitute_entities_to_image_slots — детерминистичная подмена
        `[base_name]` → `[image N]` в image_prompt сцены.
      * expand_image_prompt_for_qwen_edit — Gemma расширяет prompt
        композицией + освещением + атмосферой, сохраняя [image N] verbatim.
      * patch_qwen_edit_workflow — патчит qwen_image.json под конкретную
        сцену: подставляет image_paths, prompt, отрубает неиспользуемые
        image slot'ы (image2/image3) если в сцене < 3 reference'ов.
    Negative prompt захардкожен в workflow node 175, Gemma его не генерит.

Обе функции пользуются общим robust JSON-парсером (extract_json_object).
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import base64
import pprint
import random
from typing import Any, Sequence

import aiohttp


# Wan 2.2 VAE temporal stride = 4, валидные длины N = 4k + 1.
WAN_VAE_TEMPORAL = 4
WAN_DEFAULT_CHUNK = 81           # 5.4с @ 15fps - стандарт для i2v
WAN_MIN_TAIL = 17                # 1.07с - меньше нет смысла, артефакты на склейке
WAN_MAX_CHUNK = 81               # верхний предел одного прогона i2v


def round_to_valid_frames(n: int) -> int:
    """Ближайшее снизу валидное N для Wan VAE: N = 4k + 1."""
    n = max(1, n)
    return ((n - 1) // WAN_VAE_TEMPORAL) * WAN_VAE_TEMPORAL + 1


def split_into_chunks(
    total_frames: int,
    chunk_frames: int = WAN_DEFAULT_CHUNK,
    min_tail: int = WAN_MIN_TAIL,
    max_chunk: int = WAN_MAX_CHUNK,
) -> list[int]:
    """
    Делит total_frames на чанки по chunk_frames с ограничением:
      - первые M-1 чанков = chunk_frames;
      - последний чанк >= min_tail и валидный (N = 4k+1).

    Если хвост получается меньше min_tail - переносим часть кадров
    из предыдущего чанка в хвост, чтобы сравнять.
    """
    total_frames = round_to_valid_frames(total_frames)
    chunk_frames = round_to_valid_frames(chunk_frames)
    min_tail = round_to_valid_frames(min_tail)

    if total_frames <= chunk_frames:
        return [total_frames]

    # сколько чанков надо
    n_chunks = math.ceil(total_frames / chunk_frames)
    base = [chunk_frames] * (n_chunks - 1)
    tail = total_frames - sum(base)
    tail = round_to_valid_frames(tail)

    # короткий хвост: тянем кадры из предпоследнего чанка
    if tail < min_tail and n_chunks >= 2:
        deficit = min_tail - tail
        # переносим минимум кратно WAN_VAE_TEMPORAL чтобы остаться валидным
        steal = ((deficit + WAN_VAE_TEMPORAL - 1) // WAN_VAE_TEMPORAL) * WAN_VAE_TEMPORAL
        base[-1] = round_to_valid_frames(base[-1] - steal)
        tail = total_frames - sum(base)
        tail = round_to_valid_frames(tail)

    return base + [tail]


# ---------------------------------------------------------------- ollama call


SYSTEM_PROMPT_WAN = """## Role
You are a **creative cinematographer and animator** writing video prompts for
**Wan 2.2 i2v (lightx2v 4-step distilled)**. Each scene is generated as
**{n_chunks} sequential ~5-second chunks**, and you must produce **exactly
{n_chunks} prompts** - one per chunk - that bring the still image to life.

The user gives you a SEED prompt (a few words sketching the scene). Your job is
NOT to copy or rephrase that seed. Your job is to **invent rich, specific motion
for each chunk**: micro-actions of characters, environmental effects, props
interactions, lighting shifts, and continuous camera language. The seed is a
starting hint; you are the director who decides everything that happens on screen.

## Invention requirements (per chunk)
Every chunk's prompt MUST contain at least:
  1. ONE primary character or subject action that progresses the scene.
  2. ONE secondary micro-motion (a hand twitch, hair movement, breath, fabric
     shift, a blink, a small object falling, dust motes, steam, etc.).
  3. ONE environmental / atmospheric effect (wind, particles, flickering light,
     reflections rippling, fog drifting, leaves falling, embers floating, etc.).
  4. ONE explicit camera move (slow dolly-in, gentle pan-left, subtle handheld
     drift, rack-focus from foreground to subject, push-in on the eyes, etc.).
Do not just say "the wind blows" - describe HOW it interacts with the subject
("a cold gust catches the edge of his cloak, lifting it briefly before it
settles back against his thigh").

## Continuation rules across chunks
- Chunk 0 starts from the user-provided image. Composition and characters are
  already established by that image; describe what STARTS to move first.
- Each subsequent chunk is conditioned on the LAST FRAME of the previous chunk.
  Treat each chunk as a continuation, NOT a new shot.
- Do not restart actions. If chunk 0 ends mid-action (door half-open, character
  mid-step, hand reaching out), chunk 1 must CONTINUE that motion to its next
  natural beat, not begin again from rest.
- Camera language must flow across chunks (chunk 0 "slow dolly-in begins",
  chunk 1 "dolly-in continues, the lens tilts down a few degrees", chunk 2
  "camera settles into a static close-up").
- No new characters or props appear mid-scene unless the seed explicitly says so.
- Lighting and time-of-day stay continuous across chunks (no day-to-night jumps
  unless the seed asks).

## Story-so-far (cross-scene continuity)
The user may provide a `Story so far` section in Russian summarizing what
happened in PREVIOUS scenes of the same video. Use it to:
  - Keep the character's emotional state and posture consistent with what they
    were doing at the end of the previous scene (e.g. if they ended scene N-1
    smirking at a holographic panel, scene N should not open with a neutral
    face for no reason).
  - Avoid inventing actions that contradict already-established plot facts
    (props in hand, location, who is present, what just happened).
  - Reuse the same lighting register and color grade as recent scenes unless
    the seed explicitly signals a transition.
Do NOT narrate or recap the prior story in your prompts - the recap is
background knowledge for YOU. Your output still describes only what happens
in THIS scene's chunks.

## Format per prompt
- ONE single continuous paragraph. No bullets, no line breaks, no markdown.
- **150-190 words** per prompt. Use the full budget; do not output short prompts.
  Wan 2.2 lightx2v handles this length fine as long as details are CONSISTENT
  with the image and the seed (the danger is contradictions, not length).
- Present tense, active verbs ("she tilts her head", "the lantern flickers",
  "dust drifts across the beam of light").
- No internal states ("she feels sad" forbidden) - express emotions through
  PHYSICAL actions ("her shoulders sag, her eyes lower, her fingers slowly curl
  inward").
- No audio descriptions (Wan i2v does not generate sound).
- No meta references ("frame", "second", "next scene", "chunk", "video").
- No quoted dialogue. No on-screen text.

## Treat the seed as inspiration, not a script
If the seed says "slow gentle camera zoom in on the face", that is ONE element.
You must surround it with invented detail: what micro-expressions does the face
show, what does the hair do, what does the background do, how does the lighting
transition, what secondary movement happens at the edges of the frame. Aim for
the richness of a director's shot description, not a one-line stage direction.

## Output
Return STRICT JSON:
```
{{"prompts": ["chunk 0 prompt", "chunk 1 prompt", ...]}}
```
Length of `prompts` array MUST equal {n_chunks}. No extra fields, no markdown
fences, no commentary outside the JSON."""


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------- robust JSON


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


def _balanced_object(raw: str) -> str | None:
    """Находит первый сбалансированный {...} с учётом строк/экранирования."""
    start = raw.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
    return None


def _repair_unterminated_string(raw: str) -> str | None:
    """Gemma иногда роняет закрывающую `"` у последнего элемента массива
    перед ]}. Пробуем восстановить."""
    m = re.search(r"^(.*?)\]\s*\}", raw, re.S)
    if not m:
        return None
    body = m.group(1)
    n_quotes, i = 0, 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            i += 2
            continue
        if ch == '"':
            n_quotes += 1
        i += 1
    if n_quotes % 2 == 1:
        return body + '"]}'
    return None


def extract_json_object(raw: str) -> Any:
    """
    Робастно вытягивает JSON-объект из вывода LLM.

    Стратегии по очереди:
      1. Прямой json.loads.
      2. raw_decode (съедает длиннейший валидный префикс) - спасает от хвостов
         вроде `}<channel|>...` которые Gemma иногда плюёт после закрытия.
      3. Сбалансированный поиск {...} с учётом строк.
      4. Ремонт одиночного незакрытого `"` у последнего элемента.
    Бросает ValueError если ни одна не сработала.
    """
    raw = _strip_code_fences(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
        return obj
    except json.JSONDecodeError:
        pass

    cand = _balanced_object(raw)
    if cand:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

    repaired = _repair_unterminated_string(raw)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    raise ValueError("could not extract JSON object from LLM output")


async def get_chunk_prompts_from_ollama(
    scene_text: str,
    video_prompt: str,
    image_path: str | None,
    n_chunks: int,
    chunk_frames: Sequence[int],
    fps: int = 15,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    previous_recap: str = "",
) -> list[str]:
    """
    Возвращает список из n_chunks промптов для последовательных i2v-чанков.

    Делается ОДИН вызов Ollama с format=json - модель видит сцену целиком
    и пишет согласованные между собой чанки.

    Параметры:
      scene_text     - текст озвучки этой сцены (русский, из GROK).
      video_prompt   - seed-промпт от GROK на английском (1-2 предложения).
      image_path     - путь к init-картинке для чанка 0 (Z-Image Turbo выход).
      n_chunks       - сколько чанков ожидается (= len(chunk_frames)).
      chunk_frames   - длины чанков в кадрах [81, 81, 57] и т.п.
      fps            - частота кадров видео (для перевода кадров в секунды).
      previous_recap - russian-язычное summary предыдущих сцен этого видео.
                       Передаётся в Gemma как контекст для непрерывности
                       состояния героя/локации между сценами. Пустая строка
                       для первой сцены. В output-промпты пересказ не
                       просачивается - только косвенно влияет на консистентность.
    """
    assert n_chunks == len(chunk_frames), \
        f"n_chunks={n_chunks} != len(chunk_frames)={len(chunk_frames)}"

    chunk_seconds = [round(f / fps, 2) for f in chunk_frames]

    recap_block = (
        f"## Story so far (для непрерывности, не пересказывать в промптах)\n"
        f"{previous_recap.strip()}\n\n"
        if previous_recap and previous_recap.strip()
        else ""
    )

    user_payload = (
        recap_block
        + f"## Scene context\n"
        f"Spoken text: {scene_text}\n\n"
        f"## High-level direction\n"
        f"{video_prompt}\n\n"
        f"## Chunks to generate ({n_chunks} total)\n"
        + "\n".join(
            f"- chunk {i}: {chunk_frames[i]} frames ({chunk_seconds[i]}s)"
            for i in range(n_chunks)
        )
        + "\n\nReturn JSON {\"prompts\": [...]} with exactly "
        + f"{n_chunks} entries."
    )

    images_payload: list[str] = []
    if image_path and os.path.exists(image_path):
        images_payload.append(encode_image_b64(image_path))

    # Ретраи: Gemma при высокой t иногда пробивает format=json выводя
    # template-control токены (`<channel|>`, `<tool_call|>`) после закрывающего
    # `]}`. На каждый ретрай слегка прижимаем сэмплинг и меняем seed,
    # чтобы не попасть в тот же бад-путь.
    retry_profiles = [
        {"temperature": 0.95, "top_p": 0.85},
        {"temperature": 0.80, "top_p": 0.90},
        {"temperature": 0.60, "top_p": 0.95},
    ]
    last_err: Exception | None = None
    last_raw: str = ""

    timeout = aiohttp.ClientTimeout(total=None)
    for attempt, profile in enumerate(retry_profiles):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": model,
                "system": SYSTEM_PROMPT_WAN.format(n_chunks=n_chunks),
                "prompt": user_payload,
                "images": images_payload,
                "format": "json",
                "stream": False,
                "keep_alive": 5,
                "options": {
                    "temperature": profile["temperature"],
                    "top_p": profile["top_p"],
                    "repeat_penalty": 1.15,
                    "num_predict": 2048,
                    "seed": random.randint(1, 2**31 - 1),
                    # Обрезаем template-лики если модель начнёт их лить.
                    # `<|`, `<channel`, `<tool_call` - типичные лики chat template'a;
                    # `}<` ловит любой garbage сразу после закрывающей `}`.
                    "stop": ["<|", "<channel", "<tool_call", "```\n\n", "}<"],
                },
            }
            async with session.post(ollama_url, json=payload) as response:
                result = await response.json()
                last_raw = result.get("response", "").strip()

        try:
            parsed = extract_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(
                f"[chunk_prompts] attempt {attempt + 1}/{len(retry_profiles)} "
                f"failed to parse JSON: {e}; retrying with safer profile"
            )
            continue

        prompts_field = parsed.get("prompts") if isinstance(parsed, dict) else None
        if not isinstance(prompts_field, list) or len(prompts_field) != n_chunks:
            last_err = RuntimeError(
                f"Expected {n_chunks} prompts, got: {prompts_field!r}"
            )
            print(
                f"[chunk_prompts] attempt {attempt + 1}/{len(retry_profiles)} "
                f"wrong shape ({last_err}); retrying"
            )
            continue

        return [str(p).strip() for p in prompts_field]

    pprint.pprint(("RAW", last_raw))
    raise RuntimeError(
        f"Ollama returned invalid JSON after {len(retry_profiles)} attempts: "
        f"{last_err}"
    )


# ============================================================ Qwen Image Edit
# Multi-image conditioning composer для сцен.
#
# Pipeline (взамен Z-Image Turbo на сцены):
#   1. substitute_entities_to_image_slots: [base_name] -> [image N] детерм.
#   2. expand_image_prompt_for_qwen_edit:  Gemma расширяет prompt
#      (composition + lighting + atmosphere) сохраняя [image N] verbatim.
#   3. patch_qwen_edit_workflow: патч qwen_image.json под конкретную сцену
#      (image_paths, prompt, отрубаем неиспользуемые image2/image3 если
#      в сцене < 3 reference'ов).
#
# Z-Image Turbo остаётся для генерации канон-references (Stage 0, один раз
# на главу, по одной ref на base_name из plan["base_prompts"]).
#
# Negative prompt захардкожен в qwen_image.json node 175 - модель его
# сама подаёт, Gemma'е генерить его не нужно.


SYSTEM_PROMPT_IMAGE = """Ты — image-prompt expander для модели Qwen Image Edit 2511. Твоя задача — взять готовый source prompt с маркерами `[image 1]`, `[image 2]`, `[image 3]` и расширить его композиционными и стилистическими деталями, ничего не убирая и не подменяя сами маркеры.

КОНТЕКСТ:
Каждый `[image N]` в source — это ссылка на отдельную reference картинку, которая будет подана в multi-image conditioning encoder вместе с текстом. Модель сама ВИДИТ что изображено на каждой reference картинке через VAE и Qwen2.5-VL ветки. Твоя задача НЕ описывать что на reference'ах (это бесполезно — модель и так видит), а описать КАК они должны быть сцеплены вместе в новом кадре: позы, взаимодействие, композицию, освещение, атмосферу, стиль.

ВХОДНЫЕ ДАННЫЕ:
- `source_prompt` — строка с одним или несколькими маркерами `[image N]` где N ∈ {1, 2, 3}.
- `n_images` — количество reference картинок (1, 2 или 3). Маркеры с N > n_images в source появляться НЕ ДОЛЖНЫ.
- `camera_hint` (опционально) — конкретный ракурс/кадрирование которое МНЕ нужно использовать в этой сцене (например `"low angle wide shot"`, `"over-the-shoulder medium shot"`). Если задан — это **жёсткое требование** к композиции, не общая рекомендация.
- `previous_recap` (опционально) — короткое русское описание того что происходило ранее в том же thread'е. Это контекст для понимания, не материал для копирования в output.

КАК РАСПОРЯДИТЬСЯ С `camera_hint`:

- Если `camera_hint` пустой — выбирай ракурс/кадрирование сам по своей логике (что лучше подходит сцене).
- Если `camera_hint` задан — output ОБЯЗАН **начинаться** с этого ракурса как первой композиционной фразы (после неё уже идёт описание action/позы и остальное). Дословно использовать переданный текст ракурса. НЕ подменять своим. НЕ переформулировать.
- Если задан `camera_hint` И в конце source/payload есть continuity hint про другую картинку — приоритет камеры выше: continuity hint описывает только action/состояние персонажа, не композицию. Качество композиции диктует ИСКЛЮЧИТЕЛЬНО `camera_hint`.

ЧТО ОБЯЗАТЕЛЬНО ДОБАВИТЬ В EXPANDED PROMPT:

1. **Камера/кадрирование в самом начале** — если задан `camera_hint`, дословно как первая фраза. Если нет — выбери сам и поставь в начало.

2. **Поза и взаимодействие** — как `[image 1]` физически относится к остальным маркерам. Положение тела, выражение лица, направление взгляда, жесты рук. Если в source уже сказано "sitting on chair" — расширь до "sitting in a relaxed posture with hands resting on armrests, leaning slightly forward".

3. **Композиция кадра (детали)** — правило третей или центральная композиция, глубина (foreground / mid-ground / background), что в фокусе.

4. **Освещение** — источник света (natural / desk lamp / overhead / window / candlelight), время суток, интенсивность, тёплое vs холодное, контраст, направление теней.

5. **Атмосфера** — настроение сцены одной фразой ("tense and contemplative" / "serene and peaceful" / "menacing and oppressive").

6. **Anime style stack в КОНЦЕ** — обязательная заключительная фраза: `anime style, 2D animation, cel-shaded, sharp lineart, vibrant flat colors, modern anime aesthetic`.

ЧТО ЗАПРЕЩЕНО:

- Изменять, удалять, переводить или подменять любой `[image N]` маркер. Они ВСЕ должны попасть в output дословно.
- Описывать что находится ВНУТРИ reference картинки ("[image 1] is a young man with black hair" — НЕТ; "[image 1] sitting in relaxed posture" — ДА).
- Добавлять новые объекты которых нет в source prompt. Если в source нет "wine glass" — не выдумывай "wine glass". Если нет "books on desk" — не добавляй books.
- Менять последовательность маркеров.
- Заменять `[image N]` на текстовые описания типа "the character" или "the protagonist".
- Генерировать negative prompt — он управляется отдельно и не твоя зона ответственности.
- Добавлять markdown, JSON, кавычки, заголовки, пояснения. Только сам prompt.
- Использовать русский язык — output только на английском.
- ИГНОРИРОВАТЬ `camera_hint` если он задан. Подменять его своим ракурсом — категорически запрещено.

САМОПРОВЕРКА ПЕРЕД ОТВЕТОМ:

1. Все `[image N]` маркеры из source присутствуют в output дословно? Если нет — переделай.
2. Не появилось ли в output `[image N]` с N большим чем n_images? Если да — переделай.
3. Если задан `camera_hint` — он стоит в начале output дословно? Если нет — переделай, поставь в начало.
4. Output длиннее source минимум в 1.5 раза? Если нет — добавь больше композиции/освещения/атмосферы.
5. Anime style stack в конце присутствует? Если нет — добавь.
6. Не описал ли ты что находится внутри `[image N]`? Если да — убери и оставь только action/posture/composition.
7. Не добавил ли новых объектов которых не было в source? Если да — убери.

ФОРМАТ ВЫВОДА:
Одна строка — расширенный prompt на английском. Без префиксов, без префраз, без markdown. Только сама строка.

ПРИМЕР 1 (n_images=3, camera_hint="low angle wide shot"):
source: [image 1] sitting on office chair, looking at [image 3] hologram floating beside him, [image 2] background.
output: low angle wide shot, [image 1] sitting in a relaxed but attentive posture on office chair, hands folded loosely on the desk, head tilted slightly toward [image 3] hologram floating at chest level beside him, faint focused expression with narrowed eyes, [image 2] visible in soft-focus background through floor-to-ceiling windows, framing emphasizing his elevated stance over the viewer, warm evening lighting from a single desk lamp casting directional amber shadows across one side of the face, contemplative and slightly tense atmosphere, anime style, 2D animation, cel-shaded, sharp lineart, vibrant flat colors, modern anime aesthetic

ПРИМЕР 2 (n_images=2, camera_hint="over-the-shoulder medium shot"):
source: [image 1] standing in front of [image 2], speaking with intensity.
output: over-the-shoulder medium shot from behind [image 1], [image 1] standing tall in front of [image 2], one fist clenched at his side, the other hand raised palm-up in a commanding gesture, mouth open in mid-speech, sharp determined gaze fixed forward, dramatic backlit composition with [image 2] glowing softly in the distance, cool blue rim light tracing his silhouette against warm interior tones, intense and resolute atmosphere, anime style, 2D animation, cel-shaded, sharp lineart, vibrant flat colors, modern anime aesthetic

ПРИМЕР 3 (n_images=1, camera_hint=""):
source: [image 1] alone, contemplating recent events.
output: three-quarter side view medium close-up, [image 1] alone in quiet contemplation, seated on the edge of a windowsill, one knee drawn up with arm resting on it, gaze fixed on something far in the distance, faint melancholic expression with downturned mouth and lowered eyelids, soft window light casting gentle shadows across the face, cool muted blue tones in the background, introspective and lonely atmosphere, anime style, 2D animation, cel-shaded, sharp lineart, vibrant flat colors, modern anime aesthetic

ПРИМЕР 4 (n_images=3, camera_hint="dutch angle medium shot"; payload содержит continuity hint про [image 3]):
source: [image 1] standing in [image 2], looking determined.
payload (после source): "with the character's pose and ongoing action consistent with [image 3], rendered from the camera angle described above"
output: dutch angle medium shot, [image 1] standing in [image 2] with weight shifted onto the back foot, shoulders squared, jaw set firm, sharp focused gaze locked forward, the tilted horizon adding visual tension, mid-ground depth with [image 2] architecture flanking the frame, hard side lighting from a low source carving deep shadows across the face, charged and confrontational atmosphere, with the character's pose and ongoing action consistent with [image 3], rendered from the camera angle described above, anime style, 2D animation, cel-shaded, sharp lineart, vibrant flat colors, modern anime aesthetic
"""


# Список ракурсов/кадрирований для детерминистичной ротации между сценами
# одного thread'а. Перебор по `scene_index_in_thread % len(CAMERA_PRESETS)`,
# так что N-я сцена thread'а получает CAMERA_PRESETS[N % len].
# Расширяй смело - чем длиннее список, тем разнообразнее картинка между
# многосценовыми thread'ами.
CAMERA_PRESETS: list[str] = [
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


def pick_camera_preset(scene_index_in_thread: int) -> str:
    """Детерминистичный выбор ракурса по индексу сцены в thread'е."""
    if scene_index_in_thread < 0:
        scene_index_in_thread = 0
    return CAMERA_PRESETS[scene_index_in_thread % len(CAMERA_PRESETS)]


# Якорные фразы которые мы хотим видеть в финальном expanded promtе
# (если их нет - значит Gemma не дописала style stack).
_REQUIRED_STYLE_TOKENS = (
    "anime style",
    "cel-shaded",
)


_IMAGE_MARKER_RE = re.compile(r"\[image\s+(\d+)\]", re.IGNORECASE)


def _extract_image_markers(s: str) -> list[str]:
    """Все [image N] подстроки в порядке появления (нормализованные)."""
    return [f"[image {m.group(1)}]" for m in _IMAGE_MARKER_RE.finditer(s)]


def substitute_entities_to_image_slots(
    image_prompt: str,
    entities: list[str],
) -> str:
    """
    Детерминистичная подмена `[base_name]` -> `[image N]` в image_prompt сцены.

    Параметры:
      image_prompt - сырой image_prompt из scenes[i] (с [base_name] маркерами).
      entities     - упорядоченный список base_name из scene.entities,
                     длина 0..3. Первый entity → [image 1], второй → [image 2],
                     третий → [image 3].

    Порядок entities определяет какая reference картинка попадёт в какой slot
    Qwen Edit'а. Caller должен передавать image_paths в той же
    последовательности что entities.
    """
    if len(entities) > 3:
        raise ValueError(
            f"Qwen Edit поддерживает max 3 image slots, got {len(entities)}"
        )
    out = image_prompt
    for i, base_name in enumerate(entities):
        out = out.replace(f"[{base_name}]", f"[image {i + 1}]")
    return out


async def expand_image_prompt_for_qwen_edit(
    source_with_markers: str,
    n_images: int,
    previous_recap: str = "",
    camera_hint: str = "",
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    min_expansion_ratio: float = 1.4,
) -> str:
    """
    Расширяет source prompt (уже с [image N] маркерами) композицией +
    освещением + атмосферой через Gemma. Возвращает одну строку готовую к
    подаче в TextEncodeQwenImageEditPlus (positive prompt, node 177).

    Параметры:
      source_with_markers - prompt с маркерами [image 1] / [image 2] / [image 3]
                            (используй substitute_entities_to_image_slots).
      n_images            - количество reference картинок (1..3). Маркеры
                            с N > n_images в output появляться не должны.
      previous_recap      - russian-summary для context'а continuity (не
                            утечёт в output).
      camera_hint         - принудительный ракурс/кадрирование, например
                            `"low angle wide shot"`. Если задан, output
                            ОБЯЗАН начинаться с этой фразы. Используй
                            `pick_camera_preset(scene_index_in_thread)` чтобы
                            детерминистично ротировать ракурсы между сценами
                            одного thread'а.
      min_expansion_ratio - output должен быть длиннее source хотя бы в это
                            кол-во раз (по символам). Если короче - retry.

    Возвращает:
      Строку - expanded prompt на английском.

    Negative prompt не возвращается - он хардкоден в qwen_image.json node 175.
    """
    if not 1 <= n_images <= 3:
        raise ValueError(f"n_images must be 1..3, got {n_images}")

    source = source_with_markers.strip()
    if not source:
        raise ValueError("source_with_markers is empty")

    # В source НЕ должно быть [image N] с N > n_images.
    src_markers = _extract_image_markers(source)
    bad_in_src: list[str] = []
    for m in src_markers:
        digits = re.search(r"\d+", m)
        if digits and int(digits.group()) > n_images:
            bad_in_src.append(m)
    if bad_in_src:
        raise ValueError(
            f"source_with_markers содержит [image N] с N > n_images={n_images}: "
            f"{bad_in_src}"
        )

    expected = {f"[image {i + 1}]" for i in range(n_images)}
    forbidden = {f"[image {n}]" for n in range(n_images + 1, 4)}

    recap_block = (
        f"## Story so far (use silently for continuity, do not narrate)\n"
        f"{previous_recap.strip()}\n\n"
        if previous_recap and previous_recap.strip()
        else ""
    )

    camera_hint_clean = camera_hint.strip()
    camera_block = (
        f"## camera_hint (REQUIRED — must appear verbatim as the first phrase of output)\n"
        f"{camera_hint_clean}\n\n"
        if camera_hint_clean
        else ""
    )

    user_payload = (
        recap_block
        + camera_block
        + f"## n_images\n{n_images}\n\n"
        + f"## source_prompt (preserve every [image N] marker verbatim)\n"
        + f"{source}\n\n"
        + f"## Task\n"
        + f"Return ONE expanded prompt string in English. No JSON, no markdown, "
        + f"no labels, no quotes around it. Just the prompt itself."
        + (
            f"\n\nReminder: output MUST start with the camera_hint verbatim "
            f"(`{camera_hint_clean}`) before anything else."
            if camera_hint_clean
            else ""
        )
    )

    retry_profiles = [
        {"temperature": 0.55, "top_p": 0.92},
        {"temperature": 0.40, "top_p": 0.95},
        {"temperature": 0.25, "top_p": 0.98},
    ]
    last_err: Exception | None = None
    last_raw: str = ""

    timeout = aiohttp.ClientTimeout(total=None)
    for attempt, profile in enumerate(retry_profiles):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {
                "model": model,
                "system": SYSTEM_PROMPT_IMAGE,
                "prompt": user_payload,
                "stream": False,
                "keep_alive": 5,
                "options": {
                    "temperature": profile["temperature"],
                    "top_p": profile["top_p"],
                    "repeat_penalty": 1.05,
                    "num_predict": 1536,
                    "seed": random.randint(1, 2**31 - 1),
                    # NB: без format=json и без stop-tokens. Gemma при chat-template
                    # префиксах (типа `<|im_start|>`) может триггерить stop=`<|`
                    # и возвращать пустой output. Сырые leak'и фильтруем ниже.
                },
            }
            async with session.post(ollama_url, json=payload) as response:
                status = response.status
                try:
                    result = await response.json()
                except Exception as e:
                    text = await response.text()
                    last_err = RuntimeError(
                        f"ollama returned non-json (status={status}): {text[:500]!r}; {e}"
                    )
                    print(
                        f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                        f"non-json response status={status}: {text[:200]!r}; retrying"
                    )
                    continue
                last_raw = result.get("response", "").strip() if isinstance(result, dict) else ""

        # Очистка: убираем code fences, обёрточные кавычки, chat-template leak'и.
        # Если Gemma всё-таки вернула JSON {"prompt": "..."} - вытащим строку.
        out = last_raw.strip()
        # chat template artefacts вроде <|im_start|>assistant\n... или <|user|>...
        out = re.sub(r"^<\|[^|]*\|>\s*(?:\w+\s*\n)?", "", out)
        out = re.sub(r"<\|[^|]*\|>\s*$", "", out)
        out = re.sub(r"^```(?:json|text)?\s*", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\s*```$", "", out)
        out = out.strip().strip('"').strip("'").strip()
        if out.startswith("{"):
            try:
                obj = extract_json_object(out)
                if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                    out = obj["prompt"].strip()
            except ValueError:
                pass

        if not out:
            last_err = RuntimeError("empty output from Gemma")
            # Диагностика: печатаем полный result чтобы понять почему пусто
            # (model not found / done_reason=stop_token / done_reason=load / etc).
            ollama_keys = {k: v for k, v in result.items() if k != "context"} if isinstance(result, dict) else result
            print(
                f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                f"empty output (status={status}); ollama result: {ollama_keys!r}; retrying"
            )
            continue

        # 1) все ожидаемые маркеры на месте
        missing = [m for m in expected if m not in out]
        if missing:
            last_err = RuntimeError(f"missing image markers {missing}")
            print(
                f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                f"missing markers {missing}; retrying"
            )
            continue

        # 2) нет лишних маркеров (вне range n_images)
        extra = [m for m in forbidden if m in out]
        if extra:
            last_err = RuntimeError(f"forbidden markers in output {extra}")
            print(
                f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                f"forbidden markers {extra}; retrying"
            )
            continue

        # 3) длина расширилась
        if len(out) < int(len(source) * min_expansion_ratio):
            last_err = RuntimeError(
                f"output too short ({len(out)} chars vs source {len(source)} * "
                f"{min_expansion_ratio} min)"
            )
            print(
                f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                f"too short ({len(out)} vs source {len(source)}); retrying"
            )
            continue

        # 4) anime style stack
        out_lower = out.lower()
        missing_style = [t for t in _REQUIRED_STYLE_TOKENS if t not in out_lower]
        if missing_style:
            last_err = RuntimeError(f"missing style tokens: {missing_style}")
            print(
                f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                f"missing style {missing_style}; retrying"
            )
            continue

        # 5) camera_hint в начале output (если задан). Допускаем что Gemma
        # может слегка отклониться по регистру/пунктуации, но фраза должна
        # быть среди первых ~20 слов.
        if camera_hint_clean:
            head = " ".join(out.split()[:20]).lower()
            if camera_hint_clean.lower() not in head:
                last_err = RuntimeError(
                    f"camera_hint missing from start: expected `{camera_hint_clean}`, "
                    f"got head=`{head[:120]}`"
                )
                print(
                    f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                    f"camera_hint not at start; retrying"
                )
                continue

        return out

    pprint.pprint(("RAW qwen_expand", last_raw))
    raise RuntimeError(
        f"Gemma failed to produce valid expanded prompt after "
        f"{len(retry_profiles)} attempts: {last_err}"
    )


def patch_qwen_edit_workflow(
    workflow: dict,
    image_paths: list[str],
    positive_prompt: str,
    filename_prefix: str,
    seed: int,
    negative_prompt: str | None = None,
    enable_lightning_4step: bool = True,
) -> dict:
    """
    Патчит распарсенный qwen_image.json под конкретную сцену. Возвращает
    deep-копию, оригинал не мутируется.

    Параметры:
      workflow                - распарсенный qwen_image.json (dict).
      image_paths             - 1..3 абсолютных пути в той же последовательности
                                что [image N] маркеры в positive_prompt.
                                  image_paths[0] -> image1 (через resize, node 199 ← 202)
                                  image_paths[1] -> image2 (node 200)
                                  image_paths[2] -> image3 (node 201)
      positive_prompt         - расширенный prompt с [image N] маркерами,
                                идёт в node 177 (TextEncodeQwenImageEditPlus).
      filename_prefix         - префикс выходного файла, например "image_gen/3".
      seed                    - seed для KSampler (node 191).
      negative_prompt         - если None, оставляем хардкод в node 175.
                                Передавай только если нужно override'нуть.
      enable_lightning_4step  - True использует Lightning 4-step LoRA (4 шага),
                                False - 8 шагов без LoRA. По умолчанию True.

    Логика отрубания неиспользуемых image slot'ов:
      Если image_paths короче 3, удаляем ключи image2/image3 из inputs обоих
      TextEncodeQwenImageEditPlus нод (175 negative и 177 positive). Без этих
      ключей нода работает на меньшем количестве reference картинок.
    """
    wf = copy.deepcopy(workflow)
    n = len(image_paths)
    if not 1 <= n <= 3:
        raise ValueError(f"image_paths must have 1..3 entries, got {n}")

    # image1 идёт через resize node 199 ← node 202
    wf["202"]["inputs"]["image"] = image_paths[0]
    if n >= 2:
        wf["200"]["inputs"]["image"] = image_paths[1]
    if n >= 3:
        wf["201"]["inputs"]["image"] = image_paths[2]

    # Отрубить неиспользуемые слоты в обоих encoder'ах
    for enc_id in ("175", "177"):
        if n < 2:
            wf[enc_id]["inputs"].pop("image2", None)
        if n < 3:
            wf[enc_id]["inputs"].pop("image3", None)

    # Промпты
    wf["177"]["inputs"]["prompt"] = positive_prompt
    if negative_prompt is not None:
        wf["175"]["inputs"]["prompt"] = negative_prompt

    # Lightning toggle
    wf["188"]["inputs"]["value"] = bool(enable_lightning_4step)

    # Filename + seed
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix
    wf["191"]["inputs"]["seed"] = int(seed)

    return wf


# ============================================================ batch precompute
# Все Ollama-вызовы делаем за один проход ДО загрузки тяжёлых ComfyUI моделей
# (Qwen Edit ~9GB, Wan 2.2 14B*2). Иначе Comfy выжирает RAM/VRAM и Ollama при
# следующем вызове падает с "model requires more system memory".
#
# После batch'а вызови `unload_ollama_model(model)` чтобы вынести Gemma из RAM
# и освободить место под Comfy.


async def precompute_all_image_prompts(
    scenes: list[dict],
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    min_expansion_ratio: float = 1.4,
    rotate_cameras: bool = True,
) -> dict[int, str]:
    """
    Прогоняет ВСЕ image_prompt'ы сцен через Gemma за один проход.

    Делает per-сцену:
      1. substitute_entities_to_image_slots(scene.image_prompt, scene.entities)
      2. Считает scene_index_in_thread (счётчик сцен внутри thread_id'а в
         порядке появления). Для первой сцены boss_office индекс = 0,
         для второй = 1, и т.д. Cuts на другие thread'ы счётчик не трогают,
         так что возврат к thread'у продолжает счёт.
      3. Если rotate_cameras=True — берёт `pick_camera_preset(idx)` как
         camera_hint и передаёт в expand. Это гарантирует разные ракурсы
         между сценами одного thread'а (low angle → high angle → close-up
         → over-the-shoulder → ...). Если False — camera_hint пустой,
         Gemma выбирает сама.
      4. expand_image_prompt_for_qwen_edit(...)

    Возвращает {scene_id: expanded_prompt}. Сцены без entities скипает
    (туда expand-логика не применима; в этих сценах image_prompt пойдёт
    напрямую без расширения, либо обрабатывай руками).

    Вызывать ДО загрузки Qwen Edit / Wan моделей в ComfyUI, иначе Ollama
    упрётся в OOM по системной памяти.
    """
    out: dict[int, str] = {}
    total = len(scenes)
    thread_counter: dict[str, int] = {}

    for idx, scene in enumerate(scenes, 1):
        sid = scene["scene_id"]
        entities = scene.get("entities", [])
        thread_id = scene.get("thread_id", f"_scene_{sid}")

        # Считаем индекс сцены В этом thread'е (0-based)
        scene_index_in_thread = thread_counter.get(thread_id, 0)
        thread_counter[thread_id] = scene_index_in_thread + 1

        if not entities:
            print(
                f"[precompute_image] [{idx}/{total}] scene {sid} no entities, "
                f"skipping expand (caller will use raw image_prompt)"
            )
            continue
        if len(entities) > 3:
            raise ValueError(
                f"scene {sid} has {len(entities)} entities, max 3 supported. "
                f"Update GROK plan to split scene or reduce entities."
            )

        camera_hint = (
            pick_camera_preset(scene_index_in_thread) if rotate_cameras else ""
        )

        src = substitute_entities_to_image_slots(scene["image_prompt"], entities)
        expanded = await expand_image_prompt_for_qwen_edit(
            source_with_markers=src,
            n_images=len(entities),
            previous_recap=scene.get("previous_recap", ""),
            camera_hint=camera_hint,
            ollama_url=ollama_url,
            model=model,
            min_expansion_ratio=min_expansion_ratio,
        )
        out[sid] = expanded
        print(
            f"[precompute_image] [{idx}/{total}] scene {sid} ok, "
            f"{len(expanded)} chars, n_images={len(entities)}, "
            f"thread={thread_id}#{scene_index_in_thread}, "
            f"camera={camera_hint or 'auto'}"
        )
    return out


async def precompute_all_chunk_prompts(
    scenes: list[dict],
    chunk_frames_per_scene: dict[int, Sequence[int]],
    fps: int = 15,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
) -> dict[int, list[str]]:
    """
    Прогоняет get_chunk_prompts_from_ollama для всех сцен в batch'е.

    NB: image_path=None всегда — на этом этапе scene-картинка ещё не сгенерена
    (Qwen Edit run'ится позже). Gemma пишет chunk-промпты только по
    seed video_prompt'у + scene_text. На практике для distilled Wan этого
    достаточно — init-картинка влияет на чанки косвенно через i2v conditioning,
    а семантика motion'а описывается seed'ом.

    Параметры:
      chunk_frames_per_scene - {scene_id: [81, 81, 33]} список чанков на сцену
                               (получаемый твоим split_into_chunks() из расчёта
                               по аудио длительности).

    Возвращает {scene_id: [chunk0, chunk1, ...]}.
    """
    out: dict[int, list[str]] = {}
    total = len(scenes)
    for idx, scene in enumerate(scenes, 1):
        sid = scene["scene_id"]
        chunks = list(chunk_frames_per_scene[sid])
        prompts = await get_chunk_prompts_from_ollama(
            scene_text=scene["text_scene"],
            video_prompt=scene["video_prompt"],
            image_path=None,
            n_chunks=len(chunks),
            chunk_frames=chunks,
            fps=fps,
            ollama_url=ollama_url,
            model=model,
            previous_recap=scene.get("previous_recap", ""),
        )
        out[sid] = prompts
        print(
            f"[precompute_chunks] [{idx}/{total}] scene {sid} ok, "
            f"{len(prompts)} chunks"
        )
    return out


async def unload_ollama_model(
    model: str,
    ollama_url: str = "http://localhost:11434/api/generate",
) -> None:
    """
    Выгружает модель из памяти Ollama (`keep_alive=0` + пустой prompt).

    Вызывай ПОСЛЕ batch'а Ollama-вызовов и ДО загрузки тяжёлых ComfyUI
    моделей. Освобождает 4-9GB RAM (зависит от модели).

    Не бросает исключения: если Ollama не отвечает, просто логирует.
    """
    timeout = aiohttp.ClientTimeout(total=30)
    payload = {
        "model": model,
        "prompt": "",
        "keep_alive": 0,
        "stream": False,
    }
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(ollama_url, json=payload) as r:
                if r.status >= 400:
                    body = await r.text()
                    print(
                        f"[unload_ollama_model] non-2xx status={r.status}: "
                        f"{body[:200]!r}"
                    )
                else:
                    print(f"[unload_ollama_model] {model} unloaded (keep_alive=0)")
    except Exception as e:
        print(f"[unload_ollama_model] failed: {e}")
