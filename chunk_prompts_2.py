"""
Промпты для пайплайна:

  - Wan 2.2 i2v (lightx2v / SVI v2 продолжение): get_chunk_prompts_from_ollama
    Сцена длиной N кадров режется на M чанков по ~81 кадру (5с при 15fps).
    Ollama за ОДИН вызов отдаёт массив из M chunk-промптов.

  - LTX 2.3 22B i2v (single-pass): get_video_prompt_for_ltx_from_ollama
    Сцена генерится одним проходом (default 121 кадр @24fps ≈ 5с).
    Ollama отдаёт ОДИН flowing-paragraph prompt по гайдлайну Lightricks
    (https://docs.ltx.video/api-documentation/prompting-guide):
    establish-shot → scene → action → character → camera → ambient sound.

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


# LTX 2.3 VAE temporal stride = 8, валидные длины N = 8k + 1.
LTX_VAE_TEMPORAL = 8
LTX_DEFAULT_FRAMES = 121         # 5с @ 24fps - стандарт single-pass scene


def round_to_valid_frames_ltx(n: int, mode: str = "nearest") -> int:
    """
    Приводит число кадров к валидному для LTX 2.3 VAE: N = 8k + 1
    (LTX-Video VAE имеет temporal compression 8x).

    Параметры:
      n    - желаемое число кадров (любое >= 1).
      mode - стратегия округления:
        "nearest" - ближайшее валидное (default). При равной разнице
                    округляет ВНИЗ (полу-к-чётному не нужно: 8k+1 уже
                    распределены равномерно).
        "up"      - ближайшее ВВЕРХ валидное N >= n. Безопаснее для
                    TTS-синхронизированных сцен: ffmpeg `-t audio_duration`
                    обрежет хвост, лишних кадров не останется.
        "down"    - ближайшее ВНИЗ валидное N <= n. Экономит чуть-чуть
                    VRAM/времени, но видео может оказаться короче аудио;
                    хвост дополняется через `tpad=stop_mode=clone`
                    (заморозка последнего кадра).

    Минимум возвращаемого значения = 1 (8*0 + 1). Никогда не возвращает 0.

    Примеры:
      round_to_valid_frames_ltx(120)         -> 121 (nearest: 121 ближе чем 113)
      round_to_valid_frames_ltx(120, "down") -> 113
      round_to_valid_frames_ltx(120, "up")   -> 121
      round_to_valid_frames_ltx(125, "nearest") -> 121 (ближе 121, чем 129)
      round_to_valid_frames_ltx(126, "nearest") -> 129 (ближе 129, чем 121)
      round_to_valid_frames_ltx(0)           -> 1
      round_to_valid_frames_ltx(1)           -> 1
      round_to_valid_frames_ltx(8)           -> 9 (nearest: 9 ближе чем 1)
    """
    n = max(1, int(n))
    floor_v = ((n - 1) // LTX_VAE_TEMPORAL) * LTX_VAE_TEMPORAL + 1
    if mode == "down":
        return floor_v
    if mode == "up":
        return floor_v if floor_v == n else floor_v + LTX_VAE_TEMPORAL
    if mode == "nearest":
        ceil_v = floor_v + LTX_VAE_TEMPORAL
        # При равенстве округляем вниз (детерминированный tie-break).
        return floor_v if (n - floor_v) <= (ceil_v - n) else ceil_v
    raise ValueError(f"round_to_valid_frames_ltx: unknown mode {mode!r}")


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


SYSTEM_PROMPT_LTX = """## Role
You are a creative cinematographer writing a single-paragraph video prompt for
**LTX-2.3 22B (i2v, 24 fps)**. The whole scene is generated in ONE pass —
you produce **exactly one prompt** that brings the still image to life.

The user gives a SEED prompt (a few words sketching the scene) plus the
reference still that will be the first frame. Your job is NOT to copy the
seed. Your job is to invent rich, specific motion: micro-actions of
characters, environmental effects, prop interactions, lighting shifts, and
continuous camera language. The seed is a starting hint; you are the
director who decides everything that happens on screen.

## Output structure (LTX-2.3 prompting guide order)
Single flowing paragraph that covers, in this order, in natural prose:
  1. **Establish the shot** — cinematography term + shot scale (e.g. "a slow
     wide establishing shot", "an intimate over-the-shoulder medium shot",
     "a tight handheld close-up").
  2. **Set the scene** — lighting (rim light, golden hour, neon glow,
     flickering candle), color palette (cool cyan, warm amber, monochrome),
     surface textures, atmosphere (mist, dust motes, drifting embers).
  3. **Describe the action** — the motion as a natural sequence beginning
     to end. What starts, what develops, what concludes by the last frame.
  4. **Define the character(s)** — age, hairstyle, clothing, distinguishing
     visual features. Express emotion through PHYSICAL cues only (sagging
     shoulders, narrowed eyes, fingers curling).
  5. **Camera movement** — explicit and specific ("the camera slowly pushes
     in", "a gentle handheld drift to the right", "a low-angle dolly that
     rises into a crane shot"). Describe how subjects appear after the
     camera completes the move so the model can resolve the motion.
  6. **Ambient sound cue** — short environmental foley only (wind through
     cloth, distant footsteps, room tone, rain on stone, fabric rustle).
     This conditions the visual model; the actual audio track is replaced
     in post-production.

## Style — Pixar 3D CGI toon (P1x4r LoRA)
- Aesthetic is **stylized 3D CGI in Pixar / DreamWorks animated film
  style**, NOT photorealistic, NOT live-action footage, NOT 2D
  hand-drawn, NOT cel-shaded anime, NOT a photograph. Use phrasing
  like "Pixar-style stylized 3D rendering", "smooth PBR surfaces",
  "soft volumetric lighting", "subsurface scattering on skin",
  "exaggerated cartoon proportions with appealing 3D character design",
  "high-end animated film cinematography".
- Match level of detail to shot scale: close-ups need more facial /
  fabric / lighting detail than wide shots.
- Camera language should be relative to the subject (push-in, track,
  pan, orbit, tilt) — avoid abstract labels like "cinematic" alone.

## Forbidden
- **No proper names of any kind** — no character names, no personal
  names, no city/country/region names, no brand names. The reference
  context may contain names like "Lin Shuang", "Shen Fei", "Chang'an"
  etc. — these are for YOUR understanding only. In the output prompt,
  refer to characters ONLY by visual description: "the young woman
  with dark hair", "the white-haired man in tactical gear", "the
  taller assassin", etc. LTX has no concept of named identities; it
  only sees shapes, colors, and clothing.
- **No internal emotional state words** ("she feels sad", "he is angry",
  "she is confused"). Use physical cues instead.
- **No spoken dialogue** of any kind. NO quoted text. NO "she says ...".
  The TTS narrator track is mixed in post; LTX-generated speech would
  fight against it.
- **No music descriptions** ("orchestral score", "epic soundtrack"). The
  background score track is mixed in post.
- **No on-screen text, logos, captions, subtitles, signs with words**.
- **No meta references** ("frame", "second", "scene", "chunk", "video",
  "LTX", "AI", "model").
- **No photorealistic / live-action / photographic / real-footage descriptors**.
- **No 2D / hand-drawn / anime / manga / cel-shaded / lineart / line-art /
  watercolor / sketch / ink-drawing / comic-book descriptors**. The output
  must read as stylized 3D CGI, not 2D animation.
- No complex chaotic physics (water splashes with thousands of droplets,
  glass shattering into hundreds of shards). Stay clean and animatable.

## Continuity (Story so far)
The user MAY provide a `Story so far` block in Russian summarizing
previous scenes of the same video. Use it as background context to keep
the character's emotional posture, costume state, location, and lighting
register consistent with what was just happening. Do NOT recap that
material in your output — describe ONLY what happens in this scene.

## Length & form
- ONE single continuous paragraph. No bullets, no line breaks, no
  markdown.
- **120–180 words**, roughly 5–8 sentences.
- Present tense, active verbs ("she tilts her head", "the lantern
  flickers", "dust drifts across the beam of light").

## Output
Return STRICT JSON:
```
{"prompt": "<single English paragraph>"}
```
Exactly one key `prompt`, value is one string. No extra fields, no
markdown fences, no commentary outside the JSON."""


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

LLAMA_URL = "http://localhost:8080/v1/chat/completions"

async def call_llama_server(
    *,
    system: str,
    user: str,
    images: list[str] | None = None,  # base64-encoded картинки (PNG)
    temperature: float,
    top_p: float,
    max_tokens: int = 8192,
    repeat_penalty: float = 1.15,
    stop: list[str] | None = None,
    seed: int | None = None,
    use_json_format: bool = True,
    enable_thinking: bool = False,
    server_url: str = LLAMA_URL,
    timeout: aiohttp.ClientTimeout | None = None,
) -> tuple[str, dict]:
    """
    Универсальный клиент для llama.cpp `llama-server` (OpenAI-compat
    /v1/chat/completions). Заменяет Ollama /api/generate.

    Возвращает:
        (raw_content, full_result) — content уже взят из
        choices[0].message.content. Если ответ битый - content="".

    Параметры:
      system / user        - текстовые сообщения. Картинки больше не
                             поддерживаются (этот сервер запущен на
                             text-only модели Qwen3.6-35B-A3B без mmproj).
      use_json_format      - True ставит response_format=json_object,
                             llama.cpp форсит модель закрывать корректным
                             JSON через grammar-constrained декодинг.
                             Аналог Ollama `format: "json"`.
      enable_thinking      - False для Qwen3-моделей вырубает <think>
                             через chat_template_kwargs. Аналог
                             Ollama `think: false`.
      stop                 - стопы. Default: ["<|im_end|>"] - end-of-turn
                             маркер Qwen3.
      seed                 - default random per call.
    """
    if stop is None:
        stop = ["<|im_end|>"]
    if seed is None:
        seed = random.randint(1, 2**31 - 1)
    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=None)

    if images:
        # OpenAI vision-format: content становится массивом частей.
        # llama-server при наличии --mmproj разбирает image_url с data: URI
        # как вход для vision-tower. Изображения декодируются через clip
        # encoder и приклеиваются к user-сообщению как vision-токены.
        user_content: list[dict] = [{"type": "text", "text": user}]
        for img_b64 in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    body: dict = {
        "model": "loaded",
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "seed": seed,
        "stop": stop,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if use_json_format:
        body["response_format"] = {"type": "json_object"}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(server_url, json=body) as response:
            result = await response.json()

    try:
        content = result["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return "", result if isinstance(result, dict) else {}

    return content.strip(), result

async def get_chunk_prompts_from_ollama(
    scene_text: str,
    video_prompt: str,
    image_path: str | None,  # подаётся в vision-tower через --mmproj
    n_chunks: int,
    chunk_frames: Sequence[int],
    fps: int = 15,
    ollama_url: str = LLAMA_URL,
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

    retry_profiles = [
        {"temperature": 0.95, "top_p": 0.85},
        {"temperature": 0.80, "top_p": 0.90},
        {"temperature": 0.60, "top_p": 0.95},
    ]
    last_err: Exception | None = None
    last_raw: str = ""

    for attempt, profile in enumerate(retry_profiles):
        last_raw, result = await call_llama_server(
            system=SYSTEM_PROMPT_WAN.format(n_chunks=n_chunks),
            user=user_payload,
            images=images_payload or None,
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            max_tokens=8192,
            stop=["<end_of_turn>"],
            use_json_format=True,
            enable_thinking=True,
            server_url=ollama_url,
        )

        # Стрип <think>...</think> блоков на случай если Ollama льёт CoT
        # прямо в response (старые версии до отдельного поля `thinking`).
        last_raw = re.sub(r"<think\b[^>]*>.*?</think>", "", last_raw, flags=re.DOTALL | re.IGNORECASE).strip()

        # Diagnostic: если response пуст и done_reason=length при наличии
        # thinking - значит thinking-модель не успела до ответа дойти.
        if not last_raw and isinstance(result, dict):
            try:
                finish_reason = result["choices"][0]["finish_reason"]
            except (KeyError, IndexError, TypeError):
                finish_reason = None
            if finish_reason == "length":
                print(
                    f"[chunk_prompts] attempt {attempt + 1}/{len(retry_profiles)} "
                    f"truncated by max_tokens. Increase max_tokens (currently 8192) "
                    f"или закрути thinking=False."
                )

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


# --------------------------------------------------------------- LTX 2.3 i2v


async def get_video_prompt_for_ltx_from_ollama(
    scene_text: str,
    video_prompt: str,
    image_path: str | None,
    n_frames: int,
    fps: int = 24,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    previous_recap: str = "",
    continuity_type: str = "new_cut",  # NEW
) -> str:
    """
    Возвращает ОДНУ строку — prompt для LTX 2.3 i2v на всю сцену.

    LTX 2.3 генерит сцену single-pass (в отличие от Wan i2v, где сцена
    режется на чанки), поэтому возвращается ровно один промпт.
    Структура промпта следует официальному LTX prompting guide:
    establish-shot → scene → action → character → camera → ambient sound.

    Параметры:
      scene_text     - русский текст озвучки этой сцены (для контекста; в
                       сам prompt не копируется — TTS подставляется отдельно).
      video_prompt   - seed-промпт от GROK (1-2 предложения, английский).
      image_path     - путь к init-картинке (Qwen Edit output для этой сцены).
                       Передаётся в Gemma как image input для лучшего
                       conditioning'а описания motion'а.
      n_frames       - длина сцены в кадрах. Для LTX 2.3 валидно 8k+1
                       (default 121 = 5с @24fps). Используется только для
                       подсказки Gemma о темпе ("у тебя есть N кадров,
                       не забивай слишком много action'а").
      fps            - частота кадров (default 24 для LTX 2.3).
      previous_recap - русский recap предыдущих сцен. Используется как
                       контекст для непрерывности; в output prompt не
                       просачивается.

    Возвращает: str — single flowing paragraph 120-180 слов.

    Бросает RuntimeError если все retry-профили не смогли распарсить ответ.
    """
    duration_s = round(n_frames / fps, 2) if fps > 0 else 0.0

    recap_block = (
        f"## Story so far (для непрерывности, не пересказывать в prompt)\n"
        f"{previous_recap.strip()}\n\n"
        if previous_recap and previous_recap.strip()
        else ""
    )

    # Дополнительная инструкция для continue-сцен
    if continuity_type in ("soft_continue", "hard_continue"):
        continuity_note = (
            "## Critical continuity constraint\n"
            "The reference image is the immediate continuation of the previous "
            "scene. Character pose, expression, position, and ongoing action "
            "MUST be preserved exactly as shown in the reference image. The "
            "first second of motion should evolve smoothly from the static "
            "reference state, not start from a 'rest pose' or 'default' state.\n\n"
        )
        recap_block = recap_block + continuity_note

    user_payload = (
        recap_block
        + "## Scene context\n"
        f"Spoken text (RU; handled by TTS, do NOT include speech in prompt): "
        f"{scene_text}\n\n"
        + "## High-level direction (seed)\n"
        f"{video_prompt}\n\n"
        + "## Scene duration\n"
        f"{n_frames} frames at {fps} fps ≈ {duration_s}s of motion.\n\n"
        + 'Return JSON {"prompt": "<single English paragraph 120-180 words>"} '
        + "following the system rules. The first frame is the attached "
        + "reference image; describe what happens AFTER it, not the image itself."
    )

    images_payload: list[str] = []
    if image_path and os.path.exists(image_path):
        images_payload.append(encode_image_b64(image_path))

    retry_profiles = [
        {"temperature": 0.95, "top_p": 0.85},
        {"temperature": 0.80, "top_p": 0.90},
        {"temperature": 0.60, "top_p": 0.95},
    ]
    last_err: Exception | None = None
    last_raw: str = ""

    for attempt, profile in enumerate(retry_profiles):
        last_raw, result = await call_llama_server(
            system=SYSTEM_PROMPT_LTX,
            user=user_payload,
            images=images_payload or None,
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            max_tokens=32768,
            stop=["<end_of_turn>"],
            use_json_format=True,
            enable_thinking=True,
            server_url=ollama_url,
        )

        # Стрип <think>...</think> блоков (см. WAN-функцию).
        last_raw = re.sub(r"<think\b[^>]*>.*?</think>", "", last_raw, flags=re.DOTALL | re.IGNORECASE).strip()

        # Diagnostic: thinking-truncated case.
        if not last_raw and isinstance(result, dict):
            try:
                finish_reason = result["choices"][0]["finish_reason"]
            except (KeyError, IndexError, TypeError):
                finish_reason = None
            if finish_reason == "length":
                print(
                    f"[ltx_prompt] attempt {attempt + 1}/{len(retry_profiles)} "
                    f"truncated by max_tokens (currently 32768). "
                    f"Подними max_tokens или вырубай thinking=False."
                )

        try:
            parsed = extract_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            print(
                f"[ltx_prompt] attempt {attempt + 1}/{len(retry_profiles)} "
                f"failed to parse JSON: {e}; retrying with safer profile"
            )
            continue

        prompt_field = parsed.get("prompt") if isinstance(parsed, dict) else None
        if not isinstance(prompt_field, str) or not prompt_field.strip():
            last_err = RuntimeError(
                f"Expected non-empty 'prompt' string field, got: {parsed!r}"
            )
            print(
                f"[ltx_prompt] attempt {attempt + 1}/{len(retry_profiles)} "
                f"wrong shape ({last_err}); retrying"
            )
            continue

        return prompt_field.strip()

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


SYSTEM_PROMPT_IMAGE = """Ты — image-prompt expander для Qwen Image Edit 2511 + P1x4r Pixar LoRA.

ЗАДАЧА: взять source prompt и дополнить его композицией, освещением, атмосферой.
Вокабуляр объектов в output ограничен тем что в source — НИЧЕГО НОВОГО.

СТИЛЬ: Pixar 3D CGI toon (LoRA триггер — `P1x4r`). Вывод должен читаться как
кадр из современного Pixar/DreamWorks animated film: стилизованный 3D CGI с
экспрессивными cartoon-пропорциями, мягким volumetric lighting, subsurface
scattering на коже, smooth PBR текстурами. НЕ anime, НЕ 2D, НЕ cel-shaded,
НЕ фотореализм.

`[image N]` — ссылки на reference картинки (уже в Pixar 3D стиле). Не описывай
что ВНУТРИ них (модель их и так видит). Используй как композиционные
якоря в том порядке и в той структуре что заданы в source.

═══ 3 ПРАВИЛА ═══

1. NO NEW OBJECTS. Если в source НЕТ слова — в output НЕТ объекта. Никаких
   weapons / weather (rain, fog, smoke, dust) / particles (embers, sparks) /
   buildings (cityscape, ruins, towers) / vegetation / NPCs / vehicles /
   light sources (torches, lanterns) которых не было в source. Только
   cinematic language (камера, свет-направление, атмосфера, поза).

2. PRESERVE PROPER NOUNS. Каждое имя из `must_preserve_terms` обязано появиться
   в output дословно. Не заменяй "Zhou Kun" на "an unseen threat" или
   "a powerful figure".

3. PRESERVE ACTIONS AND STRUCTURE. "flanking Zhou Kun" → output держит
   "flanking Zhou Kun". "leaning against wall" → "leaning against wall".
   Расширяй ДЕТАЛИ действия (поза, угол тела, gaze, weight, tension),
   не подменяй само действие. Сохраняй тот же порядок и группировку
   `[image N]` маркеров что в source — не переставляй и не разделяй их.

═══ ФОРМАТ ВЫВОДА ═══

- Одна строка на английском. Без markdown, JSON, кавычек, заголовков.
- Output ОБЯЗАН начинаться с LoRA-триггера `P1x4r,` (именно в такой
  регистре — большая P, цифра 1, x, цифра 4, r). Если задан `camera_hint`,
  ставь его СРАЗУ ПОСЛЕ триггера: `P1x4r, <camera_hint>, ...`.
- Output заканчивается обязательным style stack:
  pixar style character, stylized CGI character, pixar-like 3D render, cinematic animated character, high-end animated film style
- Никаких `[bracketed_text]` кроме `[image 1]`, `[image 2]`, `[image 3]`
  (не больше чем n_images).

═══ ПРИМЕР ═══

INPUT:
  camera_hint: "extreme wide establishing shot"
  must_preserve_terms: ["Zhou Kun"]
  source: [image 1], [image 2] and [image 3] flanking Zhou Kun, combat ready poses, dynamic angle.

OUTPUT:
  P1x4r, extreme wide establishing shot, [image 1], [image 2] and [image 3] flanking Zhou Kun in a tight semi-circle formation, combat ready poses with weight low and forward, knees bent and feet planted shoulder-width apart, focused intent in their narrowed eyes locked on Zhou Kun, dynamic diagonal composition pulling the eye toward the central confrontation, dramatic side-lighting carving sharp contrast across PBR-textured faces and clothing, soft volumetric god-rays threading the scene, charged tense atmosphere, pixar style character, stylized CGI character, pixar-like 3D render, cinematic animated character, high-end animated film style

(Заметь: output начинается с `P1x4r,` (LoRA-триггер), затем camera_hint.
Action "flanking Zhou Kun" сохранён. "Zhou Kun" появился дословно. Структура
`[image 1], [image 2] and [image 3]` и порядок не изменены. Ничего не добавлено
сверх source — нет weapons, dust, ruins, smoke. PBR/volumetric — это cinematic
language под Pixar-стиль, не новые объекты.)
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
# (если их нет - значит Gemma не дописала style stack или потеряла
# LoRA-триггер P1x4r). Сравнение идёт в lower-case, поэтому "p1x4r".
_REQUIRED_STYLE_TOKENS = (
    "p1x4r",
    "pixar",
)


_IMAGE_MARKER_RE = re.compile(r"\[image\s+(\d+)\]", re.IGNORECASE)
_BRACKET_TOKEN_RE = re.compile(r"\[([^\[\]]+)\]")


# Регексп для извлечения plain-text proper nouns (имён собственных) из source.
# Ловим последовательности из 1+ Capitalized слов: "Zhou Kun", "King Aldric",
# "Hari", "Mary Smith". Stopword-фильтр отсеивает English функциональные
# слова которые могут оказаться с большой буквы в начале source.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset({
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She",
    "We", "They", "I", "You", "My", "His", "Her", "Their", "Our", "Your",
    "Is", "Was", "Are", "Were", "Be", "Been", "Being", "Has", "Had", "Have",
    "Do", "Does", "Did", "Will", "Would", "Should", "Could", "Can", "May",
    "Might", "Must", "Shall", "Should",
})


def _extract_proper_nouns(source: str) -> list[str]:
    """Извлекает уникальные plain-text proper nouns из source prompt'а.

    Возвращает список фраз (multi-word OK), в порядке появления, без
    дубликатов. Игнорирует фразы целиком из stopword'ов (типа "The").
    Возвращённый список идёт в must_preserve_terms — Gemma обязана
    сохранить каждый элемент в output дословно (substring check
    case-insensitive).

    Примеры:
      "[image 1] flanking Zhou Kun, combat ready" -> ["Zhou Kun"]
      "speaking to King Aldric in [image 2]"      -> ["King Aldric"]
      "[image 1] alone, contemplating events"     -> []
      "Mary holding the red ribbon"               -> ["Mary"]
    """
    found: list[str] = []
    seen: set[str] = set()
    for m in _PROPER_NOUN_RE.finditer(source):
        phrase = m.group(1).strip()
        words = phrase.split()
        if all(w in _PROPER_NOUN_STOPWORDS for w in words):
            continue
        # Если single-word и в stopword'ах - тоже скип
        if len(words) == 1 and words[0] in _PROPER_NOUN_STOPWORDS:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(phrase)
    return found


def _extract_image_markers(s: str) -> list[str]:
    """Все [image N] подстроки в порядке появления (нормализованные)."""
    return [f"[image {m.group(1)}]" for m in _IMAGE_MARKER_RE.finditer(s)]


def _strip_unknown_brackets(text: str) -> tuple[str, list[str]]:
    """
    Удаляет любые `[name]` токены кроме `[image N]`. Возвращает (cleaned, stripped).
    Содержимое скобок остаётся в тексте без скобок (`[wolf]` → `wolf`).
    """
    stripped: list[str] = []

    def repl(m: "re.Match[str]") -> str:
        inner = m.group(1).strip()
        if re.fullmatch(r"image\s+\d+", inner, re.IGNORECASE):
            return m.group(0)
        stripped.append(inner)
        return inner

    cleaned = _BRACKET_TOKEN_RE.sub(repl, text)
    return cleaned, stripped


def resolve_entities_from_image_prompt(
    image_prompt: str,
    base_names: list[str] | set[str] | tuple[str, ...],
) -> list[str]:
    """
    Извлекает entity-маркеры `[base_name]` из image_prompt и пересекает их
    с известными `base_names` из plan["base_prompts"].

    Возвращает упорядоченный список base_name'ов (без дубликатов) в том
    порядке в каком они впервые появились в image_prompt. Усечён до 3
    элементов (hard limit Qwen Edit'а — max 3 image slots).

    Это замена устаревшему `scene["entities"]` полю в GROK plan'е — теперь
    pipeline резолвит entities из текста image_prompt автоматически,
    GROK больше не обязан выписывать их явно. Любые `[bracketed]` маркеры
    которых нет в `base_names` просто игнорируются (они будут позже
    задавлены `_strip_unknown_brackets`).

    Параметры:
      image_prompt - сырой image_prompt из scene с маркерами [base_name].
                     Например:
                       "[main_character] standing in [shop_interior] with
                        [red_ribbon] in hand, dramatic lighting."
      base_names   - итерируемое со всеми объявленными base_name из
                     plan["base_prompts"]. Маркеры не из этого списка
                     игнорируются (например `[wide shot]` — это шум).

    Возвращает:
      Упорядоченный список base_name'ов длины 0..3.

    Пример 1 — нормальный кейс:
      resolve_entities_from_image_prompt(
          "[lin_shuang] and [lin_xue] leaning against [drainage_channel]",
          base_names={"lin_shuang", "lin_xue", "drainage_channel", "red_ribbon"},
      )
      -> ['lin_shuang', 'lin_xue', 'drainage_channel']

    Пример 2 — дубликат маркера в одной сцене:
      resolve_entities_from_image_prompt(
          "[main_character] in [shop] [main_character] holding pill",
          base_names={"main_character", "shop"},
      )
      -> ['main_character', 'shop']  # main_character не задвоен

    Пример 3 — маркер не из base_names игнорируется:
      resolve_entities_from_image_prompt(
          "wide shot of [unknown_entity] near [shop]",
          base_names={"main_character", "shop"},
      )
      -> ['shop']

    Пример 4 — больше 3 валидных маркеров → берём первые 3:
      resolve_entities_from_image_prompt(
          "[a] sees [b] beside [c] looking at [d]",
          base_names={"a", "b", "c", "d"},
      )
      -> ['a', 'b', 'c']
    """
    base_set = set(base_names)
    seen: set[str] = set()
    out: list[str] = []
    for m in _BRACKET_TOKEN_RE.finditer(image_prompt):
        inner = m.group(1).strip()
        # Скип `[image N]` мета-маркеры — это нумерованные слоты Qwen Edit'а,
        # они появляются после substitute_entities_to_image_slots, не раньше.
        if re.fullmatch(r"image\s+\d+", inner, re.IGNORECASE):
            continue
        if inner in base_set and inner not in seen:
            seen.add(inner)
            out.append(inner)
            if len(out) >= 3:
                break
    return out


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


def prepare_image_prompt_for_qwen_edit(
    image_prompt: str,
    entities: list[str],
    scene_id: int | str = "?",
) -> tuple[str, list[str]]:
    """
    Готовит scene image_prompt к Qwen Edit с защитой от багов GROK plan'а.

    Шаги:
      1. substitute_entities_to_image_slots: [base_name] → [image N].
      2. _strip_unknown_brackets: любые оставшиеся [name] которых нет в
         entities[] — это GROK забыл их объявить. Снимаем скобки → plain text
         (Gemma не будет цепляться за них).
      3. Прунинг entities[] до тех что РЕАЛЬНО упомянуты в image_prompt:
         если GROK объявил entity но не использовал её в [base_name] —
         выбрасываем из списка. Иначе будет [image 3] которое Gemma не
         увидит в source и не вставит в output → validation fails.
      4. Перенумеровка [image N] чтобы шли 1..k без пропусков (если
         entity '2' выкинута — [image 3] становится [image 2]).

    Возвращает (cleaned_source, used_entities) где used_entities ⊆ entities
    в исходном порядке. Caller строит image_paths по used_entities.

    Печатает WARNING'и для всех найденных проблем — не фейлит pipeline,
    но пользователь видит что GROK plan нужно подправить.
    """
    src = substitute_entities_to_image_slots(image_prompt, entities)

    # 1. Стрип `[unknown_name]` (entity не объявлена в entities[])
    src, leftover = _strip_unknown_brackets(src)
    if leftover:
        print(
            f"[prepare_image_prompt] scene {scene_id} WARNING: stripped "
            f"unknown brackets {leftover} — they are missing from entities[]. "
            f"GROK plan needs fixing. Stripped to plain text for now."
        )

    # 2. Прунинг unused entities (объявлены но не упомянуты)
    present_indices = {int(m.group(1)) for m in _IMAGE_MARKER_RE.finditer(src)}
    used_entities: list[str] = []
    index_remap: dict[int, int] = {}
    new_idx = 1
    unused_entities: list[str] = []
    for old_idx, ent in enumerate(entities, start=1):
        if old_idx in present_indices:
            used_entities.append(ent)
            index_remap[old_idx] = new_idx
            new_idx += 1
        else:
            unused_entities.append(ent)

    if unused_entities:
        print(
            f"[prepare_image_prompt] scene {scene_id} WARNING: entities "
            f"{unused_entities} declared but never referenced in image_prompt; "
            f"pruning. Used: {used_entities}."
        )

    # 3. Перенумеровка [image N] под новый порядок
    if index_remap and any(o != n for o, n in index_remap.items()):
        def remap_marker(m: "re.Match[str]") -> str:
            old = int(m.group(1))
            new = index_remap.get(old)
            return f"[image {new}]" if new is not None else m.group(0)

        src = _IMAGE_MARKER_RE.sub(remap_marker, src)

    return src, used_entities


async def expand_image_prompt_for_qwen_edit(
    source_with_markers: str,
    n_images: int,
    camera_hint: str = "",
    entities: list[str] | None = None,  # DEPRECATED: игнорируется, оставлен ради сигнатуры
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    min_expansion_ratio: float = 1.4,
    previous_recap: str = "",  # DEPRECATED: игнорируется, оставлен ради сигнатуры
) -> str:
    """
    Расширяет source prompt (уже с [image N] маркерами) композицией +
    освещением + атмосферой через Gemma. Возвращает одну строку готовую к
    подаче в TextEncodeQwenImageEditPlus (positive prompt, node 177).

    Каждая сцена обрабатывается изолированно — контекст предыдущих сцен
    в Gemma НЕ передаётся (до v2 был параметр previous_recap, теперь
    deprecated; принудительная continuity через cross-scene recap давала
    хуже результат, чем чистые сцены без shared context).

    Параметры:
      source_with_markers - prompt с маркерами [image 1] / [image 2] / [image 3]
                            (используй substitute_entities_to_image_slots).
      n_images            - количество reference картинок (1..3). Маркеры
                            с N > n_images в output появляться не должны.
      camera_hint         - принудительный ракурс/кадрирование, например
                            `"low angle wide shot"`. Если задан, output
                            ОБЯЗАН начинаться с `P1x4r, <camera_hint>, ...`
                            — сначала LoRA-триггер, затем ракурс. Используй
                            `pick_camera_preset(idx)` чтобы детерминистично
                            ротировать ракурсы между сценами.
      min_expansion_ratio - output должен быть длиннее source хотя бы в это
                            кол-во раз (по символам). Если короче - retry.
      entities            - DEPRECATED. Раньше использовался для построения
                            entity_roles секции payload'а (CHARACTER/LOCATION/
                            OBJECT mapping). Сейчас аргумент молча игнорируется
                            — Gemma определяет роль `[image N]` по тексту
                            source'а сама. Аргумент оставлен в сигнатуре ради
                            backward-compat с уже написанным caller'ом.
      previous_recap      - DEPRECATED. Раньше передавал short russian-summary
                            предыдущих сцен thread'а как continuity context.
                            Сейчас аргумент молча игнорируется (оставлен ради
                            обратной совместимости). НЕ используй в новом
                            коде — каждая сцена обрабатывается изолированно.

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

    # previous_recap больше не подмешивается в инпут Gemma (cross-scene
    # контекст оказался вредным для image-gen). Аргумент сохранён
    # в сигнатуре ради backward-compat, но сюда не пробрасывается.
    _ = previous_recap

    camera_hint_clean = camera_hint.strip()

    # `entities` больше не используется внутри payload'а (раньше шёл
    # CHARACTER/LOCATION/OBJECT mapping, оказался вреден — Gemma тонула
    # в ярлыках и керамическом списке keyword'ов). Аргумент остался в
    # сигнатуре ради backward-compat; сюда не пробрасывается.
    _ = entities

    # must_preserve_terms — первой секцией, чтобы Gemma не теряла имена
    # собственные (Zhou Kun, King Aldric и т.п.) в output'е.
    proper_nouns = _extract_proper_nouns(source)
    must_preserve_block = (
        "must_preserve_terms (verbatim in output):\n"
        + "\n".join(f"  - \"{p}\"" for p in proper_nouns)
        + "\n\n"
        if proper_nouns
        else ""
    )

    camera_block = (
        f"camera_hint (must appear verbatim immediately after the `P1x4r,` "
        f"LoRA trigger at the start of output): {camera_hint_clean}\n\n"
        if camera_hint_clean
        else ""
    )

    user_payload = (
        must_preserve_block
        + camera_block
        + f"n_images: {n_images}\n\n"
        + f"source_prompt:\n{source}\n\n"
        + f"Return ONE expanded English prompt string. Apply the 3 rules and the format requirements from your system instructions."
    )

    # Температуры подобраны под Gemma 4 e4b: 0.20 нижний жёсткий ретрай,
    # 0.50 верхний - всё ещё умеренно консервативный (раньше было 0.55).
    retry_profiles = [
        {"temperature": 0.50, "top_p": 0.90},
        {"temperature": 0.35, "top_p": 0.93},
        {"temperature": 0.20, "top_p": 0.96},
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
                    # Поднято до 8192 ради Qwen3+ thinking-моделей: они жгут
                    # 1.5-3k токенов в `thinking` поле, потом ещё пишут expanded
                    # prompt в `response`. Gemma не-thinking, но на 8192 ей тоже
                    # ничего не мешает (выдаст что нужно и закроется по EOT).
                    "num_predict": 8192,
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
        # Стрип <think>...</think> блоков на случай если Ollama льёт CoT прямо
        # в response (старые версии до отдельного поля `thinking`, либо
        # модели где hybrid thinking лезет в основной текст).
        out = re.sub(r"<think\b[^>]*>.*?</think>", "", out, flags=re.DOTALL | re.IGNORECASE).strip()
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

        # Auto-fix: Gemma любит использовать реальные имена entity вместо [image N]
        # (например видит в source_prompt zhao_wangshan и пишет [zhao_wangshan]).
        # Перемапим любые [entity_name] обратно в [image N] по порядку entities[].
        if entities:
            for i, ent in enumerate(entities):
                if not ent:
                    continue
                out = re.sub(
                    r"\[\s*" + re.escape(ent) + r"\s*\]",
                    f"[image {i + 1}]",
                    out,
                    flags=re.IGNORECASE,
                )

        if not out:
            # Специально-частый кейс: Qwen3+ thinking модели жгут весь
            # num_predict в поле `thinking` и до `response` не доходят.
            # Печатаем явный диагностикс прежде чем retry'ить (чтобы
            # пользователь не гадал почему пусто).
            if isinstance(result, dict) and result.get("done_reason") == "length" and result.get("thinking"):
                thinking_len = len(result.get("thinking", ""))
                last_err = RuntimeError(
                    f"thinking-truncated: thinking={thinking_len} chars consumed all of "
                    f"num_predict=8192, response empty"
                )
                print(
                    f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                    f"thinking-truncated: thinking ел {thinking_len} chars, response пуст. "
                    f"Если повторится - подними num_predict выше 8192 или используй "
                    f"non-thinking модель (gemma4:e4b и т.п.); retrying"
                )
                continue
            last_err = RuntimeError("empty output from Gemma")
            # Диагностика: печатаем полный result чтобы понять почему пусто
            # (model not found / done_reason=stop_token / done_reason=load / etc).
            # Thinking может быть огромным - усечём для удобочитаемости.
            ollama_keys: dict[str, object] = {}
            if isinstance(result, dict):
                for k, v in result.items():
                    if k == "context":
                        continue
                    if k == "thinking" and isinstance(v, str):
                        ollama_keys[k] = v[:300] + (f"...[+{len(v) - 300} chars]" if len(v) > 300 else "")
                    else:
                        ollama_keys[k] = v
            else:
                ollama_keys = result  # type: ignore[assignment]
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

        # 4) Pixar 3D style stack (вкл. LoRA-триггер P1x4r)
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

        # 6) must_preserve_terms - все plain-text proper nouns из source
        # должны фигурировать в output (case-insensitive substring check).
        # Иначе Gemma дропнула character/имя и заменила на vague phrase.
        if proper_nouns:
            out_check = out.lower()
            missing_nouns = [p for p in proper_nouns if p.lower() not in out_check]
            if missing_nouns:
                last_err = RuntimeError(
                    f"missing must_preserve_terms (proper nouns dropped): {missing_nouns}"
                )
                print(
                    f"[qwen_expand] attempt {attempt + 1}/{len(retry_profiles)} "
                    f"missing proper nouns {missing_nouns}; retrying"
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
    base_prompts: list[dict] | None = None,
    ollama_url: str = "http://localhost:11434/api/generate",
    model: str = "gemma4:e4b",
    min_expansion_ratio: float = 1.4,
    rotate_cameras: bool = True,
) -> dict[int, str]:
    """
    Прогоняет ВСЕ image_prompt'ы сцен через Gemma за один проход.

    Делает per-сцену:
      1. resolve_entities_from_image_prompt(scene.image_prompt, base_names) —
         резолвит используемые в сцене entity-маркеры `[base_name]` через
         intersection с реальными именами из `plan["base_prompts"]`. Поле
         `scene["entities"]` больше НЕ читается из GROK plan'а (оно убрано
         из схемы) — pipeline сам определяет порядок entity по тексту
         image_prompt.
      2. substitute_entities_to_image_slots(scene.image_prompt, resolved_entities)
      3. Если rotate_cameras=True — берёт `pick_camera_preset(idx-1)` как
         camera_hint, где idx — глобальный 1-based порядковый номер
         сцены. Сцены обрабатываются изолированно — понятие thread
         больше не используется. Камеры ротируются по всему потоку
         сцен детерминистично (low angle → high angle → close-up → ...).
         Если False — camera_hint пустой, Gemma выбирает сама.
      4. expand_image_prompt_for_qwen_edit(...)

    Параметры:
      scenes        - список сцен из plan["scenes"]. Каждая сцена должна
                      иметь scene_id и image_prompt. Поле scene["entities"]
                      БОЛЬШЕ НЕ ТРЕБУЕТСЯ (если оно есть — игнорируется).
      base_prompts  - список base_prompts из plan["base_prompts"], каждый
                      элемент это dict с ключом "base_name" (см. GROK
                      schema). Используется для извлечения base_names
                      и резолва entities из текста image_prompt.
                      Если None — fallback на legacy scene["entities"]
                      (только для обратной совместимости со старыми
                      plan'ами; новые planы по `GROK_Prompt_2.txt` не
                      содержат поле entities).
      ollama_url    - endpoint Ollama API.
      model         - имя модели в Ollama (gemma4:e4b по умолчанию).
      min_expansion_ratio - см. expand_image_prompt_for_qwen_edit.
      rotate_cameras - если True, ракурсы ротируются по pick_camera_preset.

    Возвращает {scene_id: expanded_prompt}. Сцены без entities скипает
    (туда expand-логика не применима; в этих сценах image_prompt пойдёт
    напрямую без расширения, либо обрабатывай руками).

    Вызывать ДО загрузки Qwen Edit / Wan моделей в ComfyUI, иначе Ollama
    упрётся в OOM по системной памяти.
    """
    out: dict[int, str] = {}
    total = len(scenes)

    # Извлекаем base_names из plan["base_prompts"] для резолва entities из
    # текста image_prompt. Если caller не передал base_prompts — fallback
    # на legacy scene["entities"] в каждой сцене (старые GROK plan'ы).
    base_names: set[str] = set()
    if base_prompts:
        base_names = {
            bp["base_name"]
            for bp in base_prompts
            if isinstance(bp, dict) and "base_name" in bp
        }
        print(
            f"[precompute_image] resolving entities from image_prompt text "
            f"using {len(base_names)} known base_names"
        )

    for idx, scene in enumerate(scenes, 1):
        sid = scene["scene_id"]
        # Резолвим entities из текста image_prompt (новый способ —
        # GROK_Prompt_2.txt больше не выводит поле entities). Если
        # base_names не передан, читаем legacy scene["entities"].
        if base_names:
            entities = resolve_entities_from_image_prompt(
                scene["image_prompt"], base_names
            )
        else:
            entities = scene.get("entities", [])

        # idx — 1-based глобальный номер сцены. Для ротации камер берём
        # idx-1 (0-based). thread_id больше не учитывается — каждая сцена
        # обрабатывается изолированно, без cross-scene context.
        scene_index_for_camera = idx - 1

        if not entities:
            print(
                f"[precompute_image] [{idx}/{total}] scene {sid} no entities, "
                f"skipping expand (caller will use raw image_prompt)"
            )
            scene["used_entities"] = []
            continue
        if len(entities) > 3:
            raise ValueError(
                f"scene {sid} has {len(entities)} entities, max 3 supported. "
                f"Update GROK plan to split scene or reduce entities."
            )

        camera_hint = (
            pick_camera_preset(scene_index_for_camera) if rotate_cameras else ""
        )

        # Защита от багов GROK plan'а: prune unused entities, strip unknown brackets,
        # remap [image N] чтобы шли 1..k без пропусков.
        src, used_entities = prepare_image_prompt_for_qwen_edit(
            image_prompt=scene["image_prompt"],
            entities=entities,
            scene_id=sid,
        )
        # Сохраняем в scene для caller'а — image_paths нужно строить по used_entities
        scene["used_entities"] = used_entities

        if not used_entities:
            print(
                f"[precompute_image] [{idx}/{total}] scene {sid} all entities "
                f"pruned (none referenced in image_prompt), skipping expand"
            )
            continue

        # previous_recap больше не передаётся: cross-scene контекст оказался
        # вредным для image-gen (Gemma размывала композицию).
        expanded = await expand_image_prompt_for_qwen_edit(
            source_with_markers=src,
            n_images=len(used_entities),
            entities=used_entities,
            camera_hint=camera_hint,
            ollama_url=ollama_url,
            model=model,
            min_expansion_ratio=min_expansion_ratio,
        )
        out[sid] = expanded
        pruned_note = (
            f" (pruned from {len(entities)})"
            if len(used_entities) != len(entities)
            else ""
        )
        print(
            f"[precompute_image] [{idx}/{total}] scene {sid} ok, "
            f"{len(expanded)} chars, n_images={len(used_entities)}{pruned_note}, "
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
