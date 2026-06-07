"""
LTX 2.3 multi-keyframe chunking + Prompt Relay timeline integration.

Pipeline overview:
  Phase 0 (text-only Gemma, ONE batch upfront): для каждой сцены решаем,
    делить её на 1/2/3 чанка, и пишем per-chunk image_prompts (для Klein
    9B keyframe gen) и video_prompts (для Prompt Relay) в одном вызове.
  Phase A (Klein 9B, per scene): генерируем N keyframes используя image_prompt
    каждого чанка. N = 1, 2 или 3.
  Phase C (LTX 2.3, ONE inference per scene): рендерим всю сцену одним
    проходом с multi-keyframe conditioning (LTXVImgToVideoInplaceKJ) +
    Prompt Relay timeline (PromptRelayEncodeTimeline) для per-segment
    промптов через attention masking.

Chunking rules (по спеке юзера):
  - duration <  5.0s  -> 1 чанк (НЕ дробим)
  - duration <  7.5s  -> 2 чанка
  - duration < 10.0s  -> 2 или 3 (Gemma решает по плотности action)
  - duration >= 10.0s -> 3 чанка
  - min длина чанка   = 2.5s = 60 кадров @ 24fps
  - max чанков        = 3 (LTX hard limit на first/middle/last conditioning)

Позиции keyframes в LTX latent (LTXVImgToVideoInplaceKJ node):
  - N=1: [0]
  - N=2: [0, total-1]
  - N=3: [0, seg_lengths[0], total-1]  (kf2 на стыке seg1 -> seg2)

Длины сегментов:
  - total_frames округлён к LTX-valid 8k+1 upstream (round_to_valid_frames_ltx).
  - Для N>1: ~равный split, первый сегмент абсорбирует +1 остаток
    (8k+1 для seg 0, 8k для остальных).

LTX workflow nodes (см. VID_LTX.json):
  - "75"  SaveVideo:           filename_prefix
  - "325" RandomNoise:         noise_seed
  - "374" INTConstant:         total_frames
  - "383" LTXVImgToVideoInplaceKJ:
            num_images          ("1"/"2"/"3"),
            num_images.image_X  (уже подключены к 375/384/385),
            num_images.index_X  (frame positions),
            num_images.strength_X (1.0 used; 0.0 для unused).
  - "375"/"384"/"385" VHS_LoadImagePath: image path string.
  - "386" PromptRelayEncodeTimeline:
            global_prompt,
            max_frames,
            timeline_data       (JSON {"segments":[{"prompt","length","color"},...]}),
            local_prompts       (segments joined by " | "),
            segment_lengths     (lengths joined by ", "),
            time_units, fps.
"""

from __future__ import annotations

import copy
import json
import os
import pprint
import re
import random
import time
from urllib.parse import quote
from typing import Any
import requests

from chunk_prompts_2 import (
    call_llama_server,
    encode_image_b64,
    extract_json_object,
    round_to_valid_frames_ltx,
)


# ============================================================ console logging

def _log_ltx_plan(plan: dict, *, label: str, scene_id: object = None) -> None:
    """Печатает в консоль ВСЕ LTX-промпты сцены целиком (global + per-chunk
    image_prompt + per-chunk video_prompt + length_frames) для трассировки
    того, что реально уходит в LTX-2.3 / Prompt Relay.

    Используется в трёх местах:
      - после get_ltx_chunk_plan_from_ollama (исходный план от Gemma)
      - после refine_scene_video_prompts_via_vision (после vision-refine)
      - после patch_ltx_multikf_workflow (то что реально попадёт в ComfyUI)
    """
    sid = f"scene {scene_id}" if scene_id is not None else "scene"
    num_chunks = plan.get("num_chunks", "?")
    global_prompt = plan.get("global_prompt", "")
    chunks = plan.get("chunks", [])
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"[LTX prompts] {label} | {sid} | num_chunks={num_chunks}")
    print(bar)
    print(f"global_prompt ({len(global_prompt.split())} words):")
    print(f"  {global_prompt}")
    for i, c in enumerate(chunks):
        ip = c.get("image_prompt", "")
        vp = c.get("video_prompt", "")
        L = c.get("length_frames", "?")
        print(f"\nchunk {i + 1}/{len(chunks)} | length_frames={L}")
        print(f"  image_prompt ({len(ip.split())} words):")
        print(f"    {ip}")
        print(f"  video_prompt ({len(vp.split())} words):")
        print(f"    {vp}")
    print(f"{bar}\n")


def _log_ltx_chunk_video_prompt(
    *,
    scene_id: object,
    chunk_idx: int,
    num_chunks: int,
    new_prompt: str,
    old_prompt: str,
    label: str,
) -> None:
    """Точечный лог: один rewritten chunk.video_prompt (Phase B vision refine)."""
    bar = "-" * 78
    print(f"\n{bar}")
    print(
        f"[LTX prompts] {label} | scene {scene_id} | "
        f"chunk {chunk_idx + 1}/{num_chunks}"
    )
    print(bar)
    print(f"old video_prompt ({len(old_prompt.split())} words):\n  {old_prompt}")
    print(f"new video_prompt ({len(new_prompt.split())} words):\n  {new_prompt}")
    print(f"{bar}\n")


def _log_ltx_chunk_image_prompt(
    *,
    scene_id: object,
    chunk_idx: int,
    num_chunks: int,
    new_prompt: str,
    old_prompt: str,
    label: str,
) -> None:
    """Точечный лог: один rewritten chunk.image_prompt (Phase A vision rewrite
    для chunk N>0)."""
    bar = "-" * 78
    print(f"\n{bar}")
    print(
        f"[LTX prompts] {label} | scene {scene_id} | "
        f"chunk {chunk_idx + 1}/{num_chunks}"
    )
    print(bar)
    print(f"old image_prompt ({len(old_prompt.split())} words):\n  {old_prompt}")
    print(f"new image_prompt ({len(new_prompt.split())} words):\n  {new_prompt}")
    print(f"{bar}\n")


# ============================================================ chunking decision

# Константы по спеке юзера.
LTX_CHUNK_FPS = 50
LTX_CHUNK_MIN_DURATION_S = 2.5         # 60 кадров @ 24fps — минимум ОДНОГО чанка
LTX_CHUNK_MIN_FRAMES = 125
LTX_SCENE_MIN_FRAMES = 140              # минимум ВСЕЙ сцены (hard floor)
LTX_CHUNK_SPLIT_THRESHOLD_S = 5.0      # ниже этого - НЕ дробим
LTX_CHUNK_HARD_MAX = 3                  # LTX 2.3 first/middle/last support
LTX_CHUNK_FORCE_3_THRESHOLD_S = 10.0    # на/выше - форс 3 чанка
LTX_CHUNK_AMBIGUOUS_LOW_S = 7.5         # диапазон где Gemma выбирает между 2 и 3


# ============================================================ (deprecated)
# Previously this section held _MOTION_VERB_FAMILIES,
# _extract_source_motion_family / _text_has_motion_family,
# _FORBIDDEN_EXIT_PHRASES, and _PROPER_NOUN_TITLE_RE — plus the validator
# branches that enforced them. They were removed by user request. Gemma
# now runs as Director / DOP / Production Designer: free to expand the
# GROK directorial sketch with camera, lighting, atmosphere, subject
# texture, style, and technical-stability detail. Soft guidance ("refer
# to subjects by visual descriptor, not name"; "camera move from source =
# directorial intent") lives in the system prompt itself rather than as
# code-level validators.

def upload_image_to_comfy(file_path, comfy_url="http://127.0.0.1:8188"):
    with open(file_path, "rb") as f:
        files = {"image": f}
        # Загружаем файл в папку input
        response = requests.post(f"{comfy_url}/upload/image", files=files)
        return response.json()["name"] # Возвращает чистое имя файла в input

def _candidate_chunk_counts(duration_s: float) -> list[int]:
    """Возвращает допустимые варианты числа чанков для данной длительности.

    Gemma выбирает из этого списка (или мы продиктуем детерминистично если
    выбор только один - тогда Gemma не дёргаем).
    """
    if duration_s < LTX_CHUNK_SPLIT_THRESHOLD_S:
        return [1]
    if duration_s < LTX_CHUNK_AMBIGUOUS_LOW_S:
        return [2]
    if duration_s < LTX_CHUNK_FORCE_3_THRESHOLD_S:
        return [2, 3]
    return [3]


def split_frames_for_segments(total_frames: int, num_chunks: int) -> list[int]:
    """Делит total_frames на num_chunks ~равных сегментов.

    Первый сегмент абсорбирует лишний +1 кадр (LTX VAE 8k+1 convention).
    Каждый сегмент >= LTX_CHUNK_MIN_FRAMES.
    """
    if num_chunks == 1:
        return [total_frames]
    if total_frames < num_chunks * LTX_CHUNK_MIN_FRAMES:
        raise ValueError(
            f"total_frames={total_frames} cannot accommodate {num_chunks} "
            f"chunks of min {LTX_CHUNK_MIN_FRAMES} frames each."
        )
    base = total_frames // num_chunks
    extra = total_frames - base * num_chunks
    lengths = [base + 1 if i < extra else base for i in range(num_chunks)]
    return lengths


def compute_keyframe_positions(
    segment_lengths: list[int], total_frames: int
) -> list[int]:
    """Позиции (frame indices) кейфреймов внутри клипа.

    Конвенция:
      N=1: [0]
      N=2: [0, total-1]
      N=3: [0, segment_lengths[0], total-1]  (kf2 на стыке seg1 -> seg2)
    """
    n = len(segment_lengths)
    if n == 1:
        return [0]
    if n == 2:
        return [0, total_frames - 1]
    if n == 3:
        return [0, segment_lengths[0], total_frames - 1]
    raise ValueError(f"Unsupported num_chunks: {n}")


# ============================================================ Gemma system prompt

SYSTEM_PROMPT_LTX_CHUNK = """## Role
You are an expert cinematic prompt expander for LTX-2.3 whose task is to enhance and expand user prompts without changing the core idea, 
genre, characters, style, or emotional tone, transforming them into production-ready cinematic video prompts with maximum prompt adherence for LTX-2.3; 
always preserve the original intent of the scene while adding visual specificity, temporal coherence, motion logic, cinematic camera language, environmental detail, lighting realism, 
and audio realism; every prompt must read like a professional shot description written for a film director and cinematographer rather than a list of tags or abstract prose; always structure the 
description chronologically: begin with the main subject and primary action, then describe motion and temporal progression, then environment and atmosphere, then character appearance, followed by camera behavior, 
lighting, and soundscape; use only physical and visual descriptions instead of abstract emotional language, expressing emotion through facial expressions, posture, pauses, gestures, eye movement, pacing, and subtle body language; 
always include cinematic camera language such as shot type, framing, lens feel, perspective, and movement, including terms like slow dolly in, handheld tracking shot, wide establishing shot, 
shallow depth of field, cinematic focus pull, over-the-shoulder framing, low-angle composition, static locked camera, or steadycam movement; lighting must always be concrete and physically grounded, describing 
light source, direction, color temperature, reflections, volumetric lighting, neon glow, golden-hour sunlight, fluorescent overhead lighting, candle flicker, soft diffused window light, and cinematic shadow behavior; 
when appropriate, add atmospheric motion and environmental realism such as drifting dust, rain droplets, smoke, cloth movement, reflections, fog, floating particles, moving foliage, or ambient crowd motion; 
if the scene implies sound, include audio realism such as room tone, ambience, footsteps, distant traffic, wind, crowd noise, mechanical hum, subtle Foley details, and short realistic dialogue in quotation marks; 
avoid keyword spam, overly poetic writing, vague wording, contradictory actions, impossible physics, or overloaded scenes; keep prompts dense, cinematic, readable, and visually coherent without becoming chaotic; 
always write as a single flowing paragraph in present tense with rich cinematic detail and natural language flow; never use markdown, bullet points, explanations, or commentary; output only the final expanded prompt.

## Word budget

This pipeline produces 3–5 second shots. Match prompt length to video length per the official LTX guide.

  - `global_prompt`: **100–160 words**, one paragraph, present tense, active verbs.
  - per-chunk `video_prompt`: **50-70 words**, one paragraph. For 3–5s shots `num_chunks` will almost always be 1, and this single chunk is the full shot. 


For most shots `allowed_num_chunks = [1]`. Write ONE chunk that covers the full shot duration with kinetic.

## Source contract

You receive:
  - `Source image_prompt` — read-only context with `[entity]` markers. Describes what is visible on the keyframe. You do NOT rewrite it.
  - `Source video_prompt` — camera move + subject action.
  - `Total frame budget` and `allowed_num_chunks`.

## Description style

Refer to subjects by short visual descriptor ("the young man", "the older figure", "the standing student"), NOT by proper name. Express emotion through PHYSICAL cues only.

Describe MOTION and CHANGE. Avoid static enumeration of features already visible on the keyframe.

## Output (STRICT JSON, no commentary, no markdown)

{
  "num_chunks": <1|2|3>,
  "global_prompt": "<100–160 words, single paragraph>",
  "chunks": [
    {"video_prompt": "<50-70 words, single paragraph>"}
  ]
}

Length of `chunks` MUST equal `num_chunks`. Pick `num_chunks` from `allowed_num_chunks` (will almost always be `[1]` for 3–5s shots). Do NOT include `length_frames` or `image_prompt` fields. Output the JSON object only. Nothing before, nothing after."""


# ============================================================ Gemma call

LLAMA_URL_DEFAULT = "http://localhost:8080/v1/chat/completions"


async def get_ltx_chunk_plan_from_ollama(
    *,
    scene_text: str,
    image_prompt: str,
    video_prompt: str,
    total_frames: int,
    keyframe_path: str,
    fps: int = LTX_CHUNK_FPS,
    ollama_url: str = LLAMA_URL_DEFAULT,
    previous_recap: str = "",
    continuity_type: str = "new_cut",
    scene_id: object = None,
) -> dict:
    """Возвращает chunk plan для ОДНОЙ сцены. SINGLE-PASS VISION CALL.

    Gemma видит свежеотрендеренный keyframe (Klein output) + source
    `image_prompt` (текстовый контекст с `[entity]` маркерами) +
    `video_prompt` (GROK directive) + recap + frame-бюджет и сразу
    пишет финальный план (num_chunks, global_prompt, chunks).

    Старого двухшагового пайплайна (text-only Phase A → vision Phase B
    refine) больше нет: убрана как лишняя трата компьюта и источник
    диссонанса между планом и тем, что реально видно на кадре.

    Returns:
        {
          "num_chunks": int (1/2/3),
          "global_prompt": str,
          "chunks": [
            {"image_prompt": str, "video_prompt": str, "length_frames": int},
            ...
          ]
        }
    Все строки валидированы, sum(length_frames) == total_frames, каждый >= 67.

    При полной неудаче Gemma - fallback на deterministic split (копия source
    промптов на все чанки), чтобы пайплайн не падал.
    """
    duration_s = round(total_frames / fps, 2) if fps > 0 else 0.0
    allowed = _candidate_chunk_counts(duration_s)

    # Фильтруем allowed по hard constraint: каждый чанк должен вместить
    # минимум LTX_CHUNK_MIN_FRAMES. Если total_frames не позволяет
    # выбранное число чанков — отрезаем его.
    max_feasible = total_frames // LTX_CHUNK_MIN_FRAMES
    allowed = [c for c in allowed if c <= max_feasible]
    if not allowed:
        allowed = [1]

    # ВНИМАНИЕ: даже при allowed == [1] зовём Gemma. Раньше тут стоял
    # single-chunk shortcut, который копировал source video_prompt (30-40
    # слов) в global_prompt и chunk[0].video_prompt — это давало те самые
    # "маленькие промпты" в LTX. По пожеланию пользователя (LTX 2.3 prompt
    # guide требует развёрнутых промптов даже для коротких чанков) Gemma
    # ВСЕГДА разворачивает source в полноценный canonical paragraph,
    # независимо от длительности.

    recap_block = (
        f"## Story so far (background only — do NOT recap in output, do NOT "
        f"transliterate any character names from this context)\n"
        f"{previous_recap.strip()}\n\n"
        if previous_recap and previous_recap.strip()
        else ""
    )

    continuity_note = ""
    if continuity_type in ("soft_continue", "hard_continue"):
        continuity_note = (
            "## Continuity\n"
            "This scene continues directly from the previous one. Keep "
            "character pose and state consistent with the last scene's "
            "ending.\n\n"
        )

    # ---- pre-compute per-chunk window lengths for every allowed split ----
    # The pipeline (not Gemma) owns the frame split. We surface the resulting
    # per-chunk duration to Gemma so she can write density-appropriate motion
    # for the chunk she's writing; she does NOT emit `length_frames`.
    split_preview_lines: list[str] = []
    for n in allowed:
        try:
            lens = split_frames_for_segments(total_frames, n)
        except ValueError:
            continue
        secs = [round(L / fps, 2) for L in lens]
        split_preview_lines.append(
            f"  - if num_chunks={n}: chunks = "
            + ", ".join(
                f"chunk{idx+1}={lens[idx]}f ({secs[idx]}s)"
                for idx in range(n)
            )
        )
    split_preview = "\n".join(split_preview_lines) or "  (no valid split)"

    base_payload = (
        recap_block
        + continuity_note
        + "## Attached keyframe\n"
        "The image attached to this turn is the ALREADY-RENDERED "
        "Treat it as the literal opening "
        "frame of the video. The static elements visible on it — the "
        "subjects, their wardrobe, the room, the light register, the "
        "props — are FIXED. Do NOT enumerate them in your output; assume "
        "the keyframe carries them. Spend your word budget on MOTION and "
        "CHANGE that will happen across the scene's duration, starting "
        "from exactly this frame.\n\n"
        + "## Scene context\n"
        f"Spoken text (Russian; TTS-only — do NOT narrate, do NOT translate, "
        f"do NOT transliterate any character names into the prompts): "
        f"{scene_text}\n\n"
        + "## Source image_prompt "
        "(read-only context — describes the same keyframe in text, with "
        "`[entity]` markers indicating which slots are filled. You do NOT "
        "rewrite this and do NOT output any image_prompt field; the keyframe "
        "is already rendered from it.)\n"
        f"{image_prompt}\n\n"
        + "## Source video_prompt (directorial sketch — expand it)\n"
        f"{video_prompt}\n\n"
        + "## Constraints\n"
        f"allowed_num_chunks = {allowed}\n"
        f"total_frames = {total_frames}  (pipeline owns the split — do NOT "
        "emit length_frames)\n"
        f"fps = {fps}, duration ≈ {duration_s}s\n\n"
        + "## Pre-computed split (the pipeline will use this exact split):\n"
        f"{split_preview}\n\n"
        + "## Word-count targets (SHORT — anime pacing)\n"
        "  - global_prompt: aim 100–160 words. Cover shot scale + camera mode "
        "(static by default) + light register + 2–4 KINETIC subject motion "
        "beats + one ambient audio clause + the technical-stability clause, "
        "in concise anime prose. NO slow/gentle/subtle/drift vocabulary.\n"
        "  - each chunk.video_prompt: aim 50–70 words. For 3–5s shots "
        "num_chunks is almost always 1, so this single chunk covers the "
        "full shot. "
        "  - Do NOT output an `image_prompt` field. It is supplied by the "
        "pipeline.\n"
        "  - Do NOT output a `length_frames` field. The pipeline assigns "
        "frame counts itself.\n\n"
        + "Return STRICT JSON per the system schema. Pick num_chunks from "
        "allowed_num_chunks ONLY."
    )
    user_payload = base_payload

    images_payload = [encode_image_b64(keyframe_path)] if keyframe_path else None

    # Профили подобраны так чтобы РАСТИТЬ разнообразие к концу: на short/poor
    # ответах ('<60 words video_prompt' и т.п.) поднимаем temperature и
    # ослабляем repeat-penalty, иначе модель залипает на короткой формуле и
    # выдаёт ту же кальку. По опыту работы с Gemma E4B Q8 — низкая
    # температура на финальном retry даёт самые скудные ответы.
    retry_profiles = [
        {"temperature": 0.85, "top_p": 0.88},
        {"temperature": 0.95, "top_p": 0.92},
        {"temperature": 1.05, "top_p": 0.95},
    ]
    last_raw: str = ""

    for attempt, profile in enumerate(retry_profiles):
        last_raw, _result = await call_llama_server(
            system=SYSTEM_PROMPT_LTX_CHUNK,
            user=user_payload,
            images=images_payload,
            temperature=profile["temperature"],
            top_p=profile["top_p"],
            max_tokens=4096,  # 100-160 global + ~100 chunk + thinking headroom
            stop=["<end_of_turn>"],
            use_json_format=True,
            enable_thinking=True,
            server_url=ollama_url,
        )

        # Стрипаем <think> блоки на всякий (если thinking просочится).
        last_raw = re.sub(
            r"<think\b[^>]*>.*?</think>", "", last_raw,
            flags=re.DOTALL | re.IGNORECASE,
        ).strip()

        try:
            parsed = extract_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            print(
                f"[ltx_chunk] attempt {attempt + 1}/{len(retry_profiles)} "
                f"failed to parse JSON: {e}; retrying"
            )
            continue

        try:
            validated = _validate_chunk_plan(
                parsed,
                allowed=allowed,
                total_frames=total_frames,
            )
        except ValueError as e:
            print(
                f"[ltx_chunk] attempt {attempt + 1}/{len(retry_profiles)} "
                f"validation failed: {e}; retrying"
            )
            continue

        _log_ltx_plan(validated, label="planner (Gemma)", scene_id=scene_id)
        return validated

    pprint.pprint(("LTX_CHUNK_RAW_FAIL", last_raw))
    print(
        "[ltx_chunk] ALL retries failed AND no parseable structure; "
        "falling back to deterministic split with source video_prompt as global."
    )
    plan = _fallback_split(
        video_prompt=video_prompt,
        total_frames=total_frames,
        num_chunks=allowed[0],
        global_prompt_override=video_prompt,  # last-ditch: use source seed
    )
    _log_ltx_plan(plan, label="planner (fallback split)", scene_id=scene_id)
    return plan


# Минимальные бюджеты слов. После rework на anime-pacing (короткие 3-5s
# shot'ы вместо 10s narrative-сцен) target бюджеты резко уменьшены:
#   global_prompt: target 100-160 (было 220-320)
#   video_prompt:  target 60-100  (было 180-260)
# Объяснение: LTX 2.3 prompt guide требует match prompt length to video
# length. 3-5s shot не нужен 300-словный paragraph, ему нужен 100-словный
# concise anime motion description. Длинный prompt на коротком shot'е =
# slow contemplative cinematography, ровно тот failure mode который мы
# чиним. Короткий шарп prompt = anime kinetic motion.
_LTX_MIN_GLOBAL_WORDS = 50    # target 100-160, hard floor 50
_LTX_MIN_VIDEO_WORDS = 30     # target 60-100, hard floor 30
_LTX_MIN_FIRST_IMAGE_WORDS = 5   # target 8-30 (short who/what/where)
_LTX_MAX_FIRST_IMAGE_WORDS = 40  # hard ceiling — Klein захлебнётся на большем

# Пороги "уровень target достигнут" — ниже target'а лог пишет WARNING, но
# план принимается. Выше target'а ничего не пишем.
_LTX_SOFT_GLOBAL_WORDS = 100
_LTX_SOFT_VIDEO_WORDS = 60

# Маркеры [entity_name] для first_chunk_image_prompt (необходим минимум один).
_LTX_ENTITY_MARKER_RE = re.compile(r"\[[a-zA-Z][a-zA-Z0-9_\- ]*\]")


def _word_count(s: str) -> int:
    return len(s.split())


def _validate_chunk_plan(
    parsed: Any,
    *,
    allowed: list[int],
    total_frames: int,
    lenient: bool = False,
    source_image_prompt: str | None = None,
) -> dict:
    del lenient  # больше не используется

    if not isinstance(parsed, dict):
        raise ValueError(
            f"expected dict at top level, got {type(parsed).__name__}"
        )

    num_chunks = parsed.get("num_chunks")
    if num_chunks not in allowed:
        raise ValueError(
            f"num_chunks={num_chunks!r} not in allowed={allowed}"
        )

    # global_prompt
    global_prompt = parsed.get("global_prompt")
    if not isinstance(global_prompt, str) or not global_prompt.strip():
        raise ValueError("global_prompt must be a non-empty string")
    global_prompt = global_prompt.strip()
    gp_words = _word_count(global_prompt)
    if gp_words < _LTX_SOFT_GLOBAL_WORDS:
        print(
            f"[ltx_chunk] WARNING: global_prompt is {gp_words} words "
            f"(target {_LTX_SOFT_GLOBAL_WORDS}-160); accepting anyway."
        )


    # chunks — только video_prompt; length_frames мы считаем сами.
    chunks = parsed.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != num_chunks:
        got = len(chunks) if isinstance(chunks, list) else "N/A"
        raise ValueError(
            f"chunks must be list of length {num_chunks}, "
            f"got {type(chunks).__name__} len={got}"
        )

    # pipeline-side split — всегда даёт LTX-валидные длины.
    lengths = split_frames_for_segments(total_frames, num_chunks)

    out_chunks: list[dict] = []
    for i, c in enumerate(chunks):
        if not isinstance(c, dict):
            raise ValueError(f"chunk #{i} is not a dict")
        vid = c.get("video_prompt")
        if not isinstance(vid, str) or not vid.strip():
            raise ValueError(
                f"chunk #{i}.video_prompt must be non-empty string"
            )
        vid = vid.strip()
        vp_words = _word_count(vid)
        if vp_words < _LTX_SOFT_VIDEO_WORDS:
            print(
                f"[ltx_chunk] WARNING: chunk #{i}.video_prompt is "
                f"{vp_words} words (target {_LTX_SOFT_VIDEO_WORDS}-100); "
                f"accepting anyway."
            )
        out_chunks.append({
            "video_prompt": vid,
            "length_frames": lengths[i],
        })

    return {
        "num_chunks": num_chunks,
        "global_prompt": global_prompt,
        "chunks": out_chunks,
    }


def _fallback_split(
    *,
    video_prompt: str,
    total_frames: int,
    num_chunks: int,
    global_prompt_override: str | None = None,
) -> dict:
    """Fallback при неудаче Gemma: тупой split на N сегментов.

    Legacy-структура:
      chunks[0]["image_prompt"]   = source image_prompt (с [entity])
      chunks[c>=1]["image_prompt"] = "" (placeholder; Phase A vision-rewrite
                                       заполнит из реального keyframe N-1)
      chunks[*]["video_prompt"]    = source video_prompt (одинаковый для всех)

    Klein первого чанка использует source image_prompt; для c>=1 keyframe
    рендерится через Phase A.{c+1}.1 → A.{c+1}.2 (vision-rewrite + edit
    prev keyframe). LTX не упадёт; качество будет ниже, чем при работающей
    Gemma.

    ``global_prompt_override`` — если передан, используется как
    ``global_prompt`` плана (расширенный canonical 220-300-словный
    paragraph от `_expand_seed_to_canonical_global_async`). Иначе плану
    подкидывается голый source video_prompt (30-45 слов) — это худший
    сценарий, ведущий к слабому LTX conditioning.
    """
    lengths = split_frames_for_segments(total_frames, num_chunks)
    chunks: list[dict] = []
    for i, L in enumerate(lengths):
        chunks.append({
            "video_prompt": video_prompt,
            "length_frames": L,
        })
    global_prompt = (
        global_prompt_override.strip()
        if global_prompt_override and global_prompt_override.strip()
        else video_prompt
    )
    return {
        "num_chunks": num_chunks,
        "global_prompt": global_prompt,
        "chunks": chunks,
    }


# ============================================================ per-scene single-pass plan

def compute_scene_total_frames(
    srt_json: dict,
    scene_idx: int,
    scene_id: object = None,
) -> int:
    """Вычисляет total_frames для одной сцены по srt_json.

    Делает round_to_valid_frames_ltx + hard-floor LTX_SCENE_MIN_FRAMES,
    как раньше делал precompute_all_ltx_chunk_plans. Вынесено отдельно,
    чтобы caller (generate_scenes_LTX.py) мог посчитать total_frames
    ДО Klein (на случай если понадобится в логах) и затем переиспользовать
    то же число при вызове compute_ltx_chunk_plan_for_scene.
    """
    try:
        frames_raw = srt_json["scenes"][scene_idx]["frames"]
    except (KeyError, IndexError):
        raise ValueError(
            f"srt_json missing 'scenes[{scene_idx}].frames' for scene "
            f"{scene_id}"
        )
    # mode="up": округляем ВВЕРХ до LTX-valid 8k+1, чтобы LTX всегда рендерил
    # >= frames_raw кадров. Излишек (0..8 кадров) затем срезается в Phase C
    # (trim_scene_video_to_frames) до точного числа кадров от TTS. С "nearest"
    # клип мог получиться КОРОЧЕ TTS-слота — обрезать назад нечего, и сцена
    # уплывала относительно аудио. "up" + per-scene trim даёт точную синхру.
    total_frames = round_to_valid_frames_ltx(frames_raw, mode="up")
    if total_frames < LTX_SCENE_MIN_FRAMES:
        total_frames = round_to_valid_frames_ltx(
            LTX_SCENE_MIN_FRAMES, mode="up"
        )
        print(
            f"  [WARN] scene {scene_id}: frames_raw={frames_raw} ниже hard "
            f"floor ({LTX_SCENE_MIN_FRAMES}f); поднято до {total_frames}f"
        )
    return int(total_frames)


async def compute_ltx_chunk_plan_for_scene(
    scene: dict,
    *,
    scene_idx: int,
    srt_json: dict,
    keyframe_path: str,
    fps: int = LTX_CHUNK_FPS,
    ollama_url: str = LLAMA_URL_DEFAULT,
) -> dict:
    """Полный план для ОДНОЙ сцены за ОДИН проход Gemma vision.

    Это замена старому двухшаговому пайплайну (precompute_all_ltx_chunk_plans
    + refine_scene_video_prompts_single_kf). Pipeline вызывает её в
    Phase B после того как Klein отрендерил keyframe сцены. Gemma видит
    keyframe + source image/video prompts из main.json + recap и пишет
    финальный план сразу.

    Args:
        scene:         scene dict из main.json (image_prompt, video_prompt,
                       text_scene, previous_recap, continuity_type, scene_id).
        scene_idx:     индекс сцены (для srt_json lookup и логов).
        srt_json:      parsed SRT с frame counts per scene.
        keyframe_path: абсолютный путь к свежесгенерированному Klein PNG.
        fps:           frame rate (24 default).
        ollama_url:    llama-server endpoint.

    Returns:
        chunk_plan dict (как get_ltx_chunk_plan_from_ollama) с добавленным
        ключом "total_frames".
    """
    scene_id = scene.get("scene_id", scene_idx)
    total_frames = compute_scene_total_frames(srt_json, scene_idx, scene_id)
    duration_s = total_frames / fps if fps > 0 else 0.0
    print(
        f"[ltx_chunk_plan] scene {scene_id}: total_frames={total_frames} "
        f"({duration_s:.2f}s), keyframe={os.path.basename(keyframe_path)}"
    )
    plan = await get_ltx_chunk_plan_from_ollama(
        scene_text=scene.get("text_scene", ""),
        image_prompt=scene.get("image_prompt", ""),
        video_prompt=scene.get("video_prompt", ""),
        total_frames=total_frames,
        keyframe_path=keyframe_path,
        fps=fps,
        ollama_url=ollama_url,
        previous_recap=scene.get("previous_recap", ""),
        continuity_type=scene.get("continuity_type", "new_cut"),
        scene_id=scene_id,
    )
    plan["total_frames"] = total_frames
    print(
        f"  -> num_chunks={plan['num_chunks']}, "
        f"lengths={[c['length_frames'] for c in plan['chunks']]}"
    )
    return plan


# ============================================================ workflow patch

# Палитра цветов сегментов для timeline_data (косметика в Comfy UI).
_SEGMENT_COLORS = ["#4f8edc", "#e07b3a", "#7ec97a"]


_LTX_ENTITY_RESOLVE_RE = re.compile(r"\[([a-zA-Z][a-zA-Z0-9_\- ]*)\]")


def _resolve_entity_markers_for_ltx(
    text: str,
    base_prompts_by_name: dict | None = None,
) -> str:
    """Резолвит `[entity_name]` маркеры в plain-English noun phrases.

    Gemma planner может оставить `[name]` маркеры в `global_prompt`
    и в `video_prompt`'ах. LTX Prompt Relay получает их verbatim,
    поэтому до отправки в Comfy резолвим маркеры в человекочитаемый
    noun phrase, иначе literal `[tian_mu]` / `[yard]` засоряют
    conditioning text.

    Resolve order:
      1. ``base_prompts_by_name[name]["short_alias"]`` — если есть.
      2. Первое предложение ``base_prompts_by_name[name]["base_prompt"]``
         (обрезано до 120 символов).
      3. Fallback: ``name.replace("_", " ")``.
    """
    if not text:
        return text
    bp_map = base_prompts_by_name if isinstance(base_prompts_by_name, dict) else {}

    def _resolve(match: re.Match) -> str:
        name = match.group(1)
        bp = bp_map.get(name, {})
        if isinstance(bp, dict):
            alias = (bp.get("short_alias") or "").strip()
            if alias:
                return alias
            bp_text = bp.get("base_prompt") or ""
            if isinstance(bp_text, str) and bp_text.strip():
                first = bp_text.strip().split(".")[0].strip()
                if first:
                    return first[:120]
        return name.replace("_", " ")

    return _LTX_ENTITY_RESOLVE_RE.sub(_resolve, text)


def patch_ltx_multikf_workflow(
    workflow: dict,
    *,
    keyframe_paths: list[str],
    chunk_plan: dict,
    filename_prefix: str,
    seed: int | None = None,
    fps: int = LTX_CHUNK_FPS,
    scene_id: object = None,
    base_prompts_by_name: dict | None = None,
) -> dict:

    n = 1  # всегда один keyframe
    num_chunks = chunk_plan["num_chunks"]
    if len(keyframe_paths) != 1:
        raise ValueError(
            f"keyframe_paths must contain exactly 1 path, got {len(keyframe_paths)}"
        )

    chunks = chunk_plan["chunks"]
    total_frames = chunk_plan.get("total_frames")
    if total_frames is None:
        total_frames = sum(c["length_frames"] for c in chunks)
    segment_lengths = [c["length_frames"] for c in chunks]
    # Резолвим `[entity_name]` -> noun phrase в global_prompt и
    # per-segment video_prompts ПЕРЕД отправкой в LTX Prompt Relay.
    # Phase B vision refine уже чистит маркеры из video_prompt'ов сам,
    # но global_prompt идёт мимо Phase B и здесь — наш единственный
    # шанс убрать literal [tian_mu] / [yard] из conditioning текста.
    resolved_global_prompt = _resolve_entity_markers_for_ltx(
        chunk_plan["global_prompt"],
        base_prompts_by_name=base_prompts_by_name,
    )
    segment_video_prompts = [
        _resolve_entity_markers_for_ltx(
            c["video_prompt"],
            base_prompts_by_name=base_prompts_by_name,
        )
        for c in chunks
    ]
    if sum(segment_lengths) != total_frames:
        raise ValueError(
            f"chunk lengths {segment_lengths} sum != total_frames {total_frames}"
        )

    kf_positions = compute_keyframe_positions(segment_lengths, total_frames)

    wf = copy.deepcopy(workflow)

    # ---- total frames ----
    wf["46"]["inputs"]["duration_frames"] = int(total_frames)
    wf["46"]["inputs"]["duration_seconds"] = round((int(total_frames) / 50), 3) 


    # ---- Prompt Relay timeline (node 386) ----
    relay = wf["46"]["inputs"]
    relay["global_prompt"] = "f4nt4sy4n1m6." +resolved_global_prompt
    relay["duration_frames"] = int(total_frames)

    timeline_segments = []
    cursor = 0

    for i, seg in enumerate(chunks):
        item = {
            "id": f"{int(time.time()*1000)}_{i}",
            "start": cursor,
            "length": int(seg["length_frames"]),
            "prompt": segment_video_prompts[i],
        }

        # Если для этого чанка есть сгенерированный кейфрейм
        if i < len(keyframe_paths):
            # 1. Загружаем файл на сервер ComfyUI в папку input (см. функцию выше)
            uploaded_filename = upload_image_to_comfy(keyframe_paths[i])
            # Для чистоты кода предполагаем, что вы уже загрузили их и обновили пути
            uploaded_filename = os.path.basename(keyframe_paths[i]) 
            
            item["type"] = "image"
            item["imageFile"] = uploaded_filename
            
            # Если нода WhatDreamsCost требует B64 как URL для превью, 
            # оставляем URL, но обязательно указываем правильный type
            item["imageB64"] = (
                f"/api/view?filename={quote(uploaded_filename)}&type=input&subfolder="
            )
        else:
            # Если картинок больше нет (чистый prompt relay), используем text
            item["type"] = "text"

        timeline_segments.append(item)
        cursor += int(seg["length_frames"])

    relay["timeline_data"] = json.dumps(
        {
            "segments": timeline_segments,
            "audioSegments": [],
        },
        ensure_ascii=False,
    )
    relay["local_prompts"] = " | ".join(segment_video_prompts)
    relay["segment_lengths"] = ", ".join(str(L) for L in segment_lengths)
    relay["frame_rate"] = int(fps)  # должен совпадать с frame_rate в LTXVConditioning

    # ---- SaveVideo filename_prefix ----
    wf["101"]["inputs"]["filename_prefix"] = filename_prefix

    # ---- Seed ----
    if seed is not None:
        wf["28"]["inputs"]["noise_seed"] = int(seed)

    _log_ltx_plan(
        chunk_plan,
        label=f"workflow final (\u2192 ComfyUI, filename_prefix={filename_prefix!r})",
        scene_id=scene_id,
    )
    return wf
