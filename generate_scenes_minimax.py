"""
MiniMax H3 ref2video pipeline — прямая генерация видео из identity-рефов.

Пайплайн (два генеративных этапа):

  ЭТАП 1: base entity refs — Z-Anime text-to-image (720p, как раньше).
          Для каждой сущности из plan["base_prompts"] рисуется один
          канонический референс в base_image_gen/ (персонажи 720x1280,
          локации/объекты 1280x720).

  ЭТАП 2: видео напрямую из референсов — video_minimax_refs.json
          (MiniMaxH3ReferenceToVideo, нода 137). Промпт собирается как
          full-reference rewrite из шести секций (гайд MiniMax H3,
          VIDEO_PROMPT_WRITING_GUIDE_ref_en.md §1): маркеры [base_name]
          в video_prompt сцены подменяются на <Subject N>, сами картинки
          цитируются как <Picture N> внутри subject_definitions, референсы
          подключаются к слотам ref_image_0..4 (лоадеры 148..152).
          Модель сама строит кадр из рефов и текста промпта; реплики
          с липсинком уже прописаны внутри video_prompt плана и
          уходят в модель как есть; массив dialogue нужен только
          для субтитров. Аудио вшито в клип.
          Клип до 15 секунд, multishot: video_prompt делится на планы
          маркерами [Shot 1] / [Shot N] At MM:SS.mmm (родной формат
          MiniMax H3). Шоты, реплики <d>[Russian] ...</d>, ID голосов
          (S1)/(S2) и подачу пишет планировщик в video_prompt —
          код ничего в промпт не добавляет.
          Тембр голоса персонажа фиксируется референсным аудио: нода 137
          принимает ref_audios.ref_audio_0..2 (лоадеры VHS_LoadAudio
          инжектятся программно). speaker_ref -> voice_refs.json /
          voice_ref -> <Audio N> в промпте — тембр стабилен от клипа
          к клипу.

  ЭТАП 3: финалка — concat клипов со звуком -> (опц.) фоновая музыка
          -25dB -> (опц.) RIFE -> (опц.) субтитры из dialogue (тайминг внутри сцены
          по окнам шотов [Shot N]) -> обложка из первого кадра первого
          клипа.
          Если клипов больше чем на FINAL_MAX_PART_S (2:30), финалка
          делится по границам сцен на части примерно равной длины,
          каждая собирается в свою папку part_1/, part_2/, ...

Что удалено относительно прежней версии пайплайна:
  - Этап генерации стартовых кадров (image_minimax.json /
    MiniMaxH3ReferenceToVideo -> SaveImage): MiniMax H3 умеет
    reference-to-video напрямую, промежуточный still не нужен.
  - Поле image_prompt в сценах main.json: композиция кадра теперь
    описывается в самом video_prompt вместе с motion и камерой
    (см. GROK_Prompt_MINIMAX.txt).
  - Паддинг в 1024x1820 (9:16): видео квадратное VIDEO_WIDTH x VIDEO_HEIGHT.

Ожидаемый формат main.json — см. GROK_Prompt_MINIMAX.txt:
  {
    "cover_text": "...",
    "base_prompts": [{base_id, base_name, base_prompt, entity_type,
                      voice_ref?}],   # путь к образцу голоса (character)
    "scenes": [{
      scene_id, scene_state{...}, previous_recap, must_carry_over,
      duration_s,          # 4..15 секунд (рекомендуется 8..15)
      video_prompt,        # multishot: [Shot 1] ... [Shot N] At MM:SS.mmm;
                           # композиция + motion + камера, [base_name] маркеры (<=5)
      scene_summary,       # англ. фраза для секции summary
      soundscape,          # англ. фраза для секции overall_soundscape
      dialogue: [{speaker, kind: "speech", line,
                  shot?,          # номер шота (1-based); обязателен в multishot
                  speaker_ref?,   # base_name говорящего (для субтитров)
                  delivery?}]     # англ. фраза подачи голоса
    }]
  }

Запуск: положить main.json рядом с этим файлом, поднять ComfyUI
(скрипт стартует его сам через ComfyManager), python generate_scenes_minimax.py

Перегенерация отдельных сцен уже готового проекта:

  python generate_scenes_minimax.py --project-dir <папка_запуска> --scenes 3 7 12

  --project-dir  папка запуска с клипами (та, что скрипт печатает как
                 «Выходная папка»); абсолютный путь, путь от текущей
                 папки или имя папки внутри OUTPUT_ROOT.
  --scenes       номера сцен (1-based, как префикс файла {N}_*.mp4):
                 `3 7 12`, `3,7,12`, диапазоны `4-6`.
  --plan         путь к плану (по умолчанию main.json). Перед
                 перегенерацией можно поправить video_prompt нужных сцен.

Что происходит: перегенерируются только выбранные сцены (референсы
берутся готовые), старые клипы и прежняя финалка уходят в
<project-dir>/replaced/<время>/, затем concat / музыка / RIFE /
субтитры / обложки собираются заново из всех клипов проекта.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import copy
import datetime
import glob as _glob
import json
import os
import re
import secrets
import shutil
import sys

from subs_align import build_aligned_ass

import aiohttp
import websockets

sys.path.append("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from comfy_manager import ComfyManager
from prompt_builder import (
    ENTITY_TYPES,
    entity_ref_filename,
    seed_prompt_for_entity,
)

COMFYUI_URL = "127.0.0.1:8188"
COMFY_ROOT = r"C:\Users\Loopy\Desktop\comfyui"
OUTPUT_ROOT = r"C:\Users\Loopy\Desktop\comfyui\ComfyUI\output"
REFS_DIR = os.path.join(OUTPUT_ROOT, "base_image_gen")

# Размеры кадра видео. Квадрат 640x640; если нужен другой размер —
# поменять тут, workflow патчится из этих констант.
VIDEO_WIDTH, VIDEO_HEIGHT = 832, 480
OUT_FPS = 24                    # fps узла CreateVideo (155); финалка строго CFR

# Длительность сцены (сек). GROK-гайд пишет 4..15 (рекомендует 8..15) —
# здесь страховочный кламп, чтобы не уронить генерацию экстремальными
# значениями. 15 сек — потолок длины одного клипа у MiniMax H3.
MINIMAX_MIN_DURATION_S = 4.0
MINIMAX_MAX_DURATION_S = 15.0

# Перезапуск ComfyUI каждые N сгенерированных клипов: чистит VRAM/ОЗУ
# от утечек и «залипших» моделей (после ~15 клипов подряд речь могла
# уезжать с русского на китайский). 0 — не перезапускать.
RESTART_EVERY_N_SCENES = 1

# Деление финалки на части. Если суммарная длина клипов больше
# FINAL_MAX_PART_S, финалка режется на ceil(длина / FINAL_MAX_PART_S)
# частей примерно равной длины. Режется только по границам сцен —
# клип никогда не разрезается. Каждая часть собирается отдельно
# (своя музыка, субтитры, обложки) в папку part_1/, part_2/, ...
# 0 — не делить.
FINAL_MAX_PART_S = 150.0        # 2:30

# Файлы финалки, которые пересобираются при перегенерации сцен.
FINAL_ARTIFACTS = (
    "list.txt", "output.mp4", "video_music.mp4", "video_interp_v.mp4",
    "video_interp.mp4", "subs_dialogue.ass", "video_final.mp4",
    "cover_frame.png", "result_cover_16x9.jpg", "result_cover_9x16.jpg",
)

# Опциональные пост-этапы финалки.
BURN_SUBTITLES = True      # прожигать ASS-субтитры в финальное видео
SUBS_LANGUAGE = "ru"       # язык forced alignment (WhisperX / wav2vec2)
MIX_BACKGROUND_MUSIC = True # если audio.mp3 лежит рядом — подмешать -25dB

# ---- интерполяция кадров (RIFE ncnn-vulkan, отдельный .exe) ----
RIFE_INTERPOLATION = False      # True: 24 fps -> 48 fps перед субтитрами
RIFE_MULTIPLIER = 2
RIFE_TARGET_FPS = None          # None -> исходный fps * multiplier
RIFE_CRF = 16


def _resolve_bin(name: str) -> str | None:
    """ffmpeg/ffprobe: PATH, затем бинарь из imageio-ffmpeg."""
    exe = shutil.which(name)
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        base = imageio_ffmpeg.get_ffmpeg_exe()
        if name == "ffmpeg":
            return base
        cand = os.path.join(
            os.path.dirname(base), name + (".exe" if os.name == "nt" else "")
        )
        if os.path.exists(cand):
            return cand
    except Exception:
        pass
    return None


FFMPEG_BIN = _resolve_bin("ffmpeg")
FFPROBE_BIN = _resolve_bin("ffprobe")


# Голосовые референсы персонажей (voice timbre -> <Audio N> в промпте).
# Сайдкар voice_refs.json рядом со скриптом: {"lin_jie": "C:/.../lin_jie.wav"}
# — машинный конфиг, переживает перегенерацию main.json GROK'ом.
# Переопределение на уровне плана: поле voice_ref в base_prompts[].
# Лимит ref_audios у MiniMaxH3ReferenceToVideo: 0..3 на один клип.
VOICE_REFS_PATH = os.path.join(BASE_DIR, "voice_refs.json")
MAX_AUDIO_REFS = 3
# Лимит H3-Base-Ref2VA: <=9 изображений, <=3 видео, <=3 аудио, <=12 файлов
# всего. В workflow-json развёрнуто пять лоадеров (148..152); слоты 6..9
# инжектятся программно, как и аудио.
MAX_IMAGE_REFS = 9
# VHS-лоадер аудио с произвольного пути (аналог VHS_LoadImagePath для
# картинок): class VHS_LoadAudio, виджет audio_file (wav/mp3/ogg/m4a/flac;
# seek_seconds/duration по умолчанию 0 = файл целиком).
IMAGE_LOADER_CLASS = "VHS_LoadImagePath"
AUDIO_LOADER_CLASS = "VHS_LoadAudio"
AUDIO_LOADER_INPUT = "audio_file"


# ============================================================ prompts

_REF_TOKEN_RE = re.compile(r"\[([a-zA-Z0-9_]+)\]")

# Реплики модели: <d>[Russian] ...</d>. Внутри <d> тоже стоят квадратные
# скобки (тег языка), это НЕ ref-маркер — такие участки при разборе и
# подмене [base_name] -> <Picture N> пропускаются как есть.
_D_TAG_RE = re.compile(r"(<d>.*?</d>)", re.DOTALL)


def _split_dialogue_spans(text: str) -> list[tuple[str, bool]]:
    """Режет текст на куски (chunk, is_dialogue). is_dialogue=True для <d>...</d>."""
    parts = _D_TAG_RE.split(text)
    return [(chunk, i % 2 == 1) for i, chunk in enumerate(parts)]


def ref_tokens_outside_dialogue(text: str) -> list[str]:
    """Все [base_name]-маркеры вне <d>...</d>, в порядке появления."""
    out: list[str] = []
    for chunk, is_dialogue in _split_dialogue_spans(text):
        if not is_dialogue:
            out.extend(_REF_TOKEN_RE.findall(chunk))
    return out

# Multishot-маркеры родного формата MiniMax H3: [Shot 1] без таймстампа,
# далее [Shot N] At MM:SS.mmm — время склейки от начала клипа.
_SHOT_TOKEN_RE = re.compile(
    r"\[Shot\s+(\d+)\](?:\s+At\s+(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?)?",
    re.IGNORECASE,
)

# Style-якорь ставится ПЕРЕД [Shot 1] — так требует full-reference режим
# MiniMax H3 (VIDEO_PROMPT_WRITING_GUIDE_ref_en.md, §5.2).
# ---- визуальный стиль ----
# STYLE_ANCHOR клеится в начало каждого video_prompt для H3.
# STYLE_TAIL — хвост, которым планировщик обязан закрывать каждый
# base_prompt (ЭТАП 1, Krea). Рефы решают больше, чем якорь: если реф
# нарисован в японском стиле, текстом в видео это уже не переломить.
# Меняешь стиль — меняй ОБА поля и хвост в GROK_Prompt_ACTION.txt.

STYLE_PRESET = "jp"   # "jp" — 2D modern anime, "cn" — китайская донхуа

STYLE_PRESETS = {
    "jp": {
        "anchor": "2D Modern Anime Style.",
        "tail": "2D Modern Anime Style",
    },
    "cn": {
        "anchor": (
            "Chinese donghua style, cel-shaded 3D CG animation, glossy "
            "subsurface skin shading, cyan rim light, warm key light, bloom, "
            "high saturation cinematic grading."
        ),
        "tail": (
            "Chinese donghua cel-shaded 3D style, narrow refined jawline, "
            "long straight nose bridge, upturned phoenix eyes with glossy "
            "irises and sharp specular highlights, smooth gradient-shaded "
            "hair with sheen bands, soft subsurface skin without hard cel "
            "shadows"
        ),
    },
}

STYLE_ANCHOR = STYLE_PRESETS[STYLE_PRESET]["anchor"]
STYLE_TAIL = STYLE_PRESETS[STYLE_PRESET]["tail"]


def parse_shot_markers(video_prompt: str) -> list[dict]:
    """Находит multishot-маркеры [Shot N] в video_prompt.

    Возвращает список {"n": int, "start_s": float | None} в порядке
    появления (start_s=None для шота без таймстампа, т.е. [Shot 1]).
    Пустой список = одношотовая сцена без разметки.
    """
    out = []
    for m in _SHOT_TOKEN_RE.finditer(video_prompt):
        start_s = None
        if m.group(2) is not None:
            ms = (m.group(4) or "0").ljust(3, "0")
            start_s = int(m.group(2)) * 60 + int(m.group(3)) + int(ms) / 1000.0
        out.append({"n": int(m.group(1)), "start_s": start_s})
    return out


# [base_name] (S3) — говорящий и его старый ID. Внутри одной сцены ID
# перенумеровываются с S1 по порядку реальных вокальных событий: так
# требует гайд (§5.4), а каждый клип для модели — отдельное видео.
_SPEAKER_TOKEN_RE = re.compile(r"\[([a-zA-Z0-9_]+)\]\s*\((S\d+)\)")
_SPEAKER_ID_RE = re.compile(r"\((S\d+)\)")

_ENTITY_NOUN = {
    "character": "character",
    "location": "environment",
    "object": "object",
}


def _split_into_shot_blocks(prompt_text: str) -> list[str]:
    """Режет video_prompt на блоки по [Shot N]; маркер остаётся в блоке."""
    matches = list(_SHOT_TOKEN_RE.finditer(prompt_text))
    if not matches:
        return [prompt_text]
    blocks = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else len(prompt_text)
        start = m.start() if k else 0
        blocks.append(prompt_text[start:end])
    return blocks


def scene_speaker_order(prompt_text: str) -> list[str]:
    """base_name говорящих в порядке их первой реплики (вне <d>...</d>)."""
    out: list[str] = []
    for chunk, is_dialogue in _split_dialogue_spans(prompt_text):
        if is_dialogue:
            continue
        for name, _sid in _SPEAKER_TOKEN_RE.findall(chunk):
            if name not in out:
                out.append(name)
    return out


def build_minimax_ref_prompt(
    prompt_text: str,
    base_prompts_by_name: dict,
    max_pictures: int = MAX_IMAGE_REFS,
) -> tuple[str, list[str], dict[str, int]]:
    """Подменяет [base_name] на <Subject N> и перенумеровывает (Sx).

    Гайд full-reference режима (§2.1, §2.2): картинка, которая только
    задаёт персонажа/локацию, НЕ адресуется в теле как <Picture N> —
    она цитируется в определении <Subject N>, а тело работает с
    субъектами. Слот <Subject N> совпадает по номеру с <Picture N>,
    то есть с порядком ref_image_0..4 (лоадеры 148..152).

    Возвращает (body_prompt, ordered_entity_names, slot_by_name).
    """
    tokens = ref_tokens_outside_dialogue(prompt_text)
    seen: list[str] = []
    for t in tokens:
        if t in base_prompts_by_name and t not in seen:
            seen.append(t)

    def _sort_key(name: str) -> int:
        # location всегда в конец — она фон, персонажи/объекты важнее для identity
        et = base_prompts_by_name[name].get("entity_type")
        return 1 if et == "location" else 0

    ordered = sorted(seen, key=_sort_key)[:max_pictures]
    slot_by_name = {name: i + 1 for i, name in enumerate(ordered)}

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in base_prompts_by_name:
            # чужие скобки (например тег языка) остаются нетронутыми
            return m.group(0)
        slot = slot_by_name.get(name)
        # entity за пределами cap'а раскрываем plain-текстом, без слота
        return f"<Subject {slot}>" if slot else name.replace("_", " ")

    body = "".join(
        chunk if is_dialogue else _REF_TOKEN_RE.sub(_sub, chunk)
        for chunk, is_dialogue in _split_dialogue_spans(prompt_text)
    )

    # (Sx) -> порядок вокальных событий этой сцены, начиная с S1
    speakers = scene_speaker_order(prompt_text)
    old_ids: list[str] = []
    for name, sid in _SPEAKER_TOKEN_RE.findall(prompt_text):
        if sid not in old_ids:
            old_ids.append(sid)
    remap = {old: f"S{k + 1}" for k, old in enumerate(old_ids)}
    if remap:
        body = "".join(
            chunk if is_dialogue
            else _SPEAKER_ID_RE.sub(
                lambda m: f"(\x00{remap.get(m.group(1), m.group(1))}\x00)", chunk
            )
            for chunk, is_dialogue in _split_dialogue_spans(body)
        ).replace("\x00", "")
    return body, ordered, slot_by_name


def compose_minimax_scene_prompt(
    scene: dict,
    base_prompts_by_name: dict,
    ordered: list[str],
    slot_by_name: dict[str, int],
    body_prompt: str,
    voice_speakers: list[str],
    style_anchor: str,
    language: str = "Russian",
) -> str:
    """Собирает шесть секций full-reference rewrite (гайд §1).

    subject_definitions / summary / retention_analysis /
    detailed_description / overall_soundscape / non_diegetic_music.
    Всё, кроме detailed_description, генерируется кодом из base_prompts
    и разметки сцены — планировщику это писать не нужно.
    """
    raw = scene.get("video_prompt") or ""

    # --- в каких шотах появляется каждая сущность
    shots_by_name: dict[str, list[int]] = {}
    for si, block in enumerate(_split_into_shot_blocks(raw), start=1):
        for name in set(ref_tokens_outside_dialogue(block)):
            if name in slot_by_name:
                shots_by_name.setdefault(name, []).append(si)

    defs: list[str] = []
    retention: list[str] = []
    for name in ordered:
        slot = slot_by_name[name]
        meta = base_prompts_by_name[name]
        noun = _ENTITY_NOUN.get(meta.get("entity_type"), "subject")
        desc = (meta.get("base_prompt") or "").strip().rstrip(".")
        defs.append(f"<Subject {slot}> is the {noun} in <Picture {slot}>: {desc}.")
        shots = shots_by_name.get(name) or [1]
        where = ", ".join(f"[Shot {n}]" for n in shots)
        retention.append(
            f"<Subject {slot}> (appears in {where}): fully_preserved - the "
            f"referenced appearance, outfit and defining features are kept "
            f"unchanged for the whole clip."
        )

    # --- голосовые рефы: <Audio N> ↔ говорящий ↔ (Sx) по порядку реплик
    for k, name in enumerate(voice_speakers, start=1):
        slot = slot_by_name.get(name)
        who = f"<Subject {slot}>" if slot else name.replace("_", " ")
        defs.append(
            f"<Audio {k}> is the voice-timbre reference for {who} (S{k}), "
            f"containing a spoken {language} vocal layer."
        )
        retention.append(
            f"<Audio {k}>: reference - the target speaker follows its voice "
            f"timbre and delivery without copying the original signal."
        )

    n_shots = max(1, len(parse_shot_markers(raw)))
    task = ("[reference generation + audio reference]" if voice_speakers
            else "[reference generation]")
    summary = (scene.get("scene_summary") or "").strip()
    if summary:
        summary = _REF_TOKEN_RE.sub(
            lambda m: (f"<Subject {slot_by_name[m.group(1)]}>"
                       if m.group(1) in slot_by_name
                       else m.group(1).replace("_", " ")),
            summary,
        )
    else:
        subjects = ", ".join(f"<Subject {slot_by_name[n]}>" for n in ordered)
        summary = (
            f"The target video is a {n_shots}-shot continuous scene built from "
            f"{subjects}; the referenced subjects act and speak on screen."
        )

    soundscape = (scene.get("soundscape") or "").strip() or (
        "Room tone of the described location and the physical sounds of the "
        "shown actions continue throughout the clip, with no added score."
    )

    return (
        "subject_definitions:\n" + "\n".join(defs) + "\n\n"
        "summary:\n" + f"{task} {summary}" + "\n\n"
        "retention_analysis:\n" + "\n".join(retention) + "\n\n"
        "detailed_description:\n" + f"{style_anchor}\n{body_prompt.strip()}" + "\n\n"
        "overall_soundscape:\n" + soundscape + "\n\n"
        "non_diegetic_music:\nN/A\n"
    )


def patch_minimax_refs_video_workflow(
    workflow: dict,
    image_paths: list[str],      # 1..9 абсолютных путей к entity-рефам
    positive_prompt: str,        # compose_minimax_video_prompt(...) с <Picture N>
    duration_s: float,           # секунды; нода 137 принимает length в СЕКУНДАХ
    filename_prefix: str,
    seed: int | None = None,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    audio_paths: list[str] | None = None,  # 0..3 голосовых рефа -> <Audio 1..3>
) -> dict:
    """��атчит video_minimax_refs.json под одну сцену.

    Ноды video_minimax_refs.json:
      137 MiniMaxH3ReferenceToVideo — prompt, width/height, length (СЕКУНДЫ,
          в отличие от MiniMaxH3ImageToVideo, где length задавался в кадрах
          через MathExpression), ref_images.ref_image_0..4 <- лоадеры
          148..152. Неиспользуемые слоты ref_image_N удаляем: лоадеры без
          рёбер становятся unreachable и ComfyUI их не выполняет.
      132 RandomNoise             — noise_seed
      156 SaveVideo               — filename_prefix (аудио вшито в клип)
      155 CreateVideo             — fps 24 (не трогаем)
      161 MiniMaxH3SigmaShift     — shift video/audio (не трогаем)
      + программно добавляемые VHS_LoadAudio (id 162..164) — голосовые
          рефы в ref_audios.ref_audio_0..2; подключены только когда у
          говорящих в сцене есть voice_ref
    """
    wf = copy.deepcopy(workflow)
    n = len(image_paths)
    if not 1 <= n <= MAX_IMAGE_REFS:
        raise ValueError(
            f"image_paths должен иметь 1..{MAX_IMAGE_REFS} элементов, got {n}"
        )

    node_inputs = wf["137"]["inputs"]
    # слоты 0..4 уже развёрнуты в workflow-json лоадерами 148..152,
    # слоты 5..8 добавляем программно (id 170+ свободны)
    preset_loaders = ["148", "149", "150", "151", "152"]
    for i, path in enumerate(image_paths):
        if i < len(preset_loaders):
            wf[preset_loaders[i]]["inputs"]["image"] = path
            continue
        loader_id = str(170 + i - len(preset_loaders))
        wf[loader_id] = {
            "class_type": IMAGE_LOADER_CLASS,
            "inputs": {"image": path, "custom_width": 0, "custom_height": 0},
        }
        node_inputs[f"ref_images.ref_image_{i}"] = [loader_id, 0]

    # отрезаем неиспользуемые ref-слоты из развёрнутых в json
    for slot in range(n, len(preset_loaders)):
        node_inputs.pop(f"ref_images.ref_image_{slot}", None)

    # голосовые референсы: ref_audios.ref_audio_0..2 (лимит ноды 0..3).
    # VHS-лоадеры инжектим программно, чтобы workflow-json оставался
    # чистым; id 162+ свободны (текущий максимум workflow — 161).
    for j, path in enumerate(audio_paths or []):
        loader_id = str(162 + j)
        wf[loader_id] = {
            "class_type": AUDIO_LOADER_CLASS,
            "inputs": {AUDIO_LOADER_INPUT: path},
        }
        node_inputs[f"ref_audios.ref_audio_{j}"] = [loader_id, 0]

    duration_s = max(
        MINIMAX_MIN_DURATION_S, min(MINIMAX_MAX_DURATION_S, float(duration_s))
    )
    node_inputs["prompt"] = positive_prompt
    node_inputs["width"] = width
    node_inputs["height"] = height

    seconds_node = wf["158"]["inputs"]
    seconds_node["value"] = duration_s

    wf["156"]["inputs"]["filename_prefix"] = filename_prefix
    if seed is not None:
        wf["132"]["inputs"]["noise_seed"] = int(seed)
    return wf


# ============================================================ fs helpers


def get_unique_dir_name(base_name: str) -> str:
    if not os.path.exists(base_name):
        return base_name
    counter = 2
    while True:
        new_name = f"{base_name}_{counter}"
        if not os.path.exists(new_name):
            return new_name
        counter += 1


def find_video_for_scene(unique_path: str, scene_idx: int) -> str | None:
    """Самый свежий {scene_idx+1}_*.mp4, записанный SaveVideo для сцены."""
    pattern = os.path.join(unique_path, f"{scene_idx + 1}_*.mp4")
    matches = sorted(_glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def parse_scene_numbers(tokens: list[str], n_scenes: int) -> list[int]:
    """`3 7 12`, `3,7,12`, `4-6` -> отсортированные 0-based индексы."""
    out: set[int] = set()
    for tok in tokens:
        for part in str(tok).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                if "-" in part:
                    a, b = part.split("-", 1)
                    nums = range(int(a), int(b) + 1)
                else:
                    nums = [int(part)]
            except ValueError:
                raise SystemExit(f"--scenes: не понимаю {part!r}")
            for n in nums:
                if not (1 <= n <= n_scenes):
                    raise SystemExit(
                        f"--scenes: сцены {n} нет (в плане 1..{n_scenes})"
                    )
                out.add(n - 1)
    if not out:
        raise SystemExit("--scenes: не указано ни одной сцены")
    return sorted(out)


def resolve_project_dir(path: str) -> str:
    """Абсолютный путь, путь от cwd или имя папки внутри OUTPUT_ROOT."""
    for cand in (path, os.path.join(OUTPUT_ROOT, path)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise SystemExit(f"--project-dir: папка не найдена: {path}")


def archive_files(paths: list[str], archive_dir: str) -> None:
    """Переносит файлы в archive_dir (создаётся лениво), ничего не удаляя."""
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return
    os.makedirs(archive_dir, exist_ok=True)
    for p in paths:
        dst = os.path.join(archive_dir, os.path.basename(p))
        if os.path.exists(dst):
            base, ext = os.path.splitext(dst)
            k = 2
            while os.path.exists(f"{base}_{k}{ext}"):
                k += 1
            dst = f"{base}_{k}{ext}"
        shutil.move(p, dst)


def split_into_parts(durations: list[float], max_part_s: float) -> list[list[int]]:
    """Делит клипы (по порядку) на части не длиннее max_part_s.

    Число частей = ceil(сумма / max_part_s); внутри этого числа точки
    разреза подбираются так, чтобы самая длинная часть была как можно
    короче (части выходят примерно равными). Возвращает списки индексов
    в durations. Если max_part_s <= 0 или всё влезает — одна часть.
    """
    n = len(durations)
    total = sum(durations)
    if n == 0:
        return []
    if max_part_s <= 0 or total <= max_part_s:
        return [list(range(n))]
    k = min(n, math.ceil(total / max_part_s))
    pref = [0.0]
    for d in durations:
        pref.append(pref[-1] + d)
    INF = float("inf")
    # best[j][i] — минимальная «самая длинная часть» для первых i клипов в j частях
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for p in range(j - 1, i):
                v = max(best[j - 1][p], pref[i] - pref[p])
                if v < best[j][i]:
                    best[j][i] = v
                    cut[j][i] = p
    parts: list[list[int]] = []
    i = n
    for j in range(k, 0, -1):
        p = cut[j][i]
        parts.append(list(range(p, i)))
        i = p
    parts.reverse()
    return parts


async def run_ffmpeg(cmd: list[str], cwd: str) -> None:
    if not FFMPEG_BIN:
        raise RuntimeError(
            "ffmpeg не найден: поставь его в PATH "
            "(winget install --id=Gyan.FFmpeg -e) или "
            "uv pip install imageio-ffmpeg"
        )
    cmd = [FFMPEG_BIN, *cmd[1:]]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg упал ({' '.join(cmd[:3])}...): "
            f"{stderr.decode(errors='ignore')[-500:]}"
        )


async def build_final_video(
    out_dir: str,
    clip_paths: list[str],
    part_scenes: list[dict],
    *,
    cover_text: str,
    cover_clip: str | None,
    cover_t: float,
    part_label: str | None,
) -> str:
    """Собирает одну финалку из clip_paths в out_dir:
    concat -> музыка -> RIFE -> субтитры -> обложки. Возвращает путь видео."""
    os.makedirs(out_dir, exist_ok=True)
    clip_paths = [os.path.abspath(p) for p in clip_paths]

    with open(os.path.join(out_dir, "list.txt"), "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")

    # concat ЧЕРЕЗ ФИЛЬТР (не -c copy!):
    #   у MiniMax-клипов длина аудиодорожки не совпадает с длиной видео
    #   (AAC-кадры по 1024 сэмпла + encoder delay/priming в каждом файле).
    #   При `-f concat -c copy` эти хвосты складываются, и рассинхрон
    #   накапливается от клипа к клипу; плееры вроде TikTok/мобильных
    #   декодеров игнорируют edit list и показывают дрейф звука.
    #   Пересборка с явным CFR-видео и непрерывным аудио убирает дрейф.
    print("  concat клипов...")
    n_clips = len(clip_paths)
    in_args: list[str] = []
    for p in clip_paths:
        in_args += ["-i", p]
    chains = "".join(
        f"[{k}:v]fps={OUT_FPS},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=bicubic,"
        f"setsar=1,format=yuv420p,setpts=PTS-STARTPTS[v{k}];"
        f"[{k}:a]aresample=48000:async=1:first_pts=0,"
        f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"asetpts=PTS-STARTPTS[a{k}];"
        for k in range(n_clips)
    )
    pairs = "".join(f"[v{k}][a{k}]" for k in range(n_clips))
    filter_complex = f"{chains}{pairs}concat=n={n_clips}:v=1:a=1[v][a]"
    await run_ffmpeg(
        ["ffmpeg", "-y", *in_args,
         "-filter_complex", filter_complex,
         "-map", "[v]", "-map", "[a]",
         "-fps_mode", "cfr", "-r", str(OUT_FPS),
         "-video_track_timescale", str(OUT_FPS * 1000),
         "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
         "-movflags", "+faststart",
         "output.mp4"],
        cwd=out_dir,
    )
    current = "output.mp4"

    # -------- опционально: фоновая музыка -25dB
    music_src = os.path.join(OUTPUT_ROOT, "AI_VIDEO", "audio2.mp3")
    if MIX_BACKGROUND_MUSIC and os.path.exists(music_src):
        print("  подмешиваю фоновую музыку...")
        shutil.copy2(music_src, os.path.join(out_dir, "audio2.mp3"))
        await run_ffmpeg(
            ["ffmpeg", "-y", "-i", current, "-stream_loop", "-1",
             "-i", "audio2.mp3",
             "-filter_complex",
             "[1:a]volume=-25dB[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
             "-map", "0:v", "-map", "[aout]",
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-ar", "48000", "-ac", "2",
             "-movflags", "+faststart",
             "video_music.mp4"],
            cwd=out_dir,
        )
        current = "video_music.mp4"

    # -------- опционально: интерполяция кадров RIFE
    # Строго ДО прожига субтитров: иначе интерполируются сами буквы
    # и титры начинают двоиться. Звук RIFE не переносит (собирает видео
    # из PNG) — возвращаем его ремуксом из исходного файла.
    if RIFE_INTERPOLATION:
        try:
            from rife_interp import rife_interpolate

            print(f"  интерполяция RIFE x{RIFE_MULTIPLIER}...")
            await asyncio.to_thread(
                rife_interpolate,
                os.path.join(out_dir, current),
                os.path.join(out_dir, "video_interp_v.mp4"),
                RIFE_MULTIPLIER,
                RIFE_TARGET_FPS,
                None,
                None,
                RIFE_CRF,
            )
            await run_ffmpeg(
                ["ffmpeg", "-y",
                 "-i", "video_interp_v.mp4", "-i", current,
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "copy", "-shortest",
                 "video_interp.mp4"],
                cwd=out_dir,
            )
            current = "video_interp.mp4"
        except Exception as exc:
            print(f"  [WARN] интерполяция пропущена: {exc}")

    # -------- субтитры: речь в кадре, жёлтые снизу
    if BURN_SUBTITLES:
        print("  субтитры...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # шрифт кладём рядом с .ass и указываем fontsdir="." — так в
        # filtergraph нет двоеточий/бэкслешей винды, которые ffmpeg
        # молча съедал вместе с путём к шрифту
        for _f in ("Montserrat-VariableFont_wght.ttf", "Montserrat-Bold.ttf"):
            _src = os.path.join(script_dir, _f)
            if os.path.exists(_src):
                shutil.copy2(_src, os.path.join(out_dir, _f))

        dlg_ass = os.path.join(out_dir, "subs_dialogue.ass")
        print("  WhisperX forced alignment по клипам...")
        n_events = await asyncio.to_thread(
            build_aligned_ass,
            part_scenes,
            [(p, 0.0, 0.0) for p in clip_paths],
            dlg_ass,
            ffmpeg_bin=FFMPEG_BIN,
            ffprobe_bin=FFPROBE_BIN,
            video_width=VIDEO_WIDTH,
            video_height=VIDEO_HEIGHT,
            language=SUBS_LANGUAGE,
            max_words=SUB_MAX_WORDS,
            max_chars=SUB_MAX_CHARS,
        )

        if n_events <= 0:
            print("  [WARN] в .ass нет ни одного события — прожиг пропущен "
                  "(проверь dialogue[] в main.json и лог выравнивания)")
        else:
            await run_ffmpeg(
                ["ffmpeg", "-y", "-i", current,
                 "-vf", "ass=subs_dialogue.ass:fontsdir=.",
                 "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                 "-tune", "animation", "-pix_fmt", "yuv420p",
                 "-fps_mode", "cfr", "-r", str(OUT_FPS),
                 "-video_track_timescale", str(OUT_FPS * 1000),
                 "-movflags", "+faststart",
                 "-af", "aresample=48000:async=1:first_pts=0",
                 "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
                 "video_final.mp4"],
                cwd=out_dir,
            )
            current = "video_final.mp4"

    final_path = os.path.join(out_dir, current)
    print(f"  видео: {final_path}")

    # -------- обложки (16x9 и 9x16): кадр cover_clip @ cover_t
    if cover_text and cover_clip:
        print("  обложки...")
        await run_ffmpeg(
            ["ffmpeg", "-y", "-ss", f"{cover_t:.3f}",
             "-i", os.path.abspath(cover_clip),
             "-frames:v", "1", "cover_frame.png"],
            cwd=out_dir,
        )
        from create_cover import create_covers
        create_covers(
            input_image_path=os.path.join(out_dir, "cover_frame.png"),
            text=cover_text,
            output_dir=out_dir,
            part_label=part_label,
        )
    return final_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="MiniMax H3 ref2video pipeline. Без флагов — полный "
                    "прогон; с --project-dir и --scenes — перегенерация "
                    "выбранных сцен и пересборка финалки."
    )
    ap.add_argument("--project-dir", "--regen-dir", dest="project_dir",
                    help="папка запуска с готовыми клипами")
    ap.add_argument("--scenes", nargs="+", metavar="N",
                    help="номера сцен для перегенерации: 3 7 12 | 3,7,12 | 4-6")
    ap.add_argument("--plan", default="main.json",
                    help="файл плана (по умолчанию main.json)")
    args = ap.parse_args()
    if bool(args.project_dir) != bool(args.scenes):
        ap.error("--project-dir и --scenes нужно указывать вместе")
    return args


def load_voice_refs(plan: dict) -> dict[str, str]:
    """base_name -> абс. путь к голосовому референсу (wav/mp3, 2..15 сек).

    Источники (поле voice_ref в base_prompts побеждает сайдкар):
      1) voice_refs.json рядом со скриптом — машинный конфиг, переживает
         перегенерацию main.json;
      2) voice_ref в base_prompts[] — на уровне конкретного плана.
    Несуществующий файл / не-character / не���звестный base_name -> WARN и
    пропуск: сцена уйдёт в генерацию без аудио-рефа, как раньше.
    """
    refs: dict[str, str] = {}
    if os.path.exists(VOICE_REFS_PATH):
        try:
            with open(VOICE_REFS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                refs.update({str(k): str(v) for k, v in raw.items()})
        except Exception as e:
            print(f"  [WARN] voice_refs.json не читается: {e}")
    types_by_name = {
        bp.get("base_name"): bp.get("entity_type")
        for bp in plan.get("base_prompts", [])
    }
    for bp in plan.get("base_prompts", []):
        vr = (bp.get("voice_ref") or "").strip() if isinstance(bp, dict) else ""
        if vr:
            refs[bp.get("base_name", "")] = vr
    out: dict[str, str] = {}
    for name, path in refs.items():
        if name not in types_by_name:
            print(
                f"  [WARN] voice ref для неизвестного base_name {name!r} — "
                f"пропуск"
            )
            continue
        if types_by_name[name] != "character":
            print(
                f"  [WARN] voice ref у {name} ({types_by_name[name]}) — "
                f"голос есть смысл вешать только на character; пропуск"
            )
            continue
        p = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
        if not os.path.exists(p):
            print(f"  [WARN] voice ref {name}: файл не найден: {p} — пропуск")
            continue
        out[name] = p
    return out


# ============================================================ subtitles


SUB_MAX_CHARS = 30          # символов в одном субтитр-событии
SUB_MAX_WORDS = 4           # слов в одном субтитр-событии


# ============================================================ утилиты


async def media_duration(path: str, fallback: float) -> float:
    if not FFPROBE_BIN:
        return fallback
    proc = await asyncio.create_subprocess_exec(
        FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    try:
        return float(out.decode(errors="ignore").strip())
    except (TypeError, ValueError):
        return fallback


# ============================================================ ComfyUI client


class ComfyUIClient:
    """Асинхронный клиент ComfyUI с дампами ошибок."""

    def __init__(self, server_address: str):
        self.server_address = server_address
        import uuid
        self.client_id = str(uuid.uuid4())

    async def queue_prompt(self, prompt_workflow: dict) -> str:
        """Ставит workflow в очередь. При HTTP 400 / отсутствии prompt_id
        дампит ответ + workflow в errors/{ts}_queue_validation/ и бросает
        RuntimeError со сводкой node_errors."""
        p = {"prompt": prompt_workflow, "client_id": self.client_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{self.server_address}/prompt", json=p
            ) as response:
                status = response.status
                raw_text = None
                data = None
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    try:
                        raw_text = await response.text()
                    except Exception:
                        raw_text = "<failed to read response body>"

                if isinstance(data, dict) and "prompt_id" in data:
                    return data["prompt_id"]

                self._save_queue_error_log(
                    status=status, raw_data=data,
                    raw_text=raw_text, workflow=prompt_workflow,
                )
                err = (data or {}).get("error") or {}
                node_errors = (data or {}).get("node_errors") or {}
                lines = []
                for nid, ninfo in list(node_errors.items())[:10]:
                    ninfo = ninfo or {}
                    for e in (ninfo.get("errors") or [])[:5]:
                        e = e or {}
                        lines.append(
                            f"    node {nid} ({ninfo.get('class_type', '?')}): "
                            f"{e.get('message', '?')}"
                        )
                raise RuntimeError(
                    f"ComfyUI /prompt отклонил workflow (HTTP {status}): "
                    f"{err.get('type', 'unknown')}: "
                    f"{err.get('message', '<no message>')}\n"
                    f"  node_errors:\n"
                    + ("\n".join(lines) if lines else "    <none>")
                )

    def _save_queue_error_log(self, *, status, raw_data=None, raw_text=None,
                              workflow=None):
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_queue_validation")
            counter, base_sub = 2, sub
            while os.path.exists(sub):
                sub = f"{base_sub}_{counter}"
                counter += 1
            os.makedirs(sub, exist_ok=True)
            text_lines = [
                "=== ComfyUI /prompt validation error ===",
                f"timestamp:   {ts}",
                f"http_status: {status}",
                "",
            ]
            if raw_data is not None:
                text_lines += [
                    "=== response.json ===",
                    json.dumps(raw_data, ensure_ascii=False, indent=2,
                               default=str),
                    "",
                ]
            if raw_text is not None:
                text_lines += ["=== response.text ===", raw_text, ""]
            with open(os.path.join(sub, "queue_error.txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            if workflow is not None:
                with open(os.path.join(sub, "workflow.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)
            print(f"  [ComfyUI ERROR] queue-лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить queue-лог: {e}")

    async def wait_for_execution(self, prompt_id: str, workflow=None) -> list:
        """Ждёт конца генерации по WS и возвращает имена выходных файлов."""
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        async with websockets.connect(ws_url) as ws:
            while True:
                out = await ws.recv()
                if not isinstance(out, str):
                    continue
                message = json.loads(out)
                mtype = message.get("type")

                if mtype == "execution_error":
                    data = message.get("data") or {}
                    if data.get("prompt_id") == prompt_id:
                        self._save_error_log(prompt_id, data, workflow)
                        print(
                            f"  [ComfyUI ERROR] node {data.get('node_id')} "
                            f"({data.get('node_type')}): "
                            f"{data.get('exception_message', '?')}"
                        )

                if mtype == "execution_interrupted":
                    data = message.get("data") or {}
                    if data.get("prompt_id") == prompt_id:
                        print("  [ComfyUI WARN] execution interrupted")
                        break

                if mtype == "executing":
                    data = message.get("data") or {}
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break

            return await self.get_history(prompt_id, workflow=workflow)

    def _save_error_log(self, prompt_id, error_data, workflow=None):
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_{prompt_id[:12]}")
            os.makedirs(sub, exist_ok=True)
            tb = error_data.get("traceback") or []
            tb_text = ("\n".join(str(x) for x in tb)
                       if isinstance(tb, list) else str(tb))
            text_lines = [
                "=== ComfyUI execution_error ===",
                f"timestamp:         {ts}",
                f"prompt_id:         {prompt_id}",
                f"node_id:           {error_data.get('node_id')}",
                f"node_type:         {error_data.get('node_type')}",
                f"exception_type:    {error_data.get('exception_type')}",
                f"exception_message: {error_data.get('exception_message')}",
                "",
                "=== traceback ===",
                tb_text,
            ]
            with open(os.path.join(sub, "error.txt"), "w",
                      encoding="utf-8") as f:
                f.write("\n".join(text_lines))
            if workflow is not None:
                with open(os.path.join(sub, "workflow.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)
            print(f"  [ComfyUI ERROR] лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить error-лог: {e}")

    async def get_history(self, prompt_id: str, workflow=None) -> list:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{self.server_address}/history/{prompt_id}"
            ) as response:
                history = await response.json(content_type=None)

                if prompt_id not in history:
                    print(
                        f"  [ComfyUI WARN] prompt_id {prompt_id} отсутствует "
                        f"в history (нода вероятно упала на validation)"
                    )
                    return []

                entry = history[prompt_id]
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    self._save_history_error(prompt_id, entry, workflow)

                output_files = []
                for node_id in entry.get("outputs") or {}:
                    node_output = entry["outputs"][node_id]
                    for key in ("images", "gifs", "videos", "audio"):
                        for item in node_output.get(key, []):
                            output_files.append(item["filename"])
                return output_files

    def _save_history_error(self, prompt_id, history_entry, workflow=None):
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_{prompt_id[:12]}_history")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(sub, "history.json"), "w",
                      encoding="utf-8") as f:
                json.dump(history_entry, f, ensure_ascii=False, indent=2,
                          default=str)
            if workflow is not None:
                with open(os.path.join(sub, "workflow.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)
            print(f"  [ComfyUI ERROR] history-лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить history-лог: {e}")


# ============================================================ schema validation


def validate_plan(plan: dict) -> None:
    """Строгая проверка main.json ref2video-формата (без image_prompt)."""
    if "scenes" not in plan or not isinstance(plan["scenes"], list):
        raise ValueError("main.json: нет массива 'scenes'")
    if "base_prompts" not in plan or not isinstance(plan["base_prompts"], list):
        raise ValueError("main.json: нет массива 'base_prompts'")
    if not plan["scenes"]:
        raise ValueError("main.json: пустой массив 'scenes'")
    if (plan.get("tts_text") or "").strip() or any(
        (sc.get("narration") or "").strip() for sc in plan["scenes"]
    ):
        print(
            "  [WARN] в плане есть tts_text/narration — закадровый голос "
            "из пайплайна удалён, эти поля игнорируются. План делается "
            "по GROK_Prompt_ACTION.txt."
        )

    base_names: set[str] = set()
    for i, bp in enumerate(plan["base_prompts"]):
        et = bp.get("entity_type")
        if et not in ENTITY_TYPES:
            raise ValueError(
                f"base_prompts[{i}] ({bp.get('base_name')!r}): entity_type "
                f"{et!r}, expected one of {ENTITY_TYPES}"
            )
        base_names.add(bp.get("base_name"))
        if bp.get("voice_ref") and et != "character":
            print(
                f"  [WARN] base_prompts[{i}] ({bp.get('base_name')!r}): "
                f"voice_ref на {et} — голосовой реф имеет смысл только "
                f"у character"
            )
        if et == "character" and not bp.get("voice_style"):
            print(
                f"  [WARN] base_prompts[{i}] ({bp.get('base_name')!r}): нет "
                f"'voice_style' — реплики персонажа будут звучать как "
                f"нейтральный диктор (см. ШАГ 1 в GROK_Prompt_MINIMAX.txt)"
            )

    # — мягкие проверки обложки (не ломают запуск) —
    if not plan.get("cover_text_short"):
        print(
            "  [WARN] нет 'cover_text_short' — на обложке будет длинный "
            "cover_text мелким кеглем"
        )
    if not plan.get("cover_scene_id"):
        print(
            "  [WARN] нет 'cover_scene_id' — кадр для обложки возьму из "
            "первого клипа (часто слабый кадр)"
        )

    for i, scene in enumerate(plan["scenes"]):
        sid = scene.get("scene_id", i)
        if "image_prompt" in scene:
            print(
                f"  [WARN] scene {sid}: поле 'image_prompt' больше не "
                f"используется (ref2video без стартового кадра) — игнорирую. "
                f"Перегенерируй план по новому GROK_Prompt_MINIMAX.txt."
            )
        for key in ("video_prompt", "duration_s"):
            if key not in scene:
                raise ValueError(f"scenes[{i}] (id={sid}): нет поля {key!r}")
        if not isinstance(scene["video_prompt"], str) or not scene["video_prompt"].strip():
            raise ValueError(f"scenes[{i}] (id={sid}): пустой video_prompt")

        # ref-маркеры в video_prompt
        tokens = ref_tokens_outside_dialogue(scene["video_prompt"])
        markers = [t for t in tokens if t in base_names]
        unknown = sorted({t for t in tokens if t not in base_names})
        if not markers:
            raise ValueError(
                f"scenes[{i}] (id={sid}): в video_prompt нет ни одного "
                f"[base_name] маркера — ref2video не получит референсов. "
                f"Каждая сцена обязана ссылаться минимум на одну "
                f"base-переменную (локацию!)."
            )
        if len(set(markers)) > 5:
            print(
                f"  [WARN] scene {sid}: {len(set(markers))} уникальных "
                f"[base_name] > hard cap 5 — лишние уйдут в текст plain-именами"
            )
        for t in unknown:
            print(
                f"  [WARN] scene {sid}: неизвестный маркер [{t}] "
                f"(нет в base_prompts) — подмены на <Picture N> не будет"
            )

        dur = float(scene["duration_s"])
        if not (MINIMAX_MIN_DURATION_S <= dur <= MINIMAX_MAX_DURATION_S):
            print(
                f"  [WARN] scene {sid}: duration_s={dur} вне "
                f"{MINIMAX_MIN_DURATION_S}..{MINIMAX_MAX_DURATION_S} — "
                f"будет откламплено при патче"
            )
        # --- multishot-маркеры [Shot N] (официальный формат MiniMax H3)
        shot_markers = parse_shot_markers(scene["video_prompt"])
        if dur >= 8.0 and len(shot_markers) < 2:
            print(
                f"  [WARN] scene {sid}: duration_s={dur} без multishot "
                f"[Shot N] маркеров — длинные сцены рекомендуется резать "
                f"на 2-4 шота (см. GROK_Prompt_MINIMAX.txt)"
            )
        if shot_markers:
            ns = [m["n"] for m in shot_markers]
            if ns != list(range(1, len(ns) + 1)):
                print(
                    f"  [WARN] scene {sid}: нумерация шотов {ns}, "
                    f"ожидается [Shot 1..N] подряд"
                )
            prev_t = 0.0
            for m in shot_markers[1:]:
                t = m.get("start_s")
                if t is None:
                    print(
                        f"  [WARN] scene {sid}: [Shot {m['n']}] без "
                        f"таймстампа At MM:SS.mmm"
                    )
                    continue
                if t <= prev_t:
                    print(
                        f"  [WARN] scene {sid}: таймстампы шотов не "
                        f"возрастают ([Shot {m['n']}] At {t:.3f}s)"
                    )
                if t >= dur:
                    print(
                        f"  [WARN] scene {sid}: [Shot {m['n']}] At "
                        f"{t:.3f}s выходит за duration_s={dur}"
                    )
                prev_t = t
        dlg = scene.get("dialogue")
        if not dlg:
            print(
                f"  [WARN] scene {sid}: пустой dialogue — у сцены не будет "
                f"субтитров (сама речь берётся из video_prompt)"
            )
        else:
            words = sum(len((d.get("line") or "").split()) for d in dlg)
            if words > dur * 3:
                print(
                    f"  [WARN] scene {sid}: {words} слов на {dur}s — реплики "
                    f"могут не влезть (норма ~2-2.5 слова/сек)"
                )
            n_shots = max(1, len(shot_markers))
            for j, d in enumerate(dlg):
                if len(shot_markers) > 1 and d.get("shot") is None:
                    print(
                        f"  [WARN] scene {sid}: dialogue[{j}] без поля "
                        f"'shot' — в multishot-сцене реплика уедет в шот 1"
                    )
                sh = d.get("shot")
                if sh is not None:
                    try:
                        sh_i = int(sh)
                        if not 1 <= sh_i <= n_shots:
                            print(
                                f"  [WARN] scene {sid}: dialogue[{j}] "
                                f"shot={sh} вне 1..{n_shots}"
                            )
                    except (TypeError, ValueError):
                        print(
                            f"  [WARN] scene {sid}: dialogue[{j}] "
                            f"shot={sh!r} не число"
                        )
                ref = (d.get("speaker_ref") or "").strip()
                if ref and ref not in base_names:
                    print(
                        f"  [WARN] scene {sid}: dialogue[{j}] speaker_ref "
                        f"{ref!r} нет в base_prompts"
                    )

            # один звучащий говорящий на шот — иначе липсинк уезжает
            # на единственное видимое лицо
            by_shot: dict[int, set[str]] = {}
            for d in dlg:
                try:
                    k = int(d.get("shot") or 1)
                except (TypeError, ValueError):
                    k = 1
                r = (d.get("speaker_ref") or "").strip()
                if r:
                    by_shot.setdefault(k, set()).add(r)
            for k, spk in sorted(by_shot.items()):
                if len(spk) > 1:
                    print(
                        f"  [WARN] scene {sid}: в шоте {k} говорят "
                        f"{len(spk)} персонажа ({', '.join(sorted(spk))}) — "
                        f"разнеси реплики по разным шотам"
                    )


# ============================================================ main


async def main(args):
    comfy_mgr = ComfyManager(comfy_root=COMFY_ROOT, host="127.0.0.1", port=8188)
    await comfy_mgr.start()

    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)
    validate_plan(plan)

    scenes = plan["scenes"]
    n_scenes = len(scenes)
    regen = bool(args.project_dir)
    if regen:
        todo_idx = parse_scene_numbers(args.scenes, n_scenes)
        print(f"РЕЖИМ ПЕРЕГЕНЕРАЦИИ: сцены {[i + 1 for i in todo_idx]}")
    else:
        todo_idx = list(range(n_scenes))
    base_prompts_by_name = {bp["base_name"]: bp for bp in plan["base_prompts"]}
    print(f"План получен! Сцен: {n_scenes}, entities: {len(base_prompts_by_name)}")
    voice_refs = load_voice_refs(plan)
    if voice_refs:
        print(f"  voice refs: {sorted(voice_refs)}")

    with open("img_krea2_turbo.json", "r", encoding="utf-8") as f:
        workflow_seed = json.load(f)
    with open("video_minimax_refs.json", "r", encoding="utf-8") as f:
        workflow_vid = json.load(f)

    comfy = ComfyUIClient(COMFYUI_URL)

    # -------------------------------------------------- ЭТАП 1: base refs
    # identity-рефы для персонажей/локаций/объектов (text-to-image,
    # 720p). Эти PNG напрямую скармливаются реф-слотам MiniMax H3 на
    # ЭТАПЕ 2 — без промежуточного стартового кадра.
    print("\n--- ЭТАП 1: base entity refs (Krea 2 seed) ---")
    if regen:
        # референсы уже есть с прошлого прогона — рисуем только недостающие
        refs_todo = [
            bp for bp in plan["base_prompts"]
            if not os.path.exists(os.path.join(
                REFS_DIR, entity_ref_filename(bp["base_name"], "")))
        ]
        if not refs_todo:
            print("  все референсы на месте, ЭТАП 1 пропущен")
    else:
        refs_todo = plan["base_prompts"]
    for i, bp in enumerate(refs_todo):
        base_name = bp["base_name"]
        entity_type = bp["entity_type"]
        seed_prompt = seed_prompt_for_entity(entity_type, bp["base_prompt"])

        if entity_type == "character":
            workflow_seed["52"]["inputs"]["width"] = 720
            workflow_seed["52"]["inputs"]["height"] = 1280
        else:
            workflow_seed["52"]["inputs"]["width"] = 1280
            workflow_seed["52"]["inputs"]["height"] = 720

        workflow_seed["53"]["inputs"]["seed"] = f"{secrets.randbelow(2**32)}"
        workflow_seed["51"]["inputs"]["text"] = seed_prompt
        workflow_seed["29"]["inputs"]["filename_prefix"] = (
            f"base_image_gen/{base_name}"
        )
        print(f"  ref [{i + 1}/{len(refs_todo)}] {base_name} "
              f"({entity_type})")
        prompt_id = await comfy.queue_prompt(workflow_seed)
        files = await comfy.wait_for_execution(prompt_id, workflow=workflow_seed)
        print(f"    готов: {files[0] if files else '?'}")

    if refs_todo:
        await comfy_mgr.restart()
    # выходная папка запуска
    if regen:
        unique_path = resolve_project_dir(args.project_dir)
    else:
        folder_to_create = os.path.join(
            OUTPUT_ROOT, datetime.datetime.today().strftime("%Y-%m-%d")
        )
        unique_path = get_unique_dir_name(folder_to_create)
        os.makedirs(unique_path)
    print(f"\nВыходная папка: {unique_path}")
    archive_dir = os.path.join(
        unique_path, "replaced",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    # -------------------------------------------------- ЭТАП 2: видео из рефов
    # MiniMaxH3ReferenceToVideo: [base_name] -> <Subject N>, рефы в слоты
    # ref_image_0..4 (лоадеры 148..152). Модель сама строит кадр из рефов
    # и video_prompt, звук вшит в клип.
    print("\n--- ЭТАП 2: видео напрямую из референсов (video_minimax_refs) ---")
    for gen_pos, i in enumerate(todo_idx):
        scene = scenes[i]
        final_prompt, entities, slot_by_name = build_minimax_ref_prompt(
            prompt_text=scene["video_prompt"],
            base_prompts_by_name=base_prompts_by_name,
            max_pictures=MAX_IMAGE_REFS,
        )
        image_paths = [
            os.path.join(REFS_DIR, entity_ref_filename(name, ""))
            for name in entities
        ]
        missing = [p for p in image_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"scene {scene.get('scene_id', i)}: нет ref-файлов: {missing}. "
                f"ЭТАП 1 должен был их создать."
            )

        # голосовые рефы: порядок = порядок реальных вокальных событий
        # в тексте сцены (гайд §5.4), слоты <Audio 1..3> и ID (S1..S3)
        # нумеруются одинаково
        scene_voices = [
            n for n in scene_speaker_order(scene["video_prompt"])
            if n in voice_refs
        ]
        if len(scene_voices) > MAX_AUDIO_REFS:
            print(
                f"  [WARN] scene {scene.get('scene_id', i + 1)}: говорящих "
                f"с voice ref {len(scene_voices)} > {MAX_AUDIO_REFS} — "
                f"лишние останутся без голосового рефа"
            )
            scene_voices = scene_voices[:MAX_AUDIO_REFS]
        audio_paths = [voice_refs[name] for name in scene_voices]

        # Полный full-reference rewrite: шесть секций, [base_name] уже
        # заменены на <Subject N>, (Sx) перенумерованы по порядку реплик.
        video_prompt = compose_minimax_scene_prompt(
            scene=scene,
            base_prompts_by_name=base_prompts_by_name,
            ordered=entities,
            slot_by_name=slot_by_name,
            body_prompt=final_prompt,
            voice_speakers=scene_voices,
            style_anchor=STYLE_ANCHOR,
        )
        wf_vid = patch_minimax_refs_video_workflow(
            workflow=workflow_vid,
            image_paths=image_paths,
            positive_prompt=video_prompt,
            duration_s=float(scene["duration_s"]),
            filename_prefix=os.path.join(unique_path, str(i + 1)),
            seed=secrets.randbelow(2**63),
            audio_paths=audio_paths,
        )
        print(
            f"  scene {scene.get('scene_id', i + 1)}: "
            f"duration={scene['duration_s']}s, "
            f"shots={max(1, len(parse_shot_markers(scene['video_prompt'])))}, "
            f"entities={entities}, voices={scene_voices}, "
            f"dialogue_lines={len(scene.get('dialogue') or [])}"
        )
        print(f"    prompt: {video_prompt}")
        old_clips = sorted(_glob.glob(os.path.join(unique_path, f"{i + 1}_*.mp4")))
        prompt_id = await comfy.queue_prompt(wf_vid)
        vid_files = await comfy.wait_for_execution(prompt_id, workflow=wf_vid)
        print(f"    видео готово ({vid_files[0] if vid_files else '?'})")

        # перегенерация: старые клипы сцены уходят в replaced/, но только
        # если новый клип реально появился
        if regen and old_clips:
            new_clip = find_video_for_scene(unique_path, i)
            if new_clip and new_clip not in old_clips:
                archive_files(old_clips, archive_dir)
            else:
                print(f"  [WARN] сцена {i + 1}: новый клип не найден, "
                      f"старый оставлен")

        # плановый перезапуск ComfyUI: выгружает модели и чистит ОЗУ
        if (RESTART_EVERY_N_SCENES
                and (gen_pos + 1) % RESTART_EVERY_N_SCENES == 0
                and gen_pos + 1 < len(todo_idx)):
            print(f"  ♻ перезапуск ComfyUI после {gen_pos + 1} клипов...")
            await comfy_mgr.restart()

    print("\nГенерация завершена, собираю финалку...")
    await comfy_mgr.stop()

    if regen:
        # прежняя финалка устарела — убираем в replaced/, пересоберём заново
        archive_files(
            [os.path.join(unique_path, f) for f in FINAL_ARTIFACTS]
            + sorted(_glob.glob(os.path.join(unique_path, "part_*"))),
            archive_dir,
        )

    # -------------------------------------------------- ЭТАП 3: финалка
    # список клипов — ДИНАМИЧЕСКИ по фактически созданным файлам.
    clip_paths = []
    kept_idx: list[int] = []
    for i in range(n_scenes):
        p = find_video_for_scene(unique_path, i)
        if p:
            clip_paths.append(p)
            kept_idx.append(i)
        else:
            print(f"  [WARN] сцена {i + 1}: клип не найден, пропускаю")
    if not clip_paths:
        raise RuntimeError("ни одного клипа не найдено — нечего склеивать")

    # фактические длительности клипов: по ним делим на части и кладём
    # субтитры на реальный таймлайн склейки, а не на плановые duration_s
    durations: list[float] = []
    for k, idx in enumerate(kept_idx):
        d = await media_duration(
            clip_paths[k], float(scenes[idx].get("duration_s") or 8.0)
        )
        scenes[idx]["duration_s"] = d
        durations.append(d)
    total_s = sum(durations)
    print(f"  таймлайн: {len(kept_idx)} сцен, {total_s:.1f} с")

    parts = split_into_parts(durations, FINAL_MAX_PART_S)
    n_parts = len(parts)
    if n_parts > 1:
        print(f"  длина {total_s:.1f} с > {FINAL_MAX_PART_S:.0f} с — "
              f"делю на {n_parts} части по границам сцен:")
        for pn, part in enumerate(parts, 1):
            sids = [kept_idx[k] + 1 for k in part]
            print(f"    часть {pn}: сцены {sids[0]}..{sids[-1]}, "
                  f"{sum(durations[k] for k in part):.1f} с")

    cover_text = (plan.get("cover_text_short") or plan.get("cover_text") or "").strip()
    cover_sid = plan.get("cover_scene_id")
    cover_idx = next(
        (k for k, sc in enumerate(scenes) if sc.get("scene_id") == cover_sid),
        None,
    ) if cover_sid is not None else None
    if cover_sid is not None and cover_idx is None:
        print(f"  [WARN] cover_scene_id={cover_sid} нет в scenes — "
              f"беру первый клип части")
    try:
        plan_cover_t = max(0.0, float(plan.get("cover_time_s") or 0.0))
    except (TypeError, ValueError):
        plan_cover_t = 0.0
    base_label = (plan.get("part_label") or "").strip() or None

    finals: list[str] = []
    for pn, part in enumerate(parts, 1):
        part_clips = [clip_paths[k] for k in part]
        part_idx = [kept_idx[k] for k in part]
        part_scenes = [scenes[i] for i in part_idx]

        # обложка: cover_scene_id, если сцена попала в эту часть;
        # иначе — первый клип части, кадр на 1-й секунде
        if cover_idx is not None and cover_idx in part_idx:
            cover_clip = part_clips[part_idx.index(cover_idx)]
            cover_t = plan_cover_t
        else:
            cover_clip = part_clips[0]
            cover_t = plan_cover_t if (n_parts == 1 and cover_idx is None) else 1.0

        if n_parts == 1:
            out_dir = unique_path
            label = base_label
        else:
            out_dir = os.path.join(unique_path, f"part_{pn}")
            label = f"Часть {pn}"
            print(f"\n  === часть {pn}/{n_parts} -> {out_dir}")

        finals.append(await build_final_video(
            out_dir, part_clips, part_scenes,
            cover_text=cover_text,
            cover_clip=cover_clip,
            cover_t=cover_t,
            part_label=label,
        ))

    print("\nФинальное видео:" if len(finals) == 1 else "\nФинальные видео:")
    for p in finals:
        print(f"  {p}")

    print("Пайплайн MiniMax H3 ref2video успешно завершён!")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
