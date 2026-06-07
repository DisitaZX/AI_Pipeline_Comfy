import asyncio
import base64
import copy
import datetime
import hashlib
import json
import os
import pprint
import re
import secrets
import shutil
import subprocess
import sys
import time
import uuid
import random

import aiohttp
import torch
import websockets

sys.path.append("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from comfy_manager import ComfyManager
from comics.animator import ImagePanner
from llama_manager import llama_mgr
from ltx_chunking import (
    LTX_CHUNK_FPS,
    compute_ltx_chunk_plan_for_scene,
    patch_ltx_multikf_workflow,
)
from prompt_builder import (
    ENTITY_TYPES,
    build_qwen_edit_prompt,
    entity_klein_variants,
    entity_ref_filename,
    entity_seed_variant,
    entity_variants,
    qwen_variant_instruction,
    seed_prompt_for_entity,
)
from srt.ass_encode import fragments_to_ass
from srt_timing.srt_timing import SRTSceneTimeline
from whisper_align import align_to_aeneas_json

LLAMA_URL = "http://localhost:8080/v1/chat/completions"
COMFYUI_URL = "127.0.0.1:8188"


def patch_flux2_edit_workflow(
    workflow: dict,
    image_paths: list[str],  # 1..5 absolute paths, ordered like [image 1..N]
    positive_prompt: str,
    filename_prefix: str,
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
) -> dict:
    """
    Патчит flux2-klein-edit.json под текущую сцену.

    FLUX.2 Klein использует chained ReferenceLatent: positive- и negative-
    conditioning поочерёдно "обвешиваются" reference-латентами для каждой
    image. Точка съёма для CFGGuider (нода 178) зависит от количества
    reference'ов (см. таблицу _POSITIVE_ENDPOINTS_BY_N ниже). Неиспользуемые
    звенья цепочки становятся unreachable от output-ноды 94 и ComfyUI их
    не выполнит.
    """
    wf = copy.deepcopy(workflow)
    n = len(image_paths)
    if not 1 <= n <= 5:
        raise ValueError(f"image_paths должен иметь 1..5 элементов, got {n}")

    # VHS_LoadImagePath ноды в порядке image1..image5
    loader_nodes = ["160", "161", "162", "163", "175"]
    for i, path in enumerate(image_paths):
        wf[loader_nodes[i]]["inputs"]["image"] = path

    # CFGGuider 178 positive цепляется к нужному звену ReferenceLatent цепочки
    positive_endpoint_by_n = {
        1: "148",
        2: "151",
        3: "155",
        4: "158",
        5: "169",
    }

    wf["178"]["inputs"]["positive"] = [positive_endpoint_by_n[n], 0]

    # промпты (CLIPTextEncode)
    wf["142"]["inputs"]["text"] = "Japanese anime style." + positive_prompt
    # выход
    wf["254"]["inputs"]["filename_prefix"] = filename_prefix

    # размеры — в EmptyFlux2LatentImage (138) и в Flux2Scheduler (145)
    wf["138"]["inputs"]["width"] = width
    wf["138"]["inputs"]["height"] = height
    wf["145"]["inputs"]["width"] = width
    wf["145"]["inputs"]["height"] = height

    # seed (RandomNoise)
    if seed is not None:
        wf["134"]["inputs"]["noise_seed"] = seed

    return wf

def patch_hidream_workflow(
    workflow: dict,
    image_paths: list[str],  # абсолютный путь до single input image
    positive_prompt: str,
    filename_prefix: str,
    seed: int | None = None,
    ) -> dict:

    wf = copy.deepcopy(workflow)

    loader_nodes = ["234", "235", "233", "236", "237"]
    for i, path in enumerate(image_paths):
        wf[loader_nodes[i]]["inputs"]["image"] = path

    # Отключаем image2 / image3 в обоих text-encode'ах. Loader-ноды
    # 200/201 без рёбер становятся unreachable от SaveImage.
    if len(image_paths) == 1:
        node_inputs = wf["104"]["inputs"]
        for slot in ("images.image_2", "images.image_3", "images.image_4", "images.image_5"):
            node_inputs.pop(slot, None)
    elif len(image_paths) == 2:
        node_inputs = wf["104"]["inputs"]
        for slot in ("images.image_3", "images.image_4", "images.image_5"):
            node_inputs.pop(slot, None)
    elif len(image_paths) == 3:
        node_inputs = wf["104"]["inputs"]
        for slot in ("images.image_4", "images.image_5"):
            node_inputs.pop(slot, None)
    elif len(image_paths) == 4:
        node_inputs = wf["104"]["inputs"]
        for slot in ("images.image_5",):
            node_inputs.pop(slot, None)


    # Positive prompt — в нodу 177 (negative 175 оставляем хардкод).
    wf["171"]["inputs"]["value"] = "Japanese anime style. " + positive_prompt

    # SaveImage prefix
    wf["227"]["inputs"]["filename_prefix"] = filename_prefix

    # Seed
    if seed is not None:
        wf["108"]["inputs"]["noise_seed"] = seed

    return wf

def patch_qwen_edit_workflow(
    workflow: dict,
    image_paths: list[str],  # абсолютный путь до single input image
    positive_prompt: str,
    filename_prefix: str,
    seed: int | None = None,
    width: int = 1920,
    height: int = 1088,
) -> dict:
    """
    Qwen Image Edit через TextEncodeQwenImageEditPlus умеет 1-3 input
    image'а.

    Loader-ноды 200/201 (бывшие image2/image3) после удаления
    рёбер становятся unreachable от output-ноды 9 и ComfyUI их
    не выполняет — оставляем в JSON как есть, чтобы не ломать
    общий шейп workflow'а.

    Параметры:
      workflow         - dict из qwen_image.json (deepcopy'ится).
      image_path       - абсолютный путь до ref-image (например
                         C:\\Users\\...\\base_image_gen\\yard_front_00001_.png
                         или предыдущий chunk-keyframe для c2/c3).
      positive_prompt  - текст инструкции (для variant'ов —
                         qwen_variant_instruction, для c2/c3 —
                         build_chunk_edit_prompt output).
      filename_prefix  - префикс для SaveImage (например
                         "base_image_gen/yard_right" или
                         "image_gen/2_c2").
      seed             - seed для KSampler (None → не трогаем).
      width / height   - размеры EmptyLatentImage (по умолчанию
                         1280x720 как в основном image-gen pipeline'е).

    Возвращает патченный workflow (исходный не модифицируется).

    Ноды qwen_image.json:
      9   SaveImage                       - filename_prefix
      173 FluxKontextMultiRef (negative)  - не трогаем
      174 FluxKontextMultiRef (positive)  - не трогаем
      175 TextEncodeQwenImageEditPlus     - negative prompt (хардкод),
                                            image2/image3 удаляем
      177 TextEncodeQwenImageEditPlus     - positive prompt,
                                            image2/image3 удаляем
      191 KSampler                        - seed, steps, cfg
      200 VHS_LoadImagePath (был image2)  - становится unreachable
      201 VHS_LoadImagePath (был image3)  - становится unreachable
      202 VHS_LoadImagePath (image1)      - получает image_path
      203 EmptyLatentImage                - width, height
    """
    wf = copy.deepcopy(workflow)

    loader_nodes = ["200", "201", "202"]
    for i, path in enumerate(image_paths):
        wf[loader_nodes[i]]["inputs"]["image"] = path

    # Отключаем image2 / image3 в обоих text-encode'ах. Loader-ноды
    # 200/201 без рёбер становятся unreachable от SaveImage.
    if len(image_paths) == 1:
        for nid in ("175", "177"):
            node_inputs = wf[nid]["inputs"]
            for slot in ("image2", "image3"):
                node_inputs.pop(slot, None)
    elif len(image_paths) == 2:
        for nid in ("175", "177"):
            node_inputs = wf[nid]["inputs"]
            node_inputs.pop("image3", None)


    dynamic_strength = 1.0 + random.uniform(-0.00001, 0.00001)
    dynamic_strength_2 = 0.9 + random.uniform(-0.00001, 0.00001)
    wf["179"]["inputs"]["strength_model"] = dynamic_strength
    wf["207"]["inputs"]["strength_model"] = dynamic_strength_2
    wf["210"]["inputs"]["strength_model"] = dynamic_strength_2

    # Positive prompt — в нodу 177 (negative 175 оставляем хардкод).
    wf["177"]["inputs"]["prompt"] = positive_prompt

    # SaveImage prefix
    wf["9"]["inputs"]["filename_prefix"] = filename_prefix

    # EmptyLatentImage размеры
    wf["203"]["inputs"]["width"] = width
    wf["203"]["inputs"]["height"] = height

    # Seed
    if seed is not None:
        wf["191"]["inputs"]["seed"] = seed

    return wf


def set_chunk_count(workflow: dict, n_chunks: int) -> dict:
    """
    Подгоняет VID_WAN workflow под нужное число чанков (1, 2 или 3).
    Возвращает НОВЫЙ dict (исходный не трогается).

      1: оставляем только subgraph 623, Video Combine читает прямо из 623:600
      2: оставляем 623+336, Video Combine читает из 336:329
      3: оставляем 623+336+644, Video Combine читает из 644:683 (как сейчас)
    """
    assert n_chunks in (1, 2, 3), f"n_chunks must be 1/2/3, got {n_chunks}"
    wf = json.loads(json.dumps(workflow))  # deep copy

    # источник кадров для финального VHS_VideoCombine (ID 656):
    final_source = {
        1: ["623:600", 0],
        2: ["336:329", 2],
        3: ["644:683", 2],
    }[n_chunks]
    wf["656"]["inputs"]["images"] = final_source

    # вычистить ненужные subgraph-узлы
    drop_prefixes = []
    if n_chunks < 3:
        drop_prefixes.append("644:")
    if n_chunks < 2:
        drop_prefixes.append("336:")

    for nid in list(wf.keys()):
        if any(nid.startswith(p) for p in drop_prefixes):
            del wf[nid]

    return wf


def get_audio_duration(audio_file):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        audio_file,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return float(result.stdout.strip())


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def extract_last_frame(video_path: str, output_png: str) -> bool:
    """Вытащить последний (стабильный) кадр LTX-видео в PNG.

    Берём кадр за ~0.1с до конца — последний обычно с motion blur и менее
    стабилен для подачи на Flux2 как Picture 1. Возвращает True при успехе.
    """
    if not os.path.exists(video_path):
        print(f"[extract_last_frame] видео не найдено: {video_path}")
        return False
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.1",
        "-i", video_path,
        "-frames:v", "1",
        "-update", "1",
        output_png,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(output_png):
        # Фоллбек: некоторые контейнеры не любят -sseof. Попробуем
        # просто без сика и заполним последний доступный кадр через -frames:v 1
        # на конце через -vframes 1 + -ss=duration-0.1 (требует ffprobe).
        try:
            dur = get_audio_duration(video_path)  # ffprobe duration работает и для видео
            cmd2 = [
                "ffmpeg", "-y",
                "-ss", f"{max(0.0, dur - 0.1):.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-update", "1",
                output_png,
            ]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd2,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc2.communicate()
            if proc2.returncode == 0 and os.path.exists(output_png):
                return True
        except Exception as e:
            print(f"[extract_last_frame] fallback failed: {e}")
        print(f"[extract_last_frame] не удалось вытащить кадр из {video_path}: "
              f"{stderr.decode(errors='ignore')[:200]}")
        return False
    return True


async def get_video_frame_count(video_path: str) -> int | None:
    """Вернуть число видеокадров в файле через ffprobe (или None при ошибке)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return int(stdout.decode().strip())
    except (ValueError, AttributeError):
        return None


async def trim_video_to_frames(
    video_path: str,
    target_frames: int,
    fps: int = LTX_CHUNK_FPS,
) -> bool:
    """Обрезать видео сцены ровно до target_frames кадров (число от TTS).

    LTX рендерит количество кадров, округлённое ВВЕРХ до LTX-valid 8k+1
    (см. compute_scene_total_frames, mode="up"), поэтому клип почти всегда
    на 1..8 кадров ДЛИННЕЕ слота, который TTS-таймлайн отвёл сцене. Если
    оставить как есть — видео уплывает относительно аудио и рассинхрон
    накапливается от сцены к сцене.

    Здесь срезаем хвост до точного target_frames. Перекодируем в
    libx264 / yuv420p @ fps, чтобы ВСЕ клипы имели единый кодек: финальная
    склейка идёт через `ffmpeg concat -c copy` и требует одинаковых
    параметров потока у всех входов.

    Возвращает True при успехе.
    """
    if not os.path.exists(video_path):
        print(f"[trim_video] видео не найдено: {video_path}")
        return False
    target_frames = max(1, int(target_frames))

    actual = await get_video_frame_count(video_path)
    if actual is not None and actual < target_frames:
        # LTX выдал меньше кадров чем слот TTS (теоретически не должно
        # случаться при mode="up", но страхуемся). ffmpeg -frames:v просто
        # отдаст все доступные кадры; недостающий хвост доберётся заморозкой
        # последнего кадра на финальном tpad-проходе. Перекодируем всё равно,
        # чтобы кодек совпадал с обрезанными клипами (concat -c copy).
        print(
            f"[trim_video] {os.path.basename(video_path)}: actual={actual} < "
            f"target={target_frames}; обрезать нечего, только нормализую кодек"
        )

    tmp_path = video_path + ".trim.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-frames:v", str(target_frames),
        "-r", str(int(fps)),
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        tmp_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0 or not os.path.exists(tmp_path):
        print(
            f"[trim_video] не удалось обрезать {os.path.basename(video_path)}: "
            f"{stderr.decode(errors='ignore')[:200]}"
        )
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False

    os.replace(tmp_path, video_path)
    print(
        f"[trim_video] {os.path.basename(video_path)}: "
        f"{actual if actual is not None else '?'} -> {target_frames} кадров"
    )
    return True


def find_video_for_scene(unique_path: str, scene_idx: int) -> str | None:
    """Найти .mp4 файл, сохранённый VHS_VideoCombine для сцены scene_idx (0-based).

    filename_prefix задаётся как f'{unique_path}\\\\{i+1}', а Comfy/VHS добавляет
    суффикс типа '_00001.mp4'. Берём самый свежий matching файл.
    """
    import glob as _glob
    pattern = os.path.join(unique_path, f"{scene_idx + 1}_*.mp4")
    matches = sorted(_glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


# Каталог и файл для дампа LTX-планов (Phase B). Всегда один и тот же файл —
# перезаписывается на каждом запуске. Нужен чтобы а) после ошибки/плохой
# генерации посмотреть, что именно Gemma выдала, и б) Phase C читала планы
# отсюда (можно вручную поправить JSON и перезапустить только Phase C).
VIDEO_PROMPTS_DIR = os.path.join(BASE_DIR, "video_prompts")
LTX_PROMPTS_FILE = os.path.join(VIDEO_PROMPTS_DIR, "ltx_prompts.json")


def save_ltx_prompts(
    chunk_plans: list[dict | None],
    scenes: list[dict],
    path: str = LTX_PROMPTS_FILE,
) -> None:
    """Сохраняет планы Phase B (LTX prompts) в JSON-файл.

    Формат — человекочитаемый: верхнеуровневый объект с метаданными и
    списком сцен. Каждая сцена несёт scene_index / scene_id (для навигации)
    + сам план (num_chunks, global_prompt, total_frames, chunks). Файл
    всегда перезаписывается (один и тот же путь).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scenes_out: list[dict] = []
    for i, plan_i in enumerate(chunk_plans):
        scene = scenes[i] if i < len(scenes) else {}
        entry: dict = {
            "scene_index": i,
            "scene_id": scene.get("scene_id", i),
        }
        if plan_i is None:
            entry["plan"] = None
        else:
            entry["plan"] = {
                "num_chunks": plan_i.get("num_chunks"),
                "global_prompt": plan_i.get("global_prompt", ""),
                "total_frames": plan_i.get("total_frames"),
                "chunks": [
                    {
                        "video_prompt": c.get("video_prompt", ""),
                        "length_frames": c.get("length_frames"),
                    }
                    for c in plan_i.get("chunks", [])
                ],
            }
        scenes_out.append(entry)

    payload = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_scenes": len(chunk_plans),
        "fps": LTX_CHUNK_FPS,
        "scenes": scenes_out,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ltx_prompts] планы Phase B сохранены в {path}")


def load_ltx_prompts(
    n_scenes: int,
    path: str = LTX_PROMPTS_FILE,
) -> list[dict | None]:
    """Читает планы LTX из JSON-файла (обратная операция к save_ltx_prompts).

    Возвращает список длины n_scenes: chunk_plans[i] — план сцены i в том же
    виде, что выдаёт compute_ltx_chunk_plan_for_scene (num_chunks,
    global_prompt, chunks[{video_prompt, length_frames}], total_frames),
    либо None если для сцены плана нет.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    by_index: dict[int, dict] = {}
    for entry in payload.get("scenes", []):
        idx = entry.get("scene_index")
        plan = entry.get("plan")
        if idx is None or plan is None:
            continue
        by_index[int(idx)] = {
            "num_chunks": plan.get("num_chunks"),
            "global_prompt": plan.get("global_prompt", ""),
            "total_frames": plan.get("total_frames"),
            "chunks": [
                {
                    "video_prompt": c.get("video_prompt", ""),
                    "length_frames": c.get("length_frames"),
                }
                for c in plan.get("chunks", [])
            ],
        }

    return [by_index.get(i) for i in range(n_scenes)]


def split_by_sentence(text):
    # Находим индекс фактической середины
    mid_index = len(text) // 2

    # Ищем ближайшую точку после середины
    end_of_sentence = text.find(".", mid_index)

    # Если точка после середины не найдена, ищем до середины
    if end_of_sentence == -1:
        end_of_sentence = text.rfind(".", 0, mid_index)

    # Если точек нет вообще, делим просто пополам
    if end_of_sentence == -1:
        return text[:mid_index], text[mid_index:]

    # Сдвигаем индекс на +1, чтобы точка осталась в первой части
    split_point = end_of_sentence + 1

    part1 = text[:split_point].strip()
    part2 = text[split_point:].strip()

    return part1, part2


def get_unique_dir_name(base_name):
    # Если папки нет, возвращаем исходное имя
    if not os.path.exists(base_name):
        return base_name

    counter = 2
    while True:
        new_name = f"{base_name}_{counter}"
        if not os.path.exists(new_name):
            print(new_name)
            return new_name
        counter += 1


class ComfyUIClient:
    """Класс для асинхронного взаимодействия с сервером ComfyUI."""

    def __init__(self, server_address):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    async def queue_prompt(self, prompt_workflow):
        """Отправляет workflow в очередь и возвращает ID задачи.

        Если ComfyUI отвергает workflow на этапе валидации (HTTP 400 / нет
        'prompt_id' в ответе), сохраняет полный ответ сервера + сам workflow
        в errors/{ts}_queue_validation_* и бросает RuntimeError с краткой
        сводкой по node_errors. Иначе пайплайн падал бы с бесполезным
        KeyError: 'prompt_id' и реальная причина (отсутствующая модель,
        невалидный путь файла, missing node и т.д.) терялась.
        """
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

                # --- ошибка валидации / нестандартный ответ ---
                self._save_queue_error_log(
                    status=status,
                    raw_data=data,
                    raw_text=raw_text,
                    workflow=prompt_workflow,
                )

                err = (data or {}).get("error") or {}
                node_errors = (data or {}).get("node_errors") or {}
                err_type = err.get("type", "unknown")
                err_msg = err.get("message", "<no message>")
                err_details = err.get("details", "")

                node_summary_lines: list[str] = []
                for nid, ninfo in list(node_errors.items())[:10]:
                    ninfo = ninfo or {}
                    cls = ninfo.get("class_type", "?")
                    for e in (ninfo.get("errors") or [])[:5]:
                        e = e or {}
                        emsg = e.get("message", "?")
                        edet = e.get("details", "")
                        node_summary_lines.append(
                            f"    node {nid} ({cls}): {emsg}"
                            + (f" — {edet}" if edet else "")
                        )
                node_summary = (
                    "\n".join(node_summary_lines)
                    if node_summary_lines
                    else "    <none>"
                )

                tail = (
                    raw_text[:500]
                    if raw_text is not None
                    else json.dumps(data, ensure_ascii=False)[:500]
                )

                raise RuntimeError(
                    f"ComfyUI /prompt отклонил workflow "
                    f"(HTTP {status}): {err_type}: {err_msg}\n"
                    f"  details: {err_details}\n"
                    f"  node_errors:\n{node_summary}\n"
                    f"  response_preview: {tail}"
                )

    def _save_queue_error_log(
        self, *, status, raw_data=None, raw_text=None, workflow=None
    ):
        """Сохраняет ошибку валидации /prompt в errors/{ts}_queue_validation/."""
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_queue_validation")
            # на случай если в одну секунду упало две задачи
            counter = 2
            base_sub = sub
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
                text_lines.append("=== response.json ===")
                text_lines.append(
                    json.dumps(
                        raw_data, ensure_ascii=False, indent=2, default=str
                    )
                )
                text_lines.append("")
            if raw_text is not None:
                text_lines.append("=== response.text ===")
                text_lines.append(raw_text)
                text_lines.append("")

            with open(
                os.path.join(sub, "queue_error.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(text_lines))

            if workflow is not None:
                with open(
                    os.path.join(sub, "workflow.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)

            print(f"  [ComfyUI ERROR] queue-лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить queue-лог: {e}")

    async def wait_for_execution(self, prompt_id, workflow=None):
        """Слушает вебсокет и ждет завершения генерации, возвращая имена файлов.

        При получении события execution_error - сохраняет полный traceback +
        переданный workflow JSON в errors/{ts}_{prompt_id}/ и продолжает ждать
        executing:None, чтобы корректно очистить очередь Comfy.
        """
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
                        print(f"  [ComfyUI WARN] execution interrupted")
                        break

                if mtype == "executing":
                    data = message.get("data") or {}
                    if data.get("node") is None and data.get("prompt_id") == prompt_id:
                        break  # Генерация завершена

            # После завершения запрашиваем историю, чтобы получить имена сохраненных файлов
            return await self.get_history(prompt_id, workflow=workflow)

    def _save_error_log(self, prompt_id, error_data, workflow=None):
        """Записывает traceback ComfyUI execution_error + workflow в
        errors/{timestamp}_{prompt_id}/.
        """
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_{prompt_id[:12]}")
            os.makedirs(sub, exist_ok=True)

            tb = error_data.get("traceback") or []
            tb_text = (
                "\n".join(str(line) for line in tb)
                if isinstance(tb, list)
                else str(tb)
            )

            text_lines = [
                "=== ComfyUI execution_error ===",
                f"timestamp:         {ts}",
                f"prompt_id:         {prompt_id}",
                f"node_id:           {error_data.get('node_id')}",
                f"node_type:         {error_data.get('node_type')}",
                f"exception_type:    {error_data.get('exception_type')}",
                f"exception_message: {error_data.get('exception_message')}",
                f"executed_so_far:   {error_data.get('executed')}",
                "",
                "=== traceback ===",
                tb_text,
                "",
                "=== current_inputs ===",
                json.dumps(
                    error_data.get("current_inputs"),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                "",
                "=== current_outputs ===",
                json.dumps(
                    error_data.get("current_outputs"),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                "",
            ]
            with open(
                os.path.join(sub, "error.txt"), "w", encoding="utf-8"
            ) as f:
                f.write("\n".join(text_lines))

            if workflow is not None:
                with open(
                    os.path.join(sub, "workflow.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)

            print(f"  [ComfyUI ERROR] лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить error-лог: {e}")

    async def get_history(self, prompt_id, workflow=None):
        """Получает результаты выполнения по ID задачи."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://{self.server_address}/history/{prompt_id}"
            ) as response:
                history = await response.json()

                if prompt_id not in history:
                    print(
                        f"  [ComfyUI WARN] prompt_id {prompt_id} отсутствует "
                        f"в history (нода вероятно упала на validation)"
                    )
                    return []

                entry = history[prompt_id]

                # Проверяем status на ошибки (могло быть пропущено через WS)
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    self._save_history_error(prompt_id, entry, workflow)

                output_files = []
                # Ищем ноды, которые сохранили картинки/видео/аудио
                outputs = entry.get("outputs") or {}
                for node_id in outputs:
                    node_output = outputs[node_id]
                    if "images" in node_output:
                        for img in node_output["images"]:
                            output_files.append(img["filename"])
                    elif "gifs" in node_output:  # Для видео формата (VHS)
                        for vid in node_output["gifs"]:
                            output_files.append(vid["filename"])
                    elif "videos" in node_output:  # Native SaveVideo
                        for vid in node_output["videos"]:
                            output_files.append(vid["filename"])
                    elif "audio" in node_output:  # Для аудио
                        for aud in node_output["audio"]:
                            output_files.append(aud["filename"])

                return output_files

    def _save_history_error(self, prompt_id, history_entry, workflow=None):
        """Сохраняет ошибку из history.status.messages (если WS-событие пропущено)."""
        try:
            err_dir = os.path.join(BASE_DIR, "errors")
            os.makedirs(err_dir, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            sub = os.path.join(err_dir, f"{ts}_{prompt_id[:12]}_history")
            os.makedirs(sub, exist_ok=True)
            with open(
                os.path.join(sub, "history.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(history_entry, f, ensure_ascii=False, indent=2, default=str)
            if workflow is not None:
                with open(
                    os.path.join(sub, "workflow.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(workflow, f, ensure_ascii=False, indent=2)
            print(f"  [ComfyUI ERROR] history-лог сохранён в {sub}")
        except Exception as e:
            print(f"  [ComfyUI ERROR] не удалось сохранить history-лог: {e}")


async def main():
    # print("1. Обращаемся к Qwen 3.5 за сценарием...")
    # plan = await get_plan_from_ollama()
    COMFY_ROOT = r"C:\Users\Loopy\Desktop\comfyui"
    comfy_mgr = ComfyManager(
        comfy_root=COMFY_ROOT,
        host="127.0.0.1",
        port=8188,
        # extra_args=["--lowvram"],  # если нужно
    )
    await comfy_mgr.start()
    with open("main.json", "r", encoding="utf-8") as f:
        plan = json.load(f)
    print(f"План получен! Сцен для генерации: {len(plan['scenes'])}")

    aeneas_text = ""
    aeneas_words = ""
    for k in plan["scenes"]:
        aeneas_text += k["text_scene"] + "\n"
        for w in k["text_scene"].split():
            aeneas_words += w + "\n"
    print(aeneas_text)
    with open("aeneas_text.txt", "w", encoding="utf-8") as f:
        f.write(aeneas_text)
    with open("aeneas_words.txt", "w", encoding="utf-8") as f:
        f.write(aeneas_words)

    comfy = ComfyUIClient(COMFYUI_URL)

    # Загружаем шаблоны (предварительно сохраненные через "Save (API Format)" в ComfyUI)
    with open("TTS_Cozy.json", "r", encoding="utf-8") as f:
        workflow_tts = json.load(f)
    with open("Z-Anime.json", "r", encoding="utf-8") as f:
        workflow_img = json.load(f)
    with open("flux2-klein-edit.json", "r", encoding="utf-8") as f:
        workflow_flux2 = json.load(f)
    with open("qwen_image.json", "r", encoding="utf-8") as f:
        workflow_qwen_edit = json.load(f)
    with open("image_hidream_o1_dev.json", "r", encoding="utf-8") as f:
        workflow_image_hidream = json.load(f)
    with open("VID_LTX.json", "r", encoding="utf-8") as f:
        workflow_vid = json.load(f)

    # --- ЭТАП 2: Аудио ---
    print("\n2. Запускаем генерацию TTS ...")
    tts_node_id = "1"
    workflow_tts[tts_node_id]["inputs"]["text"] = plan["tts_text"]

    prompt_id_tts = await comfy.queue_prompt(workflow_tts)
    audio_files = await comfy.wait_for_execution(prompt_id_tts, workflow=workflow_tts)
    print(f"Аудио сгенерировано: {audio_files[0] if audio_files else 'Ошибка'}")

    # ! WHISPER ALIGN (sentence-level) -> map.json !
    print(f"Запуск Whisper для синхронизации...")
    await asyncio.to_thread(
        align_to_aeneas_json,
        "output_00001_.mp3",
        "aeneas_text.txt",
        "map.json",
        language="ru",
        level="fragment",
    )

    # ! WHISPER ALIGN (word-level) -> map_words.json !
    print(f"Запуск Whisper word-level...")
    await asyncio.to_thread(
        align_to_aeneas_json,
        "output_00001_.mp3",
        "aeneas_words.txt",
        "map_words.json",
        language="ru",
        level="word",
    )

    # ! SRT_ENCODE !

    command = [
        r"C:\Users\Loopy\Desktop\comfyui\.venv\Scripts\python.exe",
        "srt/srt_encode.py ",
    ]

    print(f"Запуск SRT_ENCODE")

    process = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # ! SRTSceneTimeline !

    with open("srt/srt_encode.srt", "r", encoding="utf-8") as f:
        text_srt = f.read()
    with open("aeneas_text.txt", "r", encoding="utf-8") as f:
        text_target = f.read()

    SceneTimeline = SRTSceneTimeline()

    SRT_json = SceneTimeline.build(text_srt, text_target, fps=50)
    pprint.pprint(SRT_json)

    # ! ASS KARAOKE !

    fragments_to_ass("map_words.json", "subs.ass")

    # ============================================================
    # генерация base entity refs (multi-variant)
    # ============================================================
    # Для каждой переменной из plan["base_prompts"] generate'им набор рефов
    # под её entity_type (поле обязательно — strict-fail если отсутствует):
    #
    #   character: 1 реф
    #     - {name}_fullbody_00001_.png — text-to-image (seed), полный рост
    #       eye-level, plain background. Никаких edit-вариантов: face-портрет
    #       был удалён, во всех сценах character'а Klein получает только
    #       этот fullbody-реф. Это даёт стабильную identity-preservation
    #       (один и тот же latent на всех сценах), а close-up портрет
    #       Klein умеет композить и сам по image_prompt'у сцены.
    #
    #   location: 4 рефа eye-level (камера ~1.6m, level horizon)
    #     - {name}_front_00001_.png — text-to-image (seed), wide
    #       establishing shot спереди.
    #     - {name}_right_00001_.png / _left_00001_ / _back_00001_ —
    #       Qwen Image-Edit от seed'а с инструкцией повернуть камеру
    #       вокруг центра локации на 90°/-90°/180°.
    #
    #   object: 1 реф
    #     - {name}_00001_.png — text-to-image, base_prompt as-is.
    #       (как раньше; variant-суффикса нет.)
    #
    # Per-chain variant pick (см. Phase A ниже): для каждой entity выбираем
    # один variant из entity_variants(entity_type) детерминированным md5-хешем
    # от (chain_anchor_scene_id, base_name). Это даёт:
    #   - визуальную консистентность внутри chain'а (все сцены одной цепочки
    #     soft/hard continuity видят ОДИН ракурс location'а и ОДНУ позу
    #     character'а — "одна сцена" не дёргается между front/back);
    #   - визуальное разнообразие между chain'ами (соседние new_cut'ы
    #     с большой вероятностью получают разные variant-индексы);
    #   - повторяемость между запусками (md5 стабилен, в отличие от hash()).
    for i, scene in enumerate(plan["base_prompts"]):
        base_name = scene["base_name"]
        base_prompt = scene["base_prompt"]
        entity_type = scene.get("entity_type")
        if entity_type not in ENTITY_TYPES:
            raise ValueError(
                f"plan['base_prompts'][{i}].entity_type is {entity_type!r}; "
                f"expected one of {ENTITY_TYPES}. Re-run GROK plan stage "
                f"with the updated schema (each base_prompt must have "
                f"entity_type)."
            )

        # 1. Seed (Z-Anime text-to-image) с entity-specific framing.
        seed_prompt = seed_prompt_for_entity(entity_type, base_prompt)

        seed_filename_prefix = f"base_image_gen/{base_name}"
        print(f"\n--- base ref [{i + 1}] {base_name} (type=object) ---")
        print(f"  seed_prompt: {seed_prompt}")

        if entity_type == "character":
            workflow_img["6"]["inputs"]["width"] = 720
            workflow_img["6"]["inputs"]["height"] = 1280
        else:
            workflow_img["6"]["inputs"]["width"] = 1280
            workflow_img["6"]["inputs"]["height"] = 720

        workflow_img["17"]["inputs"]["seed"] = f"{secrets.randbelow(2**32)}"
        workflow_img["20"]["inputs"]["text"] = seed_prompt
        workflow_img["93"]["inputs"]["filename_prefix"] = seed_filename_prefix

        prompt_id_img = await comfy.queue_prompt(workflow_img)
        img_files = await comfy.wait_for_execution(
            prompt_id_img, workflow=workflow_img
        )
        seed_filename = img_files[0] if img_files else None
        print(f"  seed ready: {seed_filename}")


    # ---- ЭТАП 1.5 убран ----
    # Раньше тут был upfront text-only прогон Gemma по всем сценам
    # (precompute_all_ltx_chunk_plans). Затем в depth-loop'е Phase B делал
    # ВТОРОЙ проход Gemma уже с vision'ом, переписывая то, что Phase A
    # выдала text-only. Это была двойная трата компьюта + источник
    # рассогласования "plan vs keyframe".
    #
    # Новая схема: Gemma пишет план ОДИН РАЗ, ПОСЛЕ того как Klein
    # отрендерил keyframe сцены. См. depth-loop Phase B ниже — там
    # вызывается compute_ltx_chunk_plan_for_scene с keyframe'ом.
    # chunk_plans заполняется по мере прохождения depth'ов.
    chunk_plans: list[dict | None] = [None] * len(plan["scenes"])

    REFS_DIR = r"C:\Users\Loopy\Desktop\comfyui\ComfyUI\output\base_image_gen"
    GEN_IMG_DIR = r"C:\Users\Loopy\Desktop\comfyui\ComfyUI\output\image_gen"

    base_prompts_by_name = {bp["base_name"]: bp for bp in plan.get("base_prompts", [])}

    # --- ЭТАП 2 (chain-aware batch / variant B) ---
    #
    # Идея: вместо двух отдельных циклов "все Flux2 → все LTX" обрабатываем сцены
    # *по слоям chain'ов*. Chain — это блок [new_cut] + последующие [soft/hard]
    # сцены до следующего new_cut. На каждой глубине d:
    #   - Phase A: Flux2-keyframe для всех сцен глубины d одним батчем (одна
    #              загрузка модели). Picture 1 для soft/hard = последний кадр
    #              предыдущей LTX-сцены той же chain (вытащен ffmpeg'ом в Phase C
    #              предыдущей итерации).
    #   - Phase B: Llama → video_prompt по свежему keyframe'у (Comfy выгружен).
    #   - Phase C: LTX-видео для всех сцен глубины d. После каждого видео
    #              ffmpeg'ом тащим last_frame для следующей глубины.
    #
    # На выходе continuity нести через РЕАЛЬНОЕ движение из LTX, а не через
    # идентичный Flux2-still предыдущей сцены — это и фиксит "одинаковые кадры".

    LAST_FRAMES_DIR = (
        r"C:\Users\Loopy\Desktop\comfyui\ComfyUI\output\last_frames"
    )
    os.makedirs(LAST_FRAMES_DIR, exist_ok=True)

    # unique_path нужен ДО depth-loop'а — LTX пишет туда же.
    folder_to_create = (
        f"C:\\Users\\Loopy\\Desktop\\comfyui\\ComfyUI\\output\\"
        f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
    )
    unique_path = get_unique_dir_name(folder_to_create)
    os.makedirs(unique_path)

    scenes = plan["scenes"]
    n_scenes = len(scenes)
    await comfy_mgr.restart()


    # Comfy после ЭТАП 1.5 выгружен (llama работала). Поднимаем перед
    # Phase A; дальше внутри depth-loop'а Comfy↔Llama меняются локально:
    # A (Comfy) -> B (Llama vision) -> C (Comfy).

    print(f"--- Depth Phase A: Klein keyframes ---")

    for i in range(n_scenes):
        scene = scenes[i]

        final_prompt, entities = build_qwen_edit_prompt(
            image_prompt=scene["image_prompt"],
            base_prompts_by_name=base_prompts_by_name,
            max_pictures=5,
        )
        base_image_paths = [
            f"{REFS_DIR}\\"
            + entity_ref_filename(
                name, ""
            )
            for name in entities
        ]

        image_paths = base_image_paths
        image_paths = image_paths[:5]

        kf_filename_prefix = f"image_gen/{i + 1}_c1"
        print(
            f"    scene {scene['scene_id']} c1: "
            f"entities={entities}"
        )
        print(f"      final_prompt: {final_prompt}")
        print(f"      image_paths: {image_paths}")
        wf_kf = patch_hidream_workflow(
            workflow=workflow_image_hidream,
            image_paths=image_paths,
            positive_prompt=final_prompt,
            filename_prefix=kf_filename_prefix,
            seed=secrets.randbelow(2**63),
        )
        prompt_id_img = await comfy.queue_prompt(wf_kf)
        await comfy.wait_for_execution(prompt_id_img, workflow=wf_kf)
        kf_path = f"{GEN_IMG_DIR}\\{i + 1}_c1_00001_.png"
        print(f"      keyframe готов: {kf_path}")

    # ----- Phase B: Gemma vision plan (single pass) -----
    # Gemma видит свежеотрендеренный keyframe + source image/video
    # prompts из main.json + recap. Один проход — никаких drafts'ов
    # и refine'ов поверх. Результат: финальный chunk_plan
    # (num_chunks + global_prompt + per-chunk video_prompt + length_frames).
    print(f"\n--- Depth / Phase B: Gemma vision plan ---")
    await comfy_mgr.stop()
    await llama_mgr.start()
    print(f"llama-server готов на {llama_mgr.base_url}")
    try:
        for i in range(n_scenes):
            scene = scenes[i]
            print(
                f"  scene {scene['scene_id']}: gemma vision plan from "
                f"keyframe {os.path.basename(f"{GEN_IMG_DIR}\\{i + 1}_c1_00001_.png")}"
            )
            plan_i = await compute_ltx_chunk_plan_for_scene(
                scene,
                scene_idx=i,
                srt_json=SRT_json,
                keyframe_path=f"{GEN_IMG_DIR}\\{i + 1}_c1_00001_.png",
                fps=LTX_CHUNK_FPS,
                ollama_url=LLAMA_URL,
            )
            chunk_plans[i] = plan_i
    finally:
        print("Гасим llama-server...")
        await llama_mgr.stop()

    # Дамп всех планов Phase B в video_prompts/ltx_prompts.json. Файл всегда
    # перезаписывается. Назначение: после ошибки/неудачной генерации можно
    # глазами посмотреть, что Gemma выдала, а Phase C читает планы отсюда —
    # т.е. JSON можно вручную поправить и перезапустить только Phase C.
    save_ltx_prompts(chunk_plans, scenes)

    # chunk_plans[i] теперь содержит финальный план: global_prompt +
    # per-chunk video_prompts (vision-grounded) + length_frames.
    # patch_ltx_multikf_workflow инжектит их в Prompt Relay timeline (node 386).
    print(f"\n--- Depth / Phase C: LTX multi-kf videos ---")
    # Phase C читает планы из файла (а не из памяти): если правили JSON
    # вручную между фазами — подхватятся именно правки.
    chunk_plans = load_ltx_prompts(n_scenes)
    await comfy_mgr.start()
    for i in range(n_scenes):
        scene = scenes[i]
        plan_i = chunk_plans[i]
        if plan_i is None:
            raise RuntimeError(
                f"chunk_plans[{i}] is None — Phase B (Gemma vision plan) "
                f"не отработал для сцены {scene.get('scene_id', i)}"
            )
        scene_kf_paths = [f"{GEN_IMG_DIR}\\{i + 1}_c1_00001_.png"]
        wf_vid = patch_ltx_multikf_workflow(
            workflow=workflow_vid,
            keyframe_paths=scene_kf_paths,
            chunk_plan=plan_i,
            filename_prefix=f"{unique_path}\\{i + 1}",
            seed=secrets.randbelow(2**63),
            fps=LTX_CHUNK_FPS,
            scene_id=scene.get("scene_id", i),
            base_prompts_by_name=base_prompts_by_name,
        )
        print(
            f"  scene {scene['scene_id']}: "
            f"num_kf={plan_i['num_chunks']}, "
            f"total_frames={plan_i['total_frames']}, queue LTX"
        )
        prompt_id_vid = await comfy.queue_prompt(wf_vid)
        vid_files = await comfy.wait_for_execution(prompt_id_vid, workflow=wf_vid)
        print(
            f"  scene {scene['scene_id']}: видео готово "
            f"({vid_files[0] if vid_files else '?'})"
        )

        # --- обрезка до точного числа кадров от TTS ---
        # LTX рендерит total_frames, округлённый ВВЕРХ до LTX-valid 8k+1
        # (compute_scene_total_frames, mode="up"), поэтому клип на 1..8
        # кадров длиннее слота, который TTS отвёл сцене. Срезаем хвост до
        # точного SRT_json["scenes"][i]["frames"], иначе видео уплывает
        # относительно аудио и рассинхрон копится от сцены к сцене.
        try:
            tts_frames = int(SRT_json["scenes"][i]["frames"])
        except (KeyError, IndexError, ValueError, TypeError):
            tts_frames = None
        scene_video_path = find_video_for_scene(unique_path, i)
        if tts_frames and tts_frames > 0 and scene_video_path:
            await trim_video_to_frames(
                scene_video_path, tts_frames, fps=LTX_CHUNK_FPS
            )
        else:
            print(
                f"  scene {scene['scene_id']}: пропускаю обрезку "
                f"(tts_frames={tts_frames}, file={scene_video_path})"
            )


    print("\nПайплайн (chain-aware) успешно завершен!")
    await comfy_mgr.stop()

    print("Копирование файлов")
    source_dir = f"C:\\Users\\Loopy\\Desktop\\comfyui\\ComfyUI\\output\\AI_VIDEO"
    target_dir = f"{unique_path}"
    files_to_copy = ["list.txt", "output_00001_.mp3", "audio.mp3", "subs.ass"]
    for file_name in files_to_copy:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Скопирован: {file_name}")
    command = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "list.txt",
        "-c",
        "copy",
        "output.mp4",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir,
    )

    stdout, stderr = await process.communicate()

    audio_duration = get_audio_duration("output_00001_.mp3")
    command = [
        "ffmpeg",
        "-i",
        "output.mp4",
        "-i",
        "output_00001_.mp3",
        "-filter_complex",
        "[0:v]tpad=stop_mode=clone:stop=-1[v]",
        "-map",
        "[v]",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        # ЖЕСТКО ограничиваем длину итогового видео длиной аудиодорожки
        "-t",
        str(audio_duration),
        # Флаг -y автоматически перезапишет output1.mp4, если он уже существует
        "-y",
        "output1.mp4",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir,
    )

    stdout, stderr = await process.communicate()

    command = [
        "ffmpeg",
        "-i",
        "output1.mp4",
        "-stream_loop",
        "-1",
        "-i",
        "audio.mp3",
        "-filter_complex",
        "[1:a]volume=-25dB[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "video.mp4",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir,
    )

    stdout, stderr = await process.communicate()

    command = [
        "ffmpeg",
        "-y",
        "-i",
        "video.mp4",
        "-vf",
        "scale=1024:1024:force_original_aspect_ratio=decrease,pad=1024:1820:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "output_9x16.mp4",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir,
    )

    stdout, stderr = await process.communicate()

    # ! SUBTITLES BURN-IN !

    command = [
        "ffmpeg",
        "-i",
        "output_9x16.mp4",
        "-vf",
        "ass=subs.ass",
        "-c:a",
        "copy",
        "video_with_subs.mp4",
    ]

    print("Накладываем субтитры...")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir,
    )

    stdout, stderr = await process.communicate()

    from create_cover import create_cover
    create_cover(
        input_image_path="C:\\Users\\Loopy\\Desktop\\comfyui\\ComfyUI\\output\\image_gen\\1_c1_00001_.png",
        text="Богатейший извозчик: 6 часть",
        output_path="result_cover.jpg",
        font_size=150,
    )

if __name__ == "__main__":
    asyncio.run(main())
