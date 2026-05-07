import asyncio
import time
import subprocess
import aiohttp
import shutil
import websockets
import json
import uuid
import os
import sys
import pprint
import secrets
import pprint
import datetime
import base64
import re
import torch
import sys
import copy
sys.path.append("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from srt_timing.srt_timing import SRTSceneTimeline
from comics.animator import ImagePanner
from srt.ass_encode import fragments_to_ass
from comfy_manager import ComfyManager
from llama_manager import llama_mgr
from chunk_prompts_2 import (split_into_chunks, get_chunk_prompts_from_ollama,
                           substitute_entities_to_image_slots, expand_image_prompt_for_qwen_edit,
                           precompute_all_image_prompts, get_video_prompt_for_ltx_from_ollama, round_to_valid_frames_ltx, pick_camera_preset)
from prompt_builder import build_qwen_edit_prompt

LLAMA_URL = "http://localhost:8080/v1/chat/completions"
COMFYUI_URL = "127.0.0.1:8188"

def patch_flux2_edit_workflow(
    workflow: dict,
    image_paths: list[str],          # 1..5 absolute paths, ordered like [image 1..N]
    positive_prompt: str,
    filename_prefix: str,
    negative_prompt: str | None = None,  # None → не трогаем хардкод в ноде 143
    width: int = 1280,
    height: int = 720,
    seed: int | None = None,
    ) -> dict:
    """
    Патчит flux2-klein-edit.json под текущую сцену.

    FLUX.2 Klein использует chained ReferenceLatent: positive- и negative-
    conditioning поочерёдно "обвешиваются" reference-латентами для каждой
    image. Точка съёма для CFGGuider зависит от количества reference'ов
    (см. таблицу _CFG_ENDPOINTS_BY_N ниже). Неиспользуемые звенья цепочки
    становятся unreachable от output-ноды 94 и ComfyUI их не выполнит.
    """
    wf = copy.deepcopy(workflow)
    n = len(image_paths)
    if not 1 <= n <= 5:
        raise ValueError(f"image_paths должен иметь 1..5 элементов, got {n}")

    # VHS_LoadImagePath ноды в порядке image1..image5
    loader_nodes = ["160", "161", "162", "163", "171"]
    for i, path in enumerate(image_paths):
        wf[loader_nodes[i]]["inputs"]["image"] = path

    # CFGGuider 144 цепляется к нужному звену ReferenceLatent цепочки
    cfg_endpoints_by_n = {
        1: ("148", "146"),
        2: ("151", "149"),
        3: ("155", "154"),
        4: ("158", "156"),
        5: ("169", "170"),
    }
    pos_node, neg_node = cfg_endpoints_by_n[n]
    wf["144"]["inputs"]["positive"] = [pos_node, 0]
    wf["144"]["inputs"]["negative"] = [neg_node, 0]

    # промпты (CLIPTextEncode)
    wf["142"]["inputs"]["text"] = positive_prompt
    if negative_prompt is not None:
        wf["143"]["inputs"]["text"] = negative_prompt

    # выход
    wf["94"]["inputs"]["filename_prefix"] = filename_prefix

    # размеры — в EmptyFlux2LatentImage (138) и в Flux2Scheduler (145)
    wf["138"]["inputs"]["width"] = width
    wf["138"]["inputs"]["height"] = height
    wf["145"]["inputs"]["width"] = width
    wf["145"]["inputs"]["height"] = height

    # seed (RandomNoise)
    if seed is not None:
        wf["134"]["inputs"]["noise_seed"] = seed

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
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return float(result.stdout.strip())

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


async def unload_ollama_model(model_name="gemma4:e4b"):
    """Принудительно выгружает модель из VRAM."""
    print(f"\n[Очистка] Выгружаем {model_name} из памяти Ollama...")
    async with aiohttp.ClientSession() as session:
        payload = {"model": model_name, "keep_alive": 0}
        await session.post(OLLAMA_URL, json=payload)

def split_by_sentence(text):
    # Находим индекс фактической середины
    mid_index = len(text) // 2

    # Ищем ближайшую точку после середины
    end_of_sentence = text.find('.', mid_index)

    # Если точка после середины не найдена, ищем до середины
    if end_of_sentence == -1:
        end_of_sentence = text.rfind('.', 0, mid_index)

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
        """Отправляет workflow в очередь и возвращает ID задачи."""
        p = {"prompt": prompt_workflow, "client_id": self.client_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"http://{self.server_address}/prompt", json=p) as response:
                data = await response.json()
                return data['prompt_id']

    async def wait_for_execution(self, prompt_id):
        """Слушает вебсокет и ждет завершения генерации, возвращая имена файлов."""
        ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
        async with websockets.connect(ws_url) as ws:
            while True:
                out = await ws.recv()
                if isinstance(out, str):
                    message = json.loads(out)
                    if message['type'] == 'executing':
                        data = message['data']
                        if data['node'] is None and data['prompt_id'] == prompt_id:
                            break  # Генерация завершена

            # После завершения запрашиваем историю, чтобы получить имена сохраненных файлов
            return await self.get_history(prompt_id)

    async def get_history(self, prompt_id):
        """Получает результаты выполнения по ID задачи."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{self.server_address}/history/{prompt_id}") as response:
                history = await response.json()

                output_files = []
                # Ищем ноды, которые сохранили картинки/видео/аудио
                outputs = history[prompt_id]['outputs']
                for node_id in outputs:
                    node_output = outputs[node_id]
                    if 'images' in node_output:
                        for img in node_output['images']:
                            output_files.append(img['filename'])
                    elif 'gifs' in node_output:  # Для видео формата
                        for vid in node_output['gifs']:
                            output_files.append(vid['filename'])
                    elif 'audio' in node_output:  # Для аудио
                        for aud in node_output['audio']:
                            output_files.append(aud['filename'])

                return output_files


async def main():
    # print("1. Обращаемся к Qwen 3.5 за сценарием...")
    # plan = await get_plan_from_ollama()
    COMFY_ROOT = r"C:\Users\Loopy\Desktop\ComfyUI_windows_portable"
    comfy_mgr = ComfyManager(
        comfy_root=COMFY_ROOT,
        host="127.0.0.1",
        port=8188,
        # extra_args=["--lowvram"],  # если нужно
    )
    await comfy_mgr.start()
    with open('main.json', 'r', encoding='utf-8') as f:
        plan = json.load(f)
    print(f"План получен! Сцен для генерации: {len(plan['scenes'])}")

    aeneas_text = ''
    aeneas_words = ''
    for k in plan["scenes"]:
        aeneas_text += k["text_scene"] + '\n'
        for w in k["text_scene"].split():
            aeneas_words += w + '\n'
    print(aeneas_text)
    with open('aeneas_text.txt', 'w', encoding='utf-8') as f:
        f.write(aeneas_text)
    with open('aeneas_words.txt', 'w', encoding='utf-8') as f:
        f.write(aeneas_words)

    comfy = ComfyUIClient(COMFYUI_URL)

    # Загружаем шаблоны (предварительно сохраненные через "Save (API Format)" в ComfyUI)
    with open('TTS_Cozy.json', 'r', encoding='utf-8') as f:
        workflow_tts = json.load(f)
    with open('image_qwen_Image_2512.json', 'r', encoding='utf-8') as f:
        workflow_img = json.load(f)
    with open('flux2-klein-edit.json', 'r', encoding='utf-8') as f:
        workflow_flux2 = json.load(f)
    with open('VID_LTX.json', 'r', encoding='utf-8') as f:
        workflow_vid = json.load(f)

    # --- ЭТАП 2: Аудио ---
    print("\n2. Запускаем генерацию TTS ...")
    tts_node_id = "1"
    workflow_tts[tts_node_id]["inputs"]["text"] = plan["tts_text"]

    prompt_id_tts = await comfy.queue_prompt(workflow_tts)
    audio_files = await comfy.wait_for_execution(prompt_id_tts)
    print(f"Аудио сгенерировано: {audio_files[0] if audio_files else 'Ошибка'}")

    # ! AENEAS !
    command = [
        "C:/Program Files/Python39/python.exe", "-m", "aeneas.tools.execute_task",
        "output_00001_.mp3",
        "aeneas_text.txt",
        "task_language=ru|os_task_file_format=json|is_text_type=plain",
        "map.json"
    ]

    print(f"Запуск Aeneas для синхронизации...")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # ! AENEAS WORD-LEVEL !

    command_words = [
        "C:/Program Files/Python39/python.exe", "-m", "aeneas.tools.execute_task",
        "output_00001_.mp3",
        "aeneas_words.txt",
        "task_language=ru|os_task_file_format=json|is_text_type=plain",
        "map_words.json"
    ]

    print(f"Запуск Aeneas word-level...")

    process = await asyncio.create_subprocess_exec(
        *command_words,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # ! SRT_ENCODE !

    command = [
        "C:/Program Files/Python39/python.exe",
        "srt/srt_encode.py "
    ]

    print(f"Запуск SRT_ENCODE")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()

    # ! SRTSceneTimeline !

    with open("srt/srt_encode.srt", "r", encoding="utf-8") as f:
        text_srt = f.read()
    with open("aeneas_text.txt", "r", encoding="utf-8") as f:
        text_target = f.read()

    SceneTimeline = SRTSceneTimeline()

    SRT_json = SceneTimeline.build(text_srt, text_target)
    pprint.pprint(SRT_json)

    # ! ASS KARAOKE !

    fragments_to_ass("map_words.json", "subs.ass")

    # генерация base
    for i, scene in enumerate(plan["base_prompts"]):
        print(f"\n--- Обработка base prompt {i + 1} ---")
        # 1. Генерируем изображение

        full_prompt = scene["base_prompt"]
        print(full_prompt)

        workflow_img["272"]["inputs"]["seed"] = f"{secrets.randbelow(2 ** 32)}"
        workflow_img["268"]["inputs"]["text"] = full_prompt
        workflow_img["60"]["inputs"]["filename_prefix"] = f"base_image_gen/{scene["base_name"]}"

        workflow_img["271"]["inputs"]["width"] = 1280
        workflow_img["271"]["inputs"]["height"] = 720

        prompt_id_img = await comfy.queue_prompt(workflow_img)
        img_files = await comfy.wait_for_execution(prompt_id_img)
        generated_image_name = img_files[0]
        print(f"Изображение готово: {generated_image_name}")

    await comfy_mgr.restart()

    REFS_DIR = r"C:\Users\Loopy\Desktop\ComfyUI_windows_portable\ComfyUI\output\base_image_gen"
    GEN_IMG_DIR = r"C:\Users\Loopy\Desktop\ComfyUI_windows_portable\ComfyUI\output\image_gen"

    """"# --- ЭТАП 1.5: Batch-прогон Gemma по всем сценам ДО загрузки Qwen Edit ---
    # Иначе Comfy выжирает RAM и Ollama падает с OOM на второй сцене.
    print("\n=== Pre-batch: image prompts через Gemma ===")
    precomputed_image_prompts = await precompute_all_image_prompts(
        scenes=plan["scenes"],
    )"""

    base_prompts_by_name = {bp["base_name"]: bp for bp in plan.get("base_prompts", [])}
    # --- ЭТАП 2: Image generation с continuity ---
    # previous_image_path — путь к FLUX.2-выходу предыдущей сцены, используется
    # как Picture 1 для soft/hard_continue сцен. None для new_cut / scene 1.
    previous_image_path: str | None = None

    for i, scene in enumerate(plan["scenes"]):
        print(f"\n--- Обработка сцены {scene['scene_id']} ---")
        print(f"Генерация изображения {i + 1}")

        continuity_type = scene.get("continuity_type", "new_cut")
        # Безопасный fallback: scene 1 всегда new_cut
        if i == 0:
            continuity_type = "new_cut"

        # Если continuity != new_cut но previous_image_path не существует
        # (например GROK ошибочно поставил soft_continue для scene 1, или
        # предыдущая сцена не сгенерилась) — даунгрейдим на new_cut.
        if continuity_type != "new_cut" and (
                previous_image_path is None
                or not os.path.exists(previous_image_path)
        ):
            print(f"  [WARN] continuity={continuity_type}, но prev image отсутствует — fallback на new_cut")
            continuity_type = "new_cut"

        camera_preset = pick_camera_preset(i)

        # Маппинг continuity_type -> continuity_mode для prompt_builder
        cmode_map = {
            "new_cut": "none",
            "soft_continue": "soft",
            "hard_continue": "hard",
        }
        continuity_mode = cmode_map.get(continuity_type, "none")

        final_prompt, entities = build_qwen_edit_prompt(
            image_prompt=scene["image_prompt"],
            base_prompts_by_name=base_prompts_by_name,
            camera_preset=camera_preset,
            max_pictures=5,
            continuity_mode=continuity_mode,
        )

        # Сборка image_paths: prev-frame идёт первым если continuity активна
        base_image_paths = [f"{REFS_DIR}\\{name}_00001_.png" for name in entities]
        if continuity_mode in ("soft", "hard"):
            image_paths = [previous_image_path] + base_image_paths
        else:
            image_paths = base_image_paths

        # Cap на 5 (FLUX.2 Klein hard limit). На всякий случай — prompt_builder
        # уже это учитывает через picture_offset, но safety net.
        image_paths = image_paths[:5]

        print(f"continuity_type: {continuity_type}")
        print(f"prev_image: {previous_image_path}")
        print("Промпт (raw):")
        print(scene["image_prompt"])
        print("Промпт (final):")
        print(final_prompt)
        print("Image paths:", image_paths)

        wf_flux2 = patch_flux2_edit_workflow(
            workflow=workflow_flux2,
            image_paths=image_paths,
            positive_prompt=final_prompt,
            filename_prefix=f"image_gen/{i + 1}",
        )

        prompt_id_img = await comfy.queue_prompt(wf_flux2)
        img_files = await comfy.wait_for_execution(prompt_id_img)
        generated_image_name = img_files[0]
        print(f"Изображение готово: {generated_image_name}")

        # Зафиксировать путь для следующей итерации.
        # ComfyUI сохраняет как "{filename_prefix}_00001_.png" => image_gen/{i+1}_00001_.png
        previous_image_path = f"{GEN_IMG_DIR}\\{i + 1}_00001_.png"

    await comfy_mgr.stop()
    folder_to_create = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\{datetime.datetime.today().strftime('%Y-%m-%d')}"
    unique_path = get_unique_dir_name(folder_to_create)
    os.makedirs(unique_path)

    # --- ЭТАП 3.1: Получение промптов для видео от Qwen3-VL через llama-server ---
    print("\n--- Сбор промптов для видео ---")
    print("Стартуем llama-server...")
    await llama_mgr.start()
    print(f"llama-server готов на {llama_mgr.base_url}")

    video_prompts = []
    scene_frames = []

    try:
        for i, scene in enumerate(plan["scenes"]):
            print(f"Генерация промпта для сцены {scene['scene_id']}...")
            image_path = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i + 1}_00001_.png"
            frames_raw = SRT_json["scenes"][i]["frames"]
            frames = round_to_valid_frames_ltx(frames_raw, mode="nearest")
            prompt = await get_video_prompt_for_ltx_from_ollama(
                scene_text=scene["text_scene"],
                video_prompt=scene["video_prompt"],
                image_path=image_path,
                n_frames=frames,
                fps=24,
                previous_recap=scene["previous_recap"],
                continuity_type=scene.get("continuity_type", "new_cut"),  # NEW
            )
            video_prompts.append(prompt)
            scene_frames.append(frames)
            print(prompt)
            print(f"Промпт для сцены {scene['scene_id']} получен.")
    finally:
        # Всегда гасим сервер, даже если что-то упало в loop'е —
        # иначе модель так и будет жрать RAM/VRAM до конца жизни процесса.
        print("Гасим llama-server...")
        await llama_mgr.stop()
        print("llama-server остановлен")

    await comfy_mgr.start()

    # --- ЭТАП 3.2: Генерация видео в ComfyUI ---
    print("\n--- Генерация видео ---")
    for i, scene in enumerate(plan["scenes"]):
        prompt = video_prompts[i]
        frames = scene_frames[i]

        workflow_vid["367"]["inputs"]["value"] = prompt
        workflow_vid["375"]["inputs"]["image"] = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i + 1}_00001_.png"
        workflow_vid["374"]["inputs"]["value"] = frames
        workflow_vid["75"]["inputs"]["filename_prefix"] = f"{unique_path}\\{i + 1}"

        print(f"Отправка сцены {scene['scene_id']} на генерацию видео...")
        prompt_id_vid = await comfy.queue_prompt(workflow_vid)
        vid_files = await comfy.wait_for_execution(prompt_id_vid)
        print(f"Видео для сцены {scene['scene_id']} готово: {vid_files[0]}")

    print("\nПайплайн успешно завершен!")
    await comfy_mgr.stop()

    print("Копирование файлов")
    source_dir = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\AI_VIDEO"
    target_dir = f"{unique_path}"
    files_to_copy = ['list.txt', 'output_00001_.mp3', 'audio.mp3', 'subs.ass']
    for file_name in files_to_copy:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Скопирован: {file_name}")
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "output.mp4"
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

    audio_duration = get_audio_duration("output_00001_.mp3")
    command = [
        "ffmpeg",
        "-i", "output.mp4",
        "-i", "output_00001_.mp3",

        "-filter_complex", "[0:v]tpad=stop_mode=clone:stop=-1[v]",

        "-map", "[v]",
        "-map", "1:a:0",

        "-c:v", "libx264",
        "-c:a", "aac",
        "-b:a", "320k",

        # ЖЕСТКО ограничиваем длину итогового видео длиной аудиодорожки
        "-t", str(audio_duration),

        # Флаг -y автоматически перезапишет output1.mp4, если он уже существует
        "-y",

        "output1.mp4"
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

    command = [
        "ffmpeg", "-i", "output1.mp4", "-stream_loop", "-1", "-i", "audio.mp3", "-filter_complex",
        "[1:a]volume=-25dB[bg];[0:a][bg]amix=inputs=2:duration=first[aout]", "-map", "0:v", "-map", "[aout]", "-c:v",
        "copy", "video.mp4"
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

    command = [
        "ffmpeg", "-y",
        "-i", "video.mp4",
        "-vf", "scale=720:405:flags=lanczos,setsar=1,pad=720:1280:0:437:color=black",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "output_9x16.mp4",
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

    # ! SUBTITLES BURN-IN !

    command = [
        "ffmpeg", "-i", "output_9x16.mp4", "-vf", "ass=subs.ass",
        "-c:a", "copy", "video_with_subs.mp4"
    ]

    print("Накладываем субтитры...")

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()


if __name__ == "__main__":
    asyncio.run(main())
