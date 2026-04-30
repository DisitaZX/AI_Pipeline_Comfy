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
sys.path.append("./")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from srt_timing.srt_timing import SRTSceneTimeline
from comics.animator import ImagePanner
from srt.ass_encode import fragments_to_ass
from comfy_manager import ComfyManager
from chunk_prompts import split_into_chunks, get_chunk_prompts_from_ollama, get_image_prompt_from_ollama
from rife_interp import rife_interpolate_inplace

OLLAMA_URL = "http://localhost:11434/api/generate"
COMFYUI_URL = "127.0.0.1:8188"

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
    with open('Test-Image-Long.json', 'r', encoding='utf-8') as f:
        workflow_img = json.load(f)
    with open('VID_WAN.json', 'r', encoding='utf-8') as f:
        workflow_vid = json.load(f)

    # --- ЭТАП 2: Аудио ---
    """print("\n2. Запускаем генерацию TTS ...")
    tts_node_id = "1"
    workflow_tts[tts_node_id]["inputs"]["text"] = plan["tts_text"]

    prompt_id_tts = await comfy.queue_prompt(workflow_tts)
    audio_files = await comfy.wait_for_execution(prompt_id_tts)
    print(f"Аудио сгенерировано: {audio_files[0] if audio_files else 'Ошибка'}")"""

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

    # --- ЭТАП 2 и 3: Изображения и Видео ---
    """for i, scene in enumerate(plan["scenes"]):
        print(f"\n--- Обработка сцены {scene['scene_id']} ---")
        # 1. Генерируем изображение
        print(f"Генерация изображения {i + 1}")
        source_prompt = scene["image_prompt"]
        for n in plan["base_prompts"]:
            source_prompt = source_prompt.replace(f"[{n['base_name']}]", n["base_prompt"])

        image_result = await get_image_prompt_from_ollama(
            source_prompt=source_prompt,
            previous_recap=scene["previous_recap"],
        )
        full_prompt = image_result["prompt"]
        negative_prompt = image_result["negative_prompt"]
        print(full_prompt)

        img_prompt_node_id = "575"
        img_seed = "307"
        workflow_img[img_seed]["inputs"]["value"] = f"{secrets.randbelow(2 ** 32)}"
        workflow_img[img_prompt_node_id]["inputs"]["value"] = full_prompt
        workflow_img["9"]["inputs"]["filename_prefix"] = f"image_gen/{i + 1}"

        workflow_img["243"]["inputs"]["value"] = 720
        workflow_img["248"]["inputs"]["value"] = 1280

        prompt_id_img = await comfy.queue_prompt(workflow_img)
        img_files = await comfy.wait_for_execution(prompt_id_img)
        generated_image_name = img_files[0]
        print(f"Изображение готово: {generated_image_name}")"""

    await unload_ollama_model()
    await comfy_mgr.restart()
    folder_to_create = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\{datetime.datetime.today().strftime('%Y-%m-%d')}"
    unique_path = get_unique_dir_name(folder_to_create)
    os.makedirs(unique_path)

    # --- ЭТАП 3.1: Получение промптов для видео от Gemma ---
    print("\n--- Сбор промптов для видео ---")
    video_prompts = []
    scene_frames = []

    for i, scene in enumerate(plan["scenes"]):
        print(f"Генерация промпта для сцены {scene['scene_id']}...")
        image_path = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i + 1}_00001_.png"
        frames = split_into_chunks(SRT_json["scenes"][i]["frames"])

        prompt = await get_chunk_prompts_from_ollama(scene["text_scene"], scene["video_prompt"], image_path, len(frames), frames, fps=16, previous_recap=scene["previous_recap"])
        video_prompts.append(prompt)
        scene_frames.append(frames)
        print(prompt)
        print(f"Промпт для сцены {scene['scene_id']} получен.")

    await unload_ollama_model()

    await comfy_mgr.restart()

    # --- ЭТАП 3.2: Генерация видео в ComfyUI ---
    print("\n--- Генерация видео ---")
    for i, scene in enumerate(plan["scenes"]):
        prompt = video_prompts[i]
        frames = scene_frames[i]
        lenPrompts = len(prompt)
        wf = set_chunk_count(workflow_vid, lenPrompts)
        if lenPrompts == 1:
            wf["623:596"]["inputs"]["text"] = prompt[0]
            wf["608"]["inputs"]["value"] = frames[0]
        elif lenPrompts == 2:
            wf["623:596"]["inputs"]["text"] = prompt[0]
            wf["608"]["inputs"]["value"] = frames[0]
            wf["336:334"]["inputs"]["text"] = prompt[1]
            wf["698"]["inputs"]["value"] = frames[1]
        else:
            wf["623:596"]["inputs"]["text"] = prompt[0]
            wf["608"]["inputs"]["value"] = frames[0]
            wf["336:334"]["inputs"]["text"] = prompt[1]
            wf["698"]["inputs"]["value"] = frames[1]
            wf["644:675"]["inputs"]["text"] = prompt[2]
            wf["700"]["inputs"]["value"] = frames[2]

        wf["702"]["inputs"]["image"] = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i + 1}_00001_.png"

        wf["656"]["inputs"]["filename_prefix"] = f"{unique_path}\\{i + 1}"

        print(f"Отправка сцены {scene['scene_id']} на генерацию видео...")
        prompt_id_vid = await comfy.queue_prompt(wf)
        vid_files = await comfy.wait_for_execution(prompt_id_vid)
        if vid_files:
            video_path = f"{unique_path}\\{vid_files[0]}"
            print(f"RIFE 16->30fps inplace: {video_path}")
            rife_interpolate_inplace(video_path, multiplier=2, target_fps=30)
            print(f"Видео для сцены {scene['scene_id']} готово: {vid_files[0]}")
        else:
            print(f"Видео для сцены {scene['scene_id']} готово: Ошибка")

    print("\nПайплайн успешно завершен!")
    await comfy_mgr.stop()
    """panner = ImagePanner()
    for i, scene in enumerate(plan["scenes"]):

        panner.animate_image(f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i+1}_00001_.png",
                             audio_duration=float(SRT_json['scenes'][i]["video_duration"]), output_path=f"{unique_path}\\{i+1}_00001_.mp4")"""

    print("Копирование файлов")
    source_dir = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\AI_VIDEO"
    target_dir = f"{unique_path}"
    files_to_copy = ['list_vid.txt', 'output_00001_.mp3', 'audio.mp3', 'subs.ass']
    for file_name in files_to_copy:
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)

        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"Скопирован: {file_name}")
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "list_vid.txt", "-c", "copy", "output.mp4"
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

    # ! SUBTITLES BURN-IN !

    command = [
        "ffmpeg", "-i", "video.mp4", "-vf", "ass=subs.ass",
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
