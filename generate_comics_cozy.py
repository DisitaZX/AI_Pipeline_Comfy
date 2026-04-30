import asyncio
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
sys.path.append("./")
from srt_timing.srt_timing import SRTSceneTimeline
from comics.animator import ImagePanner

OLLAMA_URL = "http://localhost:11434/api/generate"
COMFYUI_URL = "127.0.0.1:8188"


async def get_plan_from_ollama():
    """Отправляет главу в Qwen 3.5 и получает JSON-план."""
    with open("chapter1.txt", "r", encoding="utf-8") as f:
        chapter_text = f.read()
    prompt = f"""
    Выступи в роли профессионального сценариста и AI-режиссера. Мне нужно создать сценарий и визуальные промпты для мини-сериала в формате вертикальных коротких видео (Shorts/TikTok/Reels). Я дам тебе главу ранобэ.

Инструменты, которые я использую:
- Озвучка: qwen3tts (стиль - китайский аниме рассказчик).
- Генерация изображений: Z Image Turbo (anime lora).
- Генерация видео: Hunyuan 1.5.

Визуальный стиль: (masterpiece, best quality, anime style) Но эти три характеристики я вставляю в каждый промпт независимо от твоего вывода.

Выполни задачу строго по следующим шагам:

ШАГ 1: Текст для озвучки (2 части)
Напиши захватывающую историю.
- Хронометраж: 2 минуты (примерно 280-300 слов).
- Текст должен звучать эпично и кинематографично. Язык текста: русский.

ШАГ 2: Базовые переменные для консистентности (на английском)
Создай 3-4 общие переменные для генерации изображений (например, Главный герой, Локация, Ключевой объект). Каждая переменная должна быть подробно описана ровно в 1 абзац. В описании обязательно используй сильные маркеры выбранного визуального стиля. Эти переменные будут подставляться в промпты в квадратных скобках (например, [Main Character]). Язык промптов: английский.

ШАГ 3: Раскадровка (12 сцен для каждой части)
Для каждой из двух частей распиши 12 сцен.
Формат каждой сцены:
1. Текст озвучки: (кусок текста из Шага 1), нельзя делить предложение на две части, только полные предложения.
2. Промпт для Z Image Turbo (на английском): Описание композиции, освещения и ситуации, обязательно с использованием переменных в квадратных скобках и тегов стиля.
3. Промпт для Hunyuan 1.5 (на английском): Описание ИСКЛЮЧИТЕЛЬНО движения, анимации, поведения камеры и эффектов. Не описывай внешность заново, описывай только то, что происходит в кадре (например, "Slow pan left, dust particles floating, hair moving in the wind").

    Ответь СТРОГО в формате JSON без markdown разметки и лишних слов.

    Структура JSON:
    {{
      "tts_text": "Сжатый текст главы для озвучки на 2 минуты",
      "base_prompts": [
        {{
          "base_id": 1,
          "base_name": "main_character",
          "base_prompt": "Большой промпт описания на английском"
        }}
      ]
      "scenes": [
        {{
          "scene_id": 1,
          "text_scene": "Здесь должен быть текст озвучки конкретной сцены",
          "image_prompt": "prompt for image generation, anime style",
          "video_prompt": "prompt for video animation, camera movement"
        }}
      ]
    }}

    Текст главы: {chapter_text}
    """
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "model": "qwen3.5:9b",
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        async with session.post(OLLAMA_URL, json=payload) as response:
            result = await response.json()
            return json.loads(result["thinking"])


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
    #print("1. Обращаемся к Qwen 3.5 за сценарием...")
    #plan = await get_plan_from_ollama()
    with open('main.json', 'r', encoding='utf-8') as f:
        plan = json.load(f)
    print(f"План получен! Сцен для генерации: {len(plan['scenes'])}")

    aeneas_text = ''
    for k in plan["scenes"]:
        aeneas_text += k["text_scene"] + '\n'
    print(aeneas_text)
    with open('aeneas_text.txt', 'w', encoding='utf-8') as f:
        f.write(aeneas_text)

    comfy = ComfyUIClient(COMFYUI_URL)

    # Загружаем шаблоны (предварительно сохраненные через "Save (API Format)" в ComfyUI)
    with open('TTS_Cozy.json', 'r', encoding='utf-8') as f:
        workflow_tts = json.load(f)
    with open('Test-Image_Long.json', 'r', encoding='utf-8') as f:
        workflow_img = json.load(f)
    with open('VID.json', 'r', encoding='utf-8') as f:
        workflow_vid = json.load(f)


    # --- ЭТАП 1: Аудио ---
    """print("\n2. Запускаем генерацию TTS...")
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


    # --- ЭТАП 2 и 3: Изображения и Видео ---
    """for i, scene in enumerate(plan["scenes"]):
        print(f"\n--- Обработка сцены {scene['scene_id']} ---")
        # 1. Генерируем изображение
        print(f"Генерация изображения {i+1}")
        full_prompt = scene["image_prompt"]
        for n in plan['base_prompts']:
            base_name = f'[{n['base_name']}]'
            if base_name in scene["image_prompt"]:
                full_prompt += ", " + base_name + ":" + n['base_prompt']
        print(full_prompt)

        img_prompt_node_id = "575"
        img_seed = "307"
        workflow_img[img_seed]["inputs"]["value"] = f"{secrets.randbelow(2**32)}"
        workflow_img[img_prompt_node_id]["inputs"]["value"] = full_prompt
        workflow_img["9"]["inputs"]["filename_prefix"] = f"image_gen/{i+1}"

        prompt_id_img = await comfy.queue_prompt(workflow_img)
        img_files = await comfy.wait_for_execution(prompt_id_img)
        generated_image_name = img_files[0]
        print(f"Изображение готово: {generated_image_name}")"""

    """for i, scene in enumerate(plan["scenes"]):
        # 2. Генерируем видео на основе изображения

        vid_load_image_node_id = "7151"
        vid_prompt_node_id = "7152:63"

        # Подставляем имя файла, полученное на предыдущем шаге!
        workflow_vid[vid_load_image_node_id]["inputs"]["image"] = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i+1}_00001_.png"
        workflow_vid[vid_prompt_node_id]["inputs"]["text"] = scene["video_prompt"]
        workflow_vid["7153"]["inputs"]["filename_prefix"] = f"{i+1}"
        workflow_vid["7152:64"]["inputs"]["length"] = frames[i]

        prompt_id_vid = await comfy.queue_prompt(workflow_vid)
        vid_files = await comfy.wait_for_execution(prompt_id_vid)
        print(f"Видео для сцены {scene['scene_id']} готово: {vid_files[0] if vid_files else 'Ошибка'}")"""

    print("\nПайплайн успешно завершен!")

    folder_to_create = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\{datetime.datetime.today().strftime('%Y-%m-%d')}"
    unique_path = get_unique_dir_name(folder_to_create)
    os.makedirs(unique_path)

    panner = ImagePanner()
    for i, scene in enumerate(plan["scenes"]):

        panner.animate_image(f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\image_gen\\{i+1}_00001_.png",
                             audio_duration=float(SRT_json['scenes'][i]["video_duration"]), output_path=f"{unique_path}\\{i+1}_00001_.mp4")

    print("Копирование файлов")
    source_dir = f"C:\\Users\\Loopy\\Desktop\\ComfyUI_windows_portable\\ComfyUI\\output\\AI_VIDEO"
    target_dir = f"{unique_path}"
    files_to_copy = ['list.txt', 'output_00001_.mp3', 'audio.mp3']
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

    command = [
        "ffmpeg", "-i", "output.mp4", "-i", "output_00001_.mp3", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "output1.mp4"
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

    command = [
        "ffmpeg", "-i", "output1.mp4", "-stream_loop", "-1", "-i", "audio.mp3", "-filter_complex", "[1:a]volume=-25dB[bg];[0:a][bg]amix=inputs=2:duration=first[aout]", "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "video.mp4"
    ]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=target_dir
    )

    stdout, stderr = await process.communicate()

if __name__ == "__main__":
    asyncio.run(main())
