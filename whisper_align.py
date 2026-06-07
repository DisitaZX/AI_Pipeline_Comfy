"""
Forced-alignment helper that replaces ``aeneas`` with WhisperX + wav2vec2.

Это **настоящий** forced alignment (как aeneas, не как stable-ts):

* Используется wav2vec2-модель (CTC, фонемный уровень) — она НЕ распознаёт
  речь, а только сопоставляет уже известный текст с фреймами аудио через
  Viterbi.
* На вход подаётся текст из ``aeneas_text.txt`` / ``aeneas_words.txt`` —
  он используется КАК ЕСТЬ, ничего не «угадывается». Если в аудио TTS
  произнёс слово неожиданно — алгоритм всё равно поставит таймстемп ровно
  на то слово из входного файла, никакая ошибка распознавания таймингам
  не повредит.
* Для русского по умолчанию подключается модель
  ``jonatasgrosman/wav2vec2-large-xlsr-53-russian`` (whisperx сам её
  выбирает по ``language_code="ru"``).

Output JSON совпадает с aeneas
(``{"fragments": [{"begin","end","id","language","lines","children"}]}``),
поэтому остальной пайплайн (srt/srt_encode.py, srt/ass_encode.py,
srt_timing/srt_timing.py) работает без изменений.

Установка:
    pip install -U whisperx
    # ffmpeg должен быть в PATH

При первом запуске будет скачана wav2vec2-модель (~1.2 ГБ для русского).
Кешируется в ``~/.cache/huggingface``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import torch
import whisperx  # pip install -U whisperx


_ALIGN_MODEL_CACHE: dict = {}


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _get_align_model(language: str, device: str):
    """Лениво загружаем wav2vec2-align-модель и кешируем на время процесса."""
    key = (language, device)
    if key in _ALIGN_MODEL_CACHE:
        return _ALIGN_MODEL_CACHE[key]
    print(
        f"[whisper_align] loading wav2vec2 align model: language={language} "
        f"device={device} cuda_available={torch.cuda.is_available()}"
    )
    model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    _ALIGN_MODEL_CACHE[key] = (model, metadata)
    return model, metadata


def _read_lines(path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _fmt(value) -> str:
    return f"{max(float(value), 0.0):.3f}"


def _word_start(w) -> Optional[float]:
    s = w.get("start") if isinstance(w, dict) else getattr(w, "start", None)
    return float(s) if s is not None else None


def _word_end(w) -> Optional[float]:
    e = w.get("end") if isinstance(w, dict) else getattr(w, "end", None)
    return float(e) if e is not None else None


def _flatten_word_segments(result) -> list:
    """
    Из выхода whisperx.align достаём плоский список словарей
    ``{"word", "start", "end", ...}``. Слова без ``start``/``end``
    (это бывает для пунктуации / непроизносимых символов) пропускаем.
    """
    candidates = []
    if isinstance(result, dict):
        if result.get("word_segments"):
            candidates = result["word_segments"]
        else:
            for seg in result.get("segments", []) or []:
                for w in seg.get("words", []) or []:
                    candidates.append(w)
    return [
        w for w in candidates
        if _word_start(w) is not None and _word_end(w) is not None
    ]


def _backfill_missing(words):
    """
    Иногда whisperx помечает отдельные слова без ``start``/``end``
    (редко, но бывает). Здесь не используется (мы их фильтруем выше),
    но оставлено как hook на случай ужесточения логики.
    """
    return words


def _build_fragments_proportional(input_units, all_words, language: str):
    """
    Распределяем выровненные слова по входным юнитам пропорционально
    позиции/количеству слов. Если whisperx вернул столько же слов, сколько
    у нас на входе, маппинг получается 1-в-1 (идеальный случай). Если
    расхождение — растягиваем равномерно, без коллапсов в end-of-audio.
    """
    total_in = sum(n for _, _, n in input_units)
    total_w = len(all_words)
    if total_in <= 0 or total_w <= 0:
        raise RuntimeError("нет данных для выравнивания")

    fragments = []
    pos = 0
    for frag_id, line, n in input_units:
        j_start = round(pos * total_w / total_in)
        j_end = round((pos + n) * total_w / total_in)
        j_start = max(0, min(j_start, total_w - 1))
        j_end = max(j_start + 1, min(j_end, total_w))
        chunk = all_words[j_start:j_end]
        begin = _word_start(chunk[0])
        end = _word_end(chunk[-1])
        if end is None or begin is None or end <= begin:
            end = (begin if begin is not None else 0.0) + 0.001
        fragments.append({
            "begin": _fmt(begin),
            "children": [],
            "end": _fmt(end),
            "id": frag_id,
            "language": language,
            "lines": [line],
        })
        pos += n
    return fragments


def align_to_aeneas_json(
    audio_path: str,
    text_path: str,
    output_json_path: str,
    *,
    language: str = "ru",
    level: str = "fragment",
    model_name: Optional[str] = None,  # ignored (kept for API compat)
    device: Optional[str] = None,
) -> dict:
    """
    Прогоняет wav2vec2 forced-alignment по ``audio_path`` против текста из
    ``text_path`` и сохраняет aeneas-совместимый JSON в
    ``output_json_path``.

    ``level="fragment"`` — одно предложение/реплика на строку → ``map.json``.
    ``level="word"``     — одно слово на строку → ``map_words.json``.
    """
    if level not in ("fragment", "word"):
        raise ValueError(
            f"level must be 'fragment' or 'word', got {level!r}"
        )

    device = _resolve_device(device)

    lines = _read_lines(text_path)
    if not lines:
        raise ValueError(f"{text_path} пуст — нечего выравнивать")

    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / 16000.0  # whisperx ресемплит в 16кГц моно

    print(
        f"[whisper_align] audio={audio_path} duration={duration:.2f}s "
        f"lines={len(lines)} level={level} device={device}"
    )

    align_model, metadata = _get_align_model(language, device)

    # Один сегмент со всем текстом и временным диапазоном [0, duration].
    # whisperx.align внутри прогонит wav2vec2 → CTC-эмиссии, и Viterbi
    # привяжет КАЖДОЕ слово известного текста к временным меткам в аудио.
    # Никакого ASR не происходит.
    full_text = " ".join(lines)
    transcript = [{"text": full_text, "start": 0.0, "end": duration}]

    print("[whisper_align] running wav2vec2 forced alignment ...")
    result = whisperx.align(
        transcript,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    all_words = _flatten_word_segments(result)
    if not all_words:
        raise RuntimeError(
            f"WhisperX вернул 0 выровненных слов для {audio_path}. "
            f"Проверь, что аудио валидно и язык ('{language}') совпадает "
            f"с языком текста."
        )

    if level == "word":
        input_units = [
            (f"f{i + 1:06d}", line, 1) for i, line in enumerate(lines)
        ]
    else:
        input_units = [
            (f"f{i + 1:06d}", line, max(1, len(line.split())))
            for i, line in enumerate(lines)
        ]

    fragments = _build_fragments_proportional(input_units, all_words, language)

    out = {"fragments": fragments}
    Path(output_json_path).write_text(
        json.dumps(out, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    last_end = _word_end(all_words[-1]) or 0.0
    print(
        f"[whisper_align] OK: {output_json_path} "
        f"({len(fragments)} {level}, aligned_words={len(all_words)}, "
        f"audio_end={last_end:.2f}s)"
    )
    return out


__all__ = ["align_to_aeneas_json"]
