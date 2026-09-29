"""Субтитры по РЕАЛЬНОЙ речи: WhisperX forced alignment поверх готовых клипов.

Идея (как в старом LTX-пайплайне с aeneas/whisperx, только автоматически):
  1. текст берём из main.json -> scenes[].dialogue[].line  (ничего не
     распознаём, слова известны заранее);
  2. аудио берём из УЖЕ СГЕНЕРИРОВАННЫХ клипов MiniMax (речь нативная,
     отдельного TTS-wav больше нет);
  3. wav2vec2 (CTC + Viterbi) ставит таймстемп каждому слову ВНУТРИ
     своего клипа -> субтитр идёт ровно за говорящим;
  4. абсолютное время = смещение клипа (кумулятивная длительность) +
     локальный таймстемп слова.

Выравнивание идёт ПОКЛИПНО, а не по всей склейке: так ошибка не копится,
и сцена без речи не утаскивает за собой остальные.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

_PUNCT_END = (".", "?", "!", ",", "—", ":", ";", "…")


def _ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _extract_wav(
    ffmpeg_bin: str, src: str, dst: str,
    start: float = 0.0, dur: float | None = None,
) -> None:
    """16 кГц моно WAV из окна [start, start+dur) файла src."""
    cmd = [ffmpeg_bin, "-y", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    subprocess.run(cmd, check=True)


def _clip_duration(ffprobe_bin: str, path: str, fallback: float) -> float:
    try:
        out = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            check=True, capture_output=True,
        ).stdout.decode(errors="ignore").strip()
        return float(out)
    except Exception:
        return fallback


def _scene_words(scene: dict) -> list[tuple[str, str, int]]:
    """[(слово, стиль, индекс реплики)] в порядке произнесения."""
    dlg = [d for d in (scene.get("dialogue") or []) if (d.get("line") or "").strip()]

    def _shot(d):
        try:
            return int(d.get("shot") or 1)
        except (TypeError, ValueError):
            return 1

    dlg.sort(key=_shot)
    words: list[tuple[str, str, int]] = []
    for li, d in enumerate(dlg):
        style = "Thought" if d.get("kind") == "thought" else "Speech"
        for w in (d["line"] or "").replace("\n", " ").split():
            words.append((w, style, li))
    return words


def _align_words(wav_path: str, words: list[str], language: str) -> list[tuple[float, float]]:
    """WhisperX forced alignment: на каждое входное слово -> (begin, end)."""
    from whisper_align import align_to_aeneas_json

    tmpdir = tempfile.mkdtemp(prefix="subs_align_")
    try:
        txt = os.path.join(tmpdir, "words.txt")
        out_json = os.path.join(tmpdir, "map_words.json")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("\n".join(words) + "\n")
        data = align_to_aeneas_json(
            wav_path, txt, out_json, language=language, level="word",
        )
        frags = data["fragments"]
        if len(frags) != len(words):
            raise RuntimeError(
                f"выравнивание вернуло {len(frags)} фрагментов на {len(words)} слов"
            )
        return [(float(f["begin"]), float(f["end"])) for f in frags]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _sanity(times: list[tuple[float, float]], dur: float) -> bool:
    """Отбраковываем явный мусор (всё слиплось в ноль / вышло за клип)."""
    if not times:
        return False
    if times[0][0] < -0.05 or times[-1][1] > dur + 0.75:
        return False
    if times[-1][1] - times[0][0] < 0.25:
        return False
    prev = -1.0
    for b, e in times:
        if e < b - 0.01 or b < prev - 0.35:
            return False
        prev = b
    return True


def _chunk(words, times, max_words: int, max_chars: int):
    """Группируем слова в титры, не смешивая стили и реплики."""
    events = []
    cur: list[int] = []
    for i, (w, style, li) in enumerate(words):
        if cur:
            pw, pstyle, pli = words[cur[0]][0], words[cur[0]][1], words[cur[0]][2]
            probe = " ".join(words[j][0] for j in cur + [i])
            same = (style == pstyle and li == pli)
            gap = times[i][0] - times[cur[-1]][1]
            if (not same or len(cur) >= max_words or len(probe) > max_chars
                    or gap > 0.7):
                events.append(cur)
                cur = []
        cur.append(i)
        if words[i][0].endswith(_PUNCT_END) and len(cur) >= 2:
            events.append(cur)
            cur = []
    if cur:
        events.append(cur)
    return events


def build_aligned_ass(
    scenes: list[dict],
    segments: list[tuple[str, float, float]],
    output_ass_path: str,
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_width: int,
    video_height: int,
    language: str = "ru",
    max_words: int = 4,
    max_chars: int = 30,
    lead_in: float = 0.06,
    tail: float = 0.12,
) -> int:
    """Строит .ass с таймингами из WhisperX. Возвращает число событий.

    segments[i] = (media_path, start_s, duration_s) — окно звука сцены i.
    Для покадровой генерации это (клип, 0.0, длина клипа); для
    Context Loop — (собранный MP4, начало окна сцены, её длительность).
    """
    fs_speech = max(20, int(video_height * 0.062))
    fs_thought = max(18, int(fs_speech * 0.92))
    margin_v = max(24, int(video_height * 0.07))
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {video_width}\nPlayResY: {video_height}\n"
        "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV\n"
        f"Style: Speech,Montserrat,{fs_speech},&H0000FFFF,&H00FFFFFF,"
        f"&H00000000,&H80000000,-1,0,1,3,1,2,24,24,{margin_v}\n"
        f"Style: Thought,Montserrat,{fs_thought},&H00FFFFFF,&H00FFFFFF,"
        f"&H00000000,&H80000000,0,-1,1,2,0,2,24,24,{int(margin_v * 1.9)}\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )

    lines_out: list[str] = []
    offset = 0.0
    aligned_scenes = 0
    for si, (scene, seg) in enumerate(zip(scenes, segments)):
        media, seg_start, seg_dur = seg
        dur = float(seg_dur) if seg_dur else _clip_duration(
            ffprobe_bin, media, float(scene.get("duration_s") or 8.0)
        )
        words = _scene_words(scene)
        if not words:
            offset += dur
            continue

        plain = [w for w, _, _ in words]
        times = None
        wav = os.path.splitext(media)[0] + f".align_{si + 1:02d}.wav"
        try:
            _extract_wav(ffmpeg_bin, media, wav, seg_start, dur)
            times = _align_words(wav, plain, language)
            if not _sanity(times, dur):
                print(f"[subs][WARN] сцена {scene.get('scene_id', si + 1)}: "
                      f"выравнивание выглядит некорректно")
                times = None
        except Exception as exc:
            print(f"[subs][WARN] сцена {scene.get('scene_id', si + 1)}: "
                  f"WhisperX не отработал ({exc})")
        finally:
            if os.path.exists(wav):
                os.remove(wav)

        if times is None:
            # без реальных таймингов субтитры не ставим: статик-текст,
            # разъезжающийся с речью, хуже его отсутствия
            print(f"[subs] сцена {scene.get('scene_id', si + 1)}: "
                  f"субтитры пропущены")
            offset += dur
            continue
        aligned_scenes += 1

        groups = _chunk(words, times, max_words, max_chars)
        for gi, g in enumerate(groups):
            start = offset + max(0.0, times[g[0]][0] - lead_in)
            end = offset + times[g[-1]][1] + tail
            if gi + 1 < len(groups):
                end = min(end, offset + times[groups[gi + 1][0]][0] - 0.04)
            end = min(end, offset + dur)
            if end <= start:
                end = start + 0.35
            text = " ".join(words[j][0] for j in g)
            style = words[g[0]][1]
            lines_out.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{text}"
            )
        offset += dur

    with open(output_ass_path, "w", encoding="utf-8-sig") as f:
        f.write(header + "\n".join(lines_out) + "\n")
    print(f"[subs] WhisperX-субтитры: {output_ass_path} "
          f"({len(lines_out)} событий, выровнено сцен: "
          f"{aligned_scenes}/{len(segments)})")
    return len(lines_out)


__all__ = ["build_aligned_ass"]
