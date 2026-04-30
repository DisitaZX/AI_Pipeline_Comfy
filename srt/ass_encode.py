import json

def seconds_to_ass_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int((sec - int(sec)) * 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"

def fragments_to_ass(map_words_json_path: str, output_ass_path: str, max_words=3):
    with open(map_words_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fragments = data["fragments"]
    sentences = []
    current_words = []

    for frag in fragments:
        word = " ".join(frag["lines"]).strip()
        begin = float(frag["begin"])
        end = float(frag["end"])

        current_words.append({"word": word, "begin": begin, "end": end})

        # Разбиваем по 3 слова (стандарт TikTok/Shorts) или по препинаниям
        if len(current_words) >= max_words or word.endswith(('.', '?', '!', ',', '—')):
            sentences.append(current_words)
            current_words = []

    if current_words:
        sentences.append(current_words)

    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 1280

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Subs,Montserrat-Bold,48,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,1,2,0,2,20,20,250

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # utf-8-sig ОБЯЗАТЕЛЕН для ffmpeg под Windows, чтобы работала кириллица
    with open(output_ass_path, "w", encoding="utf-8-sig") as f:
        f.write(ass_header)

        for sentence in sentences:
            start = seconds_to_ass_time(sentence[0]["begin"])
            end = seconds_to_ass_time(sentence[-1]["end"])

            karaoke_parts = []
            for w in sentence:
                duration_cs = round((w["end"] - w["begin"]) * 100)
                # Добавляем обязательный пробел в конце каждого слова!
                karaoke_parts.append(f"{{\\k{duration_cs}}}{w['word']} ")

            text = "".join(karaoke_parts).strip()
            f.write(f"Dialogue: 0,{start},{end},Subs,,0,0,0,,{text}\n")

    print(f"ASS с караоке-субтитрами сгенерирован: {output_ass_path}")

if __name__ == "__main__":
    fragments_to_ass("map_words.json", "subs.ass")
