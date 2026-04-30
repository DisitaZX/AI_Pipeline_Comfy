import json

def seconds_to_srt_time(sec: float) -> str:
    """Конвертирует секунды в формат SRT hh:mm:ss,ms"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def fragments_to_srt(fragments_json: str) -> str:
    data = json.loads(fragments_json)
    fragments = data.get("fragments", [])
    srt_entries = []

    for idx, frag in enumerate(fragments, start=1):
        start = float(frag["begin"])
        end = float(frag["end"])
        text = " ".join(frag["lines"]).replace("\\n", "\n")
        srt_entry = f"{idx}\n{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n{text}\n"
        srt_entries.append(srt_entry)

    return "\n".join(srt_entries)

# Пример использования
if __name__ == "__main__":
    with open("map.json", "r", encoding="utf-8") as f:
        fragments_json = f.read()

    srt_content = fragments_to_srt(fragments_json)

    with open("srt/srt_encode.srt", "w", encoding="utf-8") as f:
        f.write(srt_content)

    print("SRT сгенерирован: output.srt")