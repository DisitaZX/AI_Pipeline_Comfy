import json
import re
import random


class SRTSceneTimeline:
    def parse_time(self, t):
        h, m, s = t.split(":")
        s, ms = s.split(",")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def parse_srt(self, srt):
        pattern = re.findall(
            r'(\d+)\s+([\d:,]+)\s+-->\s+([\d:,]+)\s+([\s\S]*?)(?=\n\d+\n|\Z)',
            srt
        )

        segments = []

        for _, start, end, text in pattern:
            text = text.replace("\n", " ").strip()
            segments.append({
                "start": self.parse_time(start),
                "end": self.parse_time(end),
                "text": text
            })

        return segments

    def normalize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def find_segments(self, scene, segments):

        scene_norm = self.normalize(scene)

        found = []

        for seg in segments:
            if scene_norm in self.normalize(seg["text"]):
                found.append(seg)

        if found:
            return found

        # fallback fuzzy
        for seg in segments:
            if any(word in self.normalize(seg["text"]) for word in scene_norm.split()):
                found.append(seg)

        return found

    def build(self, srt_text, target_texts, fps=16, max_overlap_sec=0.0, seed=0):

        random.seed(seed)

        segments = self.parse_srt(srt_text)

        scenes = [s.strip() for s in target_texts.split("\n") if s.strip()]

        result = []
        timeline = []

        for scene in scenes:

            segs = self.find_segments(scene, segments)

            if not segs:
                continue

            start = segs[0]["start"]
            end = segs[-1]["end"]

            timeline.append({
                "scene_prompt": scene,
                "start": start,
                "end": end
            })

        # overlap
        for i in range(len(timeline)):

            overlap_start = random.uniform(0, max_overlap_sec)
            overlap_end = random.uniform(0, max_overlap_sec)

            timeline[i]["start"] = max(0, timeline[i]["start"] - overlap_start)
            timeline[i]["end"] = timeline[i]["end"] + overlap_end

        # fix overlaps between scenes
        for i in range(1, len(timeline)):

            if timeline[i]["start"] < timeline[i-1]["end"]:
                timeline[i]["start"] = timeline[i-1]["end"]

        audio_length = timeline[-1]["end"]

        total_frames = round(audio_length * fps)

        used_frames = 0

        for i, scene in enumerate(timeline):

            duration = scene["end"] - scene["start"]

            if i < len(timeline) - 1:
                frames = round(duration * fps)
                used_frames += frames
            else:
                frames = total_frames - used_frames
                duration = frames / fps

            result.append({
                "scene_prompt": scene["scene_prompt"],
                "audio_start": scene["start"],
                "video_duration": duration,
                "frames": frames
            })

        video_length = total_frames / fps

        output = {
            "scenes": result,
            "debug": {
                "audio_length_sec": audio_length,
                "video_length_sec": video_length,
                "frames_total": total_frames
            }
        }

        return output
        #return json.dumps(output, ensure_ascii=False, indent=2)
