# 🎬 Pipeline for extended anime video generation

**Core model: MiniMax H3 (reference-to-video, ref2va)** — video + native audio + lip-sync in a single pass.
**Multishot clips up to 15 seconds**: several shots with cuts inside one generation (`[Shot N] At MM:SS.mmm`).

---

## 📄 Step 1 — JSON plan

A story plan is written into `main.json` (exact schema and writing rules: `GROK_Prompt_MINIMAX.txt`):

- `base_prompts` — canonical entity descriptions (character / location / object), max 10.
- `scenes[]` — multishot scene scripts:
  - `video_prompt` with `[Shot 1]` / `[Shot N] At MM:SS.mmm` cut markers and `[base_name]` entity markers (≤ 5 refs per clip);
  - `duration_s` 4–15 (recommended 8–15);
  - `dialogue` — speech / thought lines, each anchored to its shot (`shot` field), optional `speaker_ref` (voice bound to a `<Picture N>` reference) and `delivery`.

## 🖌️ Step 2 — Base entity refs

One canonical 720p reference image per entity (Z-Anime text-to-image): characters 720×1280, locations/objects 1280×720. Saved to `output/base_image_gen/`.

## 🎥 Step 3 — Video generation (MiniMax H3)

`MiniMaxH3ReferenceToVideo` builds each clip directly from up to 5 reference images (`<Picture N>` slots) + the multishot prompt. Dialogue is injected per shot in the model's native format (`<d>[Russian] ...</d>`, stable speaker IDs `(S1)`, `(S2)`; thought = off-screen voiceover with closed lips). Sound is generated natively and embedded in the clip — **no separate TTS track**. 640×640 @ 24 fps.

### 🎙️ Voice references (optional, recommended)

A character's voice timbre is locked across all clips via the node's `ref_audios` inputs (`<Audio N>` voice-timbre references, limit 3 per clip; audio always rides together with image refs).

1. Cut a clean 2–15 s speech sample per character (wav/mp3/ogg/m4a/flac), e.g. from a good take:
   `ffmpeg -i "clip.mp4" -ss 2.0 -to 9.0 -vn -ac 1 -ar 32000 voices/lin_jie.wav`
2. Map samples in `voice_refs.json` next to `generate_scenes_minimax.py` (survives `main.json` regeneration):
   `{ "lin_jie": "C:/Users/Loopy/Desktop/voices/lin_jie.wav" }`
   (or set `voice_ref` on a `base_prompts[]` character for a plan-local override)
3. Scenes whose dialogue uses `speaker_ref` automatically wire the sample (`VHS_LoadAudio` → `ref_audios.ref_audio_N`) and address it in the prompt as `<Audio N>`.

## 📦 Step 4 — Final assembly

ffmpeg concat (with sound) → optional background music at −25 dB → optional shot-timed ASS subtitles from `dialogue` → cover from the first frame of the first clip.

---

## 🔧 Required

- `ffmpeg`
- ComfyUI (Windows portable) with MiniMax H3 nodes and models:
  - `minimax_h3_ref2va_int8_convrot.safetensors`
  - `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
  - `minimax_h3_video_vae_fp16.safetensors`
  - `minimax_h3_audio_vae_fp32.safetensors`

Run: put `main.json` next to `generate_scenes_minimax.py`, then `python generate_scenes_minimax.py` (ComfyUI starts/stops automatically via ComfyManager).

## 💾 Hardware & performance

| Component | Specification  |
|-----------|----------------|
| **GPU**   | RTX 3080 Ti    |
| **RAM**   | 32 GB          |

> 🚀 *Full local pipeline — from prompts to final video*
