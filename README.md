# 🎬 Pipeline for extended video generation

**Images + Videos + TTS**

Using `.safetensors` weights from HuggingFace models.  
🔧 **Wan2.2 FP8** + 4 steps lora 260k  
🎙️ **CozyVoice3**  
🖼️ **QwenImageEdit2511 FP8** 4step from lightx2v  
⚡ **ZImage-Turbo**

---

## 📄 Step 1 — JSON structure

First, a document is created that must be placed in a `main.json` file with a specific structure.

---

## 🖌️ Step 2 — Base variables (images)

Next, a full local generation of base variables (images) takes place.  
Their prompts are generated and modified, and they are combined using **QwenImageEdit 2511**.

---

## 🧠 Step 3 — Prompt generation

Prompt generation occurs locally on **Gemma4:e4b**.

---

## 🎥 Step 4 — Final video generation

**Wan 2.2 + rife_vulcan** (frame interpolation) is responsible for generating the final video.

---

## 📦 Required for installation

- `ffmpeg`  
- `aeneas`

---

## 💾 Hardware & performance

- All models are loaded into video memory and RAM **in the order they are processed**.  
- This places virtually **no load on disk space**.  

### ✅ Recommended components

| Component       | Specification      |
|----------------|--------------------|
| **GPU**        | RTX 3080 Ti        |
| **RAM**        | 32 GB              |

---

> 🚀 *Full local pipeline — from prompts to final video*