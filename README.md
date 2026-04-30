Pipeline for extended video generation. Images + Videos + TTS. 
Using .safetensors weights from HuggingFace models.
Wan2.2 FP8 + 4 steps lora 260k, CozyVoice3, QwenImageEdit2511 FP8 4step from lightx2v, ZImage-Turbo.
First, a document is created that must be placed in a main.json file with a specific structure. 
Next, a full local generation of base variables (images) takes place. Their prompts are generated and modified, and they are combined using QwenImageEdit 2511. 
Prompt generation occurs locally on Gemma4:e4b. Wan 2.2 + rife_vulcan (frame interpolation) is responsible for generating the final video. 
Also required for installation: ffmpeg and aeneas. 
All models are loaded into video memory and RAM in the order they are processed. 
This places virtually no load on disk space. Recommended components: RTX 3080 Ti, 32 GB of RAM. 
