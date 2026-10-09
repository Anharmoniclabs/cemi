# SDXL style fine-tuning -> cartoon animation

## End-to-end structure

Human original drawings -> hand-approved colored production samples -> image-level captions/provenance -> SDXL style LoRA -> character references/pose conditioning -> shot manifest -> approved keyframes -> rigged 2.5D motion -> compositor, dialog, sound -> tested scene -> final episode.

## Training dataset

Start with roughly 30–80 *distinct* excellent author-approved final illustrations; it is a working experiment size, not a guarantee. Mix subjects and environments. Don't mix low-res mood boards full of lettering into the primary style dataset. For each file record image filename, artist approval, original-vs-approved-refinement origin, and accurate caption. Prepare a Hugging Face ImageFolder folder with metadata.jsonl, where every row contains file_name and text caption beginning with cemistyle. Hold out 10–20% for QA, chosen BEFORE data preparation.

Use the official Diffusers SDXL LoRA example from a recorded commit, not an unverified remote script:
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora_sdxl.py

Training starting settings: pretrained model stabilityai/stable-diffusion-xl-base-1.0, 1024px, rank 16, LR 1e-4, batch 1, gradient accumulation 4, around 1000–1500 steps. Record checkpoint and validation comparisons. A CUDA GPU with about 24 GB VRAM is considerably more comfortable than the 11 GB laptop described for local preproduction; lower memory needs extra configuration beyond this baseline. Respect model license.

## Separate style from character identity

One style LoRA does **not** make JC retain his face or Manny retain his mass. For consistent casts, approve individual clean front, profile, back, 3/4 and expression masters; use these with tested SDXL-compatible reference conditioning (e.g. IP-Adapter) and pose/depth guidance (e.g. ControlNet). Independently verify the model family compatibility, license and output. A private per-character LoRA can be tested later after adequate identity references are curated.

## Shot manifest and animation

Every shot must carry a character ID, script scene, seed, prompt, negative, grade, model/LoRA version and approved reference pointers. Generate **keyframes** for shot boundaries and gestures, not randomly independently regenerated video frames.

For actual motion, prefer layered cartoon cutouts/Grease Pencil or equivalent 2D rig: separate face, eyes, mouth, hair, torso, forearms, hands and legs; animate physical poses at 12 fps or on twos, then export 24 fps as desired. SDXL img2video is optional for short experimental clips only, with temporal consistency review. Static slideshow panels are NOT an animated scene.

FFmpeg can combine sequentially named PNG frames into an MP4 after the frames have actually been animated. Sync voice, Foley and licensed score separately. Validate gaze, proportions, outfit, hands, props, text artifacts, lip-sync and frame flicker across the shot.

**Status:** pipeline scaffold and example tooling only. No LoRA weights trained, GPU job run or cartoon footage generated in this commit.
