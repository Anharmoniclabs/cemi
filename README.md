# Heralds of the Cemí — Animated Production Pipeline

Author-directed SDXL style LoRA and 2D/2.5D animation project for **the author's own drawing style**, not stock anime.

**Locked look:** hand-drawn asymmetric, geometric expressive faces; confident cleaned linework; lightly grouped curls; restrained anime eye polish; cel shadows. **Option B: Moody Urban Dusk**: navy, steel-blue and warm amber. **Avoid gratuitous writing, slogans or words on walls** — lettering belongs in post-production, only when story-required.

Pipeline: original hand art -> human-approved finished illustrated references -> captioned and provenance-tracked image dataset -> SDXL style LoRA -> reference-locked character keyframes -> animation layers/rigged movement -> compositing/sound -> video QA.

### Starter kit
The complete runnable Python toolkit is supplied in the ChatGPT conversation as a downloadable ZIP with:
- `cemi_pipeline` CLI: prepare, train-command, shot, render, assemble
- `configs/style.yaml` and character identity specifications
- dataset provenance and approval checks, tests, and detailed production documentation.

Extract it into a repository checkout. Required quick-start:

    pip install -r requirements.txt
    python -m cemi_pipeline --help
    python -m cemi_pipeline shot --character JC --shot-id ep01-sc01-001 --scene 'at his Bronx desk at dusk' --out outputs/jc-shot.json

Actual SDXL training requires approved training images and a CUDA machine; the repo doesn't include trained weights or a finished episode. Start with docs/PIPELINE.md and docs/STYLE_BIBLE.md.

## Copyright / production

Keep artist artwork and trained weights private until explicitly approved for public release. The first character/story bible provided by the author is authoritative over mistakes in generated concept sheets. Treat LoRA style transfer and character identity as separate systems.
