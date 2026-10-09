"""SDXL style dataset, reproducible shot generation, and PNG-sequence assembly."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def style_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def prepare(args):
    manifest = Path(args.manifest).resolve()
    parent = manifest.parent
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Expected a nonempty JSON array of approved artwork.")
    dest = Path(args.out).resolve()
    style = style_config(args.style)
    records = []
    hashes = set()
    for i, entry in enumerate(entries):
        if entry.get("approved") is not True:
            raise ValueError(f"Image {i} has not been explicitly approved.")
        if entry.get("origin") not in ("author_original", "author_approved_refinement"):
            raise ValueError(f"Image {i} needs approved origin.")
        filename = Path(entry["file"])
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError("Use relative paths without traversal.")
        image = (parent / filename).resolve()
        if parent not in image.parents or not image.is_file() or image.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"Missing/unsafe image: {filename}")
        caption = entry.get("caption", "").strip()
        if len(caption) < 16:
            raise ValueError(f"Missing descriptive caption: {filename}")
        with Image.open(image) as im:
            im.verify()
        with Image.open(image) as im:
            if min(im.size) < 512:
                raise ValueError(f"Image is too small for SDXL: {filename}")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicated image: {filename}")
        hashes.add(digest)
        output_name = f"{i:04d}{image.suffix.lower()}"
        records.append((image, output_name, caption, digest, entry["origin"]))
    if dest == parent:
        raise ValueError("Use a separate dataset output folder.")
    dest.mkdir(parents=True, exist_ok=True)
    metadata, provenance = [], []
    for source, output_name, caption, digest, origin in records:
        shutil.copyfile(source, dest / output_name)
        metadata.append({"file_name": output_name, "text": f"{style['trigger']}, {caption}"})
        provenance.append({"file_name": output_name, "sha256": digest, "origin": origin, "source": str(source)})
    (dest / "metadata.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in metadata), encoding="utf-8")
    (dest / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Prepared {len(metadata)} approved training images: {dest}")


def shot(args):
    style = style_config(args.style)
    people = json.loads(Path(args.characters).read_text(encoding="utf-8"))["characters"]
    key = args.character.upper()
    if key not in people:
        raise ValueError(f"Unknown character {key}; use one of {list(people)}.")
    p = people[key]
    data = {
        "shot_id": args.shot_id, "character_id": key, "style_id": style["style_id"],
        "prompt": f"{style['positive_prefix']}, {p['name']}, {p['look']}, {p['build']}, {p['outfit']}, {args.scene}, single consistent character, no writing or logos",
        "negative_prompt": style["negative"], "seed": args.seed,
        "width": args.width, "height": args.height, "lora_weight": args.lora_weight,
        "reference_image": args.reference, "continuity": p["continuity"],
        "stage": "concept-keyframe"
    }
    raw = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(raw, encoding="utf-8")
    print(raw)


def train_command(args):
    trainer = Path(args.trainer).resolve()
    data = Path(args.dataset).resolve()
    if not trainer.is_file():
        raise ValueError("Provide the official train_text_to_image_lora_sdxl.py trainer checkout.")
    if not (data / "metadata.jsonl").is_file():
        raise ValueError("Dataset missing: first run prepare.")
    rows = [json.loads(x) for x in (data / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or any("text" not in row or not (data / row["file_name"]).is_file() for row in rows):
        raise ValueError("Invalid ImageFolder training set.")
    base = style_config(args.style)["base_model"]
    cmd = [
        "accelerate", "launch", str(trainer),
        "--pretrained_model_name_or_path", base,
        "--train_data_dir", str(data), "--output_dir", str(Path(args.out).resolve()),
        "--resolution", str(args.resolution), "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--learning_rate", str(args.lr), "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0", "--mixed_precision", "fp16",
        "--rank", str(args.rank), "--max_train_steps", str(args.steps),
        "--checkpointing_steps", str(args.checkpoint_steps),
        "--seed", str(args.seed), "--report_to", "tensorboard",
        "--gradient_checkpointing",
    ]
    print(shlex.join(cmd))
    if args.run:
        subprocess.run(cmd, check=True)


def render(args):
    spec = json.loads(Path(args.shot).read_text(encoding="utf-8"))
    lora = Path(args.lora).resolve()
    if not lora.is_file():
        raise ValueError("Supply a real trained LoRA .safetensors file.")
    if spec.get("reference_image"):
        raise ValueError("This text2image renderer cannot enforce requested identity references; use a compatible reference adapter.")
    try:
        import torch
        from diffusers import StableDiffusionXLPipeline
    except ImportError as exc:
        raise RuntimeError("Install CUDA PyTorch and requirements-gpu.txt.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for SDXL rendering.")
    base = style_config(args.style)["base_model"]
    pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, use_safetensors=True)
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
    pipe = pipe.to("cuda")
    pipe.fuse_lora(lora_scale=float(spec.get("lora_weight", 0.8)))
    generator = torch.Generator(device="cuda").manual_seed(int(spec["seed"]))
    output_image = pipe(
        prompt=spec["prompt"], negative_prompt=spec["negative_prompt"],
        width=int(spec["width"]), height=int(spec["height"]),
        generator=generator, num_inference_steps=args.steps, guidance_scale=args.guidance
    ).images[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(out)
    out.with_suffix(".json").write_text(json.dumps({
        "shot": spec, "base": base, "lora": str(lora), "steps": args.steps,
        "guidance": args.guidance
    }, indent=2), encoding="utf-8")
    print(f"Saved keyframe {out}; not a temporally consistent animation.")


def assemble(args):
    frames = Path(args.frames)
    if not frames.is_dir() or not list(frames.glob("*.png")):
        raise ValueError("No PNG frames found.")
    if not 1 <= args.fps <= 60:
        raise ValueError("FPS must be between 1 and 60.")
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg is not installed.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
        "-i", str(frames / "*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(out)
    ], check=True)
    print(f"Created video {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cemí animated production toolkit")
    sub = parser.add_subparsers(dest="action", required=True)
    default_style = str(ROOT / "configs/style.yaml")
    d = sub.add_parser("prepare")
    d.add_argument("--manifest", required=True)
    d.add_argument("--out", default="data/style_train")
    d.add_argument("--style", default=default_style)
    d.set_defaults(func=prepare)

    t = sub.add_parser("train-command")
    t.add_argument("--trainer", required=True)
    t.add_argument("--dataset", default="data/style_train")
    t.add_argument("--out", default="models/cemi-style-v1")
    t.add_argument("--style", default=default_style)
    t.add_argument("--resolution", type=int, default=1024)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--grad-accum", type=int, default=4)
    t.add_argument("--checkpoint-steps", type=int, default=250)
    t.add_argument("--seed", type=int, default=20261009)
    t.add_argument("--run", action="store_true")
    t.set_defaults(func=train_command)

    s = sub.add_parser("shot")
    s.add_argument("--character", required=True)
    s.add_argument("--scene", required=True)
    s.add_argument("--shot-id", required=True)
    s.add_argument("--style", default=default_style)
    s.add_argument("--characters", default=str(ROOT / "configs/characters.json"))
    s.add_argument("--reference")
    s.add_argument("--seed", type=int, default=20261009)
    s.add_argument("--width", type=int, default=1024)
    s.add_argument("--height", type=int, default=1024)
    s.add_argument("--lora-weight", type=float, default=0.8)
    s.add_argument("--out")
    s.set_defaults(func=shot)

    r = sub.add_parser("render")
    r.add_argument("--shot", required=True)
    r.add_argument("--lora", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--style", default=default_style)
    r.add_argument("--steps", type=int, default=30)
    r.add_argument("--guidance", type=float, default=6.0)
    r.set_defaults(func=render)

    a = sub.add_parser("assemble")
    a.add_argument("--frames", required=True)
    a.add_argument("--fps", type=int, default=12)
    a.add_argument("--out", required=True)
    a.set_defaults(func=assemble)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
