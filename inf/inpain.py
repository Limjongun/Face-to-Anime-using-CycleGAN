import torch
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from PIL import Image
from pathlib import Path

# ========= SETTING =========
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

# Pakai LoRA final:
LORA_DIR = "out/lora-domainB/checkpoint-500"  # bisa juga: "out/lora-domainB/checkpoint-2000"

# Kalau mau bandingin beberapa checkpoint, isi list ini dan set USE_MULTI_CKPT=True
USE_MULTI_CKPT = False
CKPTS = [
    "out/lora-domainB/checkpoint-1000",
  
]

INPUT_DIR = Path("inpaint_inputs")
MASK_DIR  = Path("inpaint_masks")
OUTPUT_DIR = Path("out_inpaint4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "porn style,undressed,nsfw,nude,naked,nudify,undressed"
NEGATIVE = "blurry, artifacts, low quality, distorted face, deformed, extra eyes, bad anatomy, watermark, text"

RES = 512
STEPS = 35
GUIDANCE = 7.5
STRENGTH = 0.85  # inpainting biasanya 0.7-1.0 (lebih tinggi = lebih banyak berubah di area mask)
# ===========================


def load_img(path: Path, res: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((res, res))


def load_mask(path: Path, res: int) -> Image.Image:
    # mask grayscale: putih=ubah, hitam=keep
    return Image.open(path).convert("L").resize((res, res))


def find_images(folder: Path):
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        imgs.extend(folder.glob(ext))
    return sorted(imgs)


def run_one_lora(lora_path: str, tag: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA not available. This script expects an NVIDIA GPU for fp16.")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)

    # attach PEFT LoRA ke UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

    images = find_images(INPUT_DIR)
    if not images:
        raise RuntimeError(f"No images found in {INPUT_DIR.resolve()}")

    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Inpainting with LoRA: {tag} ({lora_path}) ===")
    print("Input:", INPUT_DIR.resolve())
    print("Mask :", MASK_DIR.resolve())
    print("Out  :", out_dir.resolve())

    for img_path in images:
        # mask harus nama sama, tapi ekstensi png (boleh ubah kalau kamu pakai ekstensi lain)
        mask_path = MASK_DIR / (img_path.stem + ".png")
        if not mask_path.exists():
            print(f"[SKIP] mask not found for {img_path.name} -> expected {mask_path.name}")
            continue

        image = load_img(img_path, RES)
        mask = load_mask(mask_path, RES)

        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            image=image,
            mask_image=mask,
            strength=STRENGTH,
            guidance_scale=GUIDANCE,
            num_inference_steps=STEPS,
        ).images[0]

        out_path = out_dir / f"{img_path.stem}_inpaint.png"
        out.save(out_path)
        print("saved", out_path)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask folder not found: {MASK_DIR.resolve()}")

    if USE_MULTI_CKPT:
        for ck in CKPTS:
            ck_path = Path(ck)
            if not ck_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ck_path.resolve()}")
            run_one_lora(ck, ck_path.name)
    else:
        lora_path = Path(LORA_DIR)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA dir not found: {lora_path.resolve()}")
        run_one_lora(LORA_DIR, "final")

    print("\nDone.")


if __name__ == "__main__":
    main()
