import torch
from diffusers import StableDiffusionImg2ImgPipeline
from peft import PeftModel
from PIL import Image
from pathlib import Path

# ====== SETTING ======
BASE_MODEL = "runwayml/stable-diffusion-v1-5"

CKPTS = [
    "out/lora-domainB/checkpoint-500",
    "out/lora-domainB/checkpoint-1000",
    "out/lora-domainB/checkpoint-1500",
    "out/lora-domainB/checkpoint-2000",
    "out/lora-domainB/checkpoint-2500",
    "out/lora-domainB/checkpoint-3000",

]

INPUT_DIR = Path("aaa")      # taruh foto domain A di sini
OUTPUT_DIR = Path("out_compare")  # hasil akan ditaruh di sini
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "porn style,undressed,nsfw,nude,naked,nudify,undressed"
NEGATIVE = "blurry, artifacts, low quality, distorted"
STRENGTH = 0.7  
GUIDANCE = 7
STEPS = 35
RES = 512  # resize inference ke 512x512
# =====================


def load_image(p: Path, res: int) -> Image.Image:
    img = Image.open(p).convert("RGB")
    return img.resize((res, res))


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR.resolve()}")

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images.extend(INPUT_DIR.glob(ext))
    images = sorted(images)

    if not images:
        raise RuntimeError(f"No images found in {INPUT_DIR.resolve()} (png/jpg/jpeg/webp)")

    print("Found", len(images), "images in", INPUT_DIR.resolve())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA not available. This script expects an NVIDIA GPU for fp16.")

    for ck in CKPTS:
        ck_path = Path(ck)
        if not ck_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ck_path.resolve()}")

        ck_name = ck_path.name  # checkpoint-500, etc.
        out_ck_dir = OUTPUT_DIR / ck_name
        out_ck_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Loading pipeline + LoRA: {ck_name} ===")

        # Recreate pipeline each checkpoint (lebih aman & simple)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)

        pipe.unet = PeftModel.from_pretrained(pipe.unet, ck)

        for img_path in images:
            img = load_image(img_path, RES)

            out = pipe(
                prompt=PROMPT,
                negative_prompt=NEGATIVE,
                image=img,
                strength=STRENGTH,
                guidance_scale=GUIDANCE,
                num_inference_steps=STEPS,
            ).images[0]

            out_name = img_path.stem + "_AtoB.png"
            out_path = out_ck_dir / out_name
            out.save(out_path)
            print("saved", out_path)

    print("\nDone. Outputs in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
