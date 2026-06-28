import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from pathlib import Path

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = "out/lora-domainB/checkpoint-500"      # atau "out/lora-domainB/checkpoint-2000"
OUTDIR = Path("out_gen_B")
OUTDIR.mkdir(parents=True, exist_ok=True)

PROMPT = "domain B style"  # ganti sesuai style kamu
NEGATIVE = "blurry, low quality, artifacts, distorted"
STEPS = 30
GUIDANCE = 7.5
SEED = 1234
N = 8  # jumlah gambar

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # attach PEFT LoRA ke UNet
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)

    generator = torch.Generator(device=device).manual_seed(SEED)

    for i in range(N):
        img = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
        ).images[0]

        out_path = OUTDIR / f"gen_B_{i:03d}.png"
        img.save(out_path)
        print("saved:", out_path)

    print("Done. Output folder:", OUTDIR.resolve())

if __name__ == "__main__":
    main()
