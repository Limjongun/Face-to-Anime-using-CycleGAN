# ui_cyclegan_flet_hardcoded.py
import os
import threading
from pathlib import Path
from typing import Tuple

import flet as ft
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.utils import save_image
from contextlib import nullcontext


# =========================
# HARDCODED ARGUMENTS (sesuai command kamu)
# =========================
CKPT_PATH = r".\anime12\ckpt\cyclegan_epoch270.pt"
IN_DIR    = r".\test3"     # (UI ini tidak batch; hanya referensi)
OUT_DIR   = r".\AALAM"
DIRECTION = "A2B"
HEIGHT    = 384
WIDTH     = 256
OVERLAP   = 64
N_BLOCKS  = 7
DEVICE    = "cuda"         # fallback otomatis ke cpu kalau cuda tidak ada

TILE      = "0"            # kalau mau tile, ubah misal "512" atau "384x256"
KEEP_ASPECT = False        # kalau True: resize preserve aspect lalu center-crop
CHANNELS_LAST = False      # GPU only
AMP = False                # GPU only


# =========================
# Model (must match training)
# =========================
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_blocks=6, ngf=64):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_c, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        c = ngf

        for _ in range(2):
            layers += [
                nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(True),
            ]
            c *= 2

        for _ in range(n_blocks):
            layers += [ResnetBlock(c)]

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(c, c // 2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(True),
            ]
            c //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, out_c, 7),
            nn.Tanh()
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =========================
# Helpers (tiling + transform)
# =========================
def denorm01(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

_BLEND_CACHE = {}

def get_blend_window(h: int, w: int, device, dtype):
    key = (h, w, str(device), str(dtype))
    if key in _BLEND_CACHE:
        return _BLEND_CACHE[key]
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    win2d = torch.outer(wy, wx).unsqueeze(0).unsqueeze(0).clamp_min(1e-6)
    _BLEND_CACHE[key] = win2d
    return win2d

def parse_tile(tile_str: str) -> Tuple[int, int]:
    if not tile_str or tile_str.strip() == "0":
        return 0, 0
    s = tile_str.lower().strip()
    if "x" in s:
        a, b = s.split("x", 1)
        return int(a), int(b)
    v = int(s)
    return v, v

def run_tiled(G, x, tile_h: int, tile_w: int, overlap: int, use_amp: bool):
    assert x.dim() == 4 and x.size(0) == 1
    _, _, H, W = x.shape

    tile_h = min(int(tile_h), H)
    tile_w = min(int(tile_w), W)
    overlap = max(0, min(int(overlap), min(tile_h, tile_w) - 1))

    stride_y = tile_h - overlap
    stride_x = tile_w - overlap
    if stride_y <= 0: stride_y = tile_h
    if stride_x <= 0: stride_x = tile_w

    out = torch.zeros_like(x)
    wgt = torch.zeros((1, 1, H, W), device=x.device, dtype=x.dtype)

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (use_amp and x.is_cuda) else nullcontext()

    with torch.no_grad(), amp_ctx:
        for y0 in range(0, H, stride_y):
            for x0 in range(0, W, stride_x):
                y1 = min(y0 + tile_h, H)
                x1 = min(x0 + tile_w, W)
                y0b = max(y1 - tile_h, 0)
                x0b = max(x1 - tile_w, 0)

                patch = x[:, :, y0b:y1, x0b:x1]
                pred = G(patch)

                ph, pw = pred.shape[-2], pred.shape[-1]
                win = get_blend_window(ph, pw, device=pred.device, dtype=pred.dtype)

                out[:, :, y0b:y1, x0b:x1] += pred * win
                wgt[:, :, y0b:y1, x0b:x1] += win

    return out / wgt.clamp_min(1e-6)

def build_transform(h: int, w: int, keep_aspect: bool):
    if keep_aspect:
        def resize_preserve(img: Image.Image):
            W0, H0 = img.size
            scale = max(h / H0, w / W0)
            new_h = int(round(H0 * scale))
            new_w = int(round(W0 * scale))
            return img.resize((new_w, new_h), resample=Image.BICUBIC)

        class ResizePreserve:
            def __call__(self, img):
                return resize_preserve(img)

        return T.Compose([
            ResizePreserve(),
            T.CenterCrop((h, w)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        return T.Compose([
            T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])


# =========================
# Load model once (cache)
# =========================
_G = None

def get_model(device: torch.device):
    global _G
    if _G is not None:
        return _G

    ckpt = torch.load(CKPT_PATH, map_location=device)
    key = "G_AB" if DIRECTION == "A2B" else "G_BA"
    if key not in ckpt:
        raise KeyError(f"Key '{key}' tidak ada di checkpoint. Keys: {list(ckpt.keys())}")

    G = ResnetGenerator(n_blocks=N_BLOCKS).to(device)
    G.load_state_dict(ckpt[key], strict=True)
    G.eval()

    if CHANNELS_LAST and device.type == "cuda":
        G = G.to(memory_format=torch.channels_last)

    _G = G
    return _G


# =========================
# Flet UI
# =========================
def main(page: ft.Page):
    page.title = "CycleGAN UI (Hardcoded Args)"
    page.window_width = 1100
    page.window_height = 720
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 16

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = ft.Text("", color=ft.Colors.BLUE_700)
    info = ft.Text(
        f"ckpt={CKPT_PATH}\n"
        f"out_dir={OUT_DIR}\n"
        f"direction={DIRECTION} | size={HEIGHT}x{WIDTH} | overlap={OVERLAP} | n_blocks={N_BLOCKS} | device={DEVICE}\n"
        f"tile={TILE} | keep_aspect={KEEP_ASPECT} | amp={AMP} | channels_last={CHANNELS_LAST}",
        size=12,
        color=ft.Colors.GREY_700
    )

    preview_in = ft.Image(width=420, height=420, fit=ft.ImageFit.CONTAIN)
    preview_out = ft.Image(width=420, height=420, fit=ft.ImageFit.CONTAIN)

    state = {"image_path": None}

    picker = ft.FilePicker()
    page.overlay.append(picker)

    def set_status(msg: str, ok: bool = True):
        status.value = msg
        status.color = ft.Colors.GREEN_700 if ok else ft.Colors.RED_700
        page.update()

    def on_pick(e: ft.FilePickerResultEvent):
        if not e.files:
            return
        p = e.files[0].path
        state["image_path"] = p
        preview_in.src = p
        preview_out.src = None
        set_status("Gambar dipilih. Klik Inferensi.")
        page.update()

    picker.on_result = on_pick

    def worker_infer():
        try:
            if not state["image_path"]:
                set_status("Pilih gambar dulu.", ok=False)
                return

            device_name = DEVICE
            if device_name == "cuda" and not torch.cuda.is_available():
                device_name = "cpu"
            device = torch.device(device_name)

            use_amp = bool(AMP and device.type == "cuda")
            tile_h, tile_w = parse_tile(TILE)

            set_status("Load model…")
            G = get_model(device)

            tfm = build_transform(HEIGHT, WIDTH, keep_aspect=KEEP_ASPECT)
            img = Image.open(state["image_path"]).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            if CHANNELS_LAST and device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)

            set_status("Inferensi…")

            if tile_h > 0 and tile_w > 0:
                y = run_tiled(G, x, tile_h=tile_h, tile_w=tile_w, overlap=OVERLAP, use_amp=use_amp)
            else:
                with torch.no_grad():
                    if use_amp:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            y = G(x)
                    else:
                        y = G(x)

            y01 = denorm01(y)

            base = Path(state["image_path"]).stem
            out_path = out_dir / f"{base}_{DIRECTION}_{HEIGHT}x{WIDTH}.png"
            save_image(y01, str(out_path))

            preview_out.src = str(out_path)
            set_status(f"Selesai ✅ Output: {out_path}")
            page.update()

        except Exception as ex:
            set_status(f"Error: {repr(ex)}", ok=False)

    def on_click_infer(e):
        threading.Thread(target=worker_infer, daemon=True).start()

    btn_pick = ft.ElevatedButton(
        "Upload Foto",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda e: picker.pick_files(
            allow_multiple=False,
            allowed_extensions=["png", "jpg", "jpeg", "webp", "bmp"],
        ),
    )

    btn_infer = ft.FilledButton(
        "Inferensi",
        icon=ft.Icons.PLAY_ARROW,
        on_click=on_click_infer,
    )

    left = ft.Container(
        padding=16,
        border_radius=16,
        bgcolor="#FFFFFF",
        border=ft.border.all(1, "#E5E7EB"),
        width=430,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text("Kontrol", size=18, weight=ft.FontWeight.BOLD),
                btn_pick,
                btn_infer,
                status,
                ft.Divider(),
                ft.Text("Hardcoded args", weight=ft.FontWeight.BOLD),
                info,
            ],
        ),
    )

    right = ft.Container(
        padding=16,
        border_radius=16,
        bgcolor="#FFFFFF",
        border=ft.border.all(1, "#E5E7EB"),
        expand=True,
        content=ft.Column(
            spacing=10,
            controls=[
                ft.Text("Preview", size=18, weight=ft.FontWeight.BOLD),
                ft.Row(
                    spacing=12,
                    controls=[
                        ft.Column([ft.Text("Input"), preview_in], spacing=6),
                        ft.Column([ft.Text("Output"), preview_out], spacing=6),
                    ],
                ),
            ],
        ),
    )

    page.add(ft.Row([left, right], spacing=14, expand=True))


if __name__ == "__main__":
    ft.app(target=main)
