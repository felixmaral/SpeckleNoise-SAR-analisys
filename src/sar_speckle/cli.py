from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .simulation import apply_speckle_intensity, coherent_multilook_intensity
from .metrics import estimate_enl


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _save_gray(path: Path, arr: np.ndarray) -> None:
    arr = np.clip(arr, 0, 1)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def _synthetic(H: int = 256, W: int = 256) -> np.ndarray:
    img = np.zeros((H, W), np.float32)
    img[:H // 2, :W // 2] = 0.3
    img[:H // 2, W // 2 :] = 0.6
    img[H // 2 :, :] = 1.0
    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Inject SAR-like speckle into an intensity image.")
    p.add_argument("--input", type=str, default=None, help="Path to grayscale image (optional)")
    p.add_argument("--L", type=float, required=True, help="Equivalent number of looks")
    p.add_argument("--model", choices=["gamma", "coherent"], default="gamma")
    p.add_argument("--correlate-sigma", type=float, default=0.0, help="Gaussian sigma (pixels)")
    p.add_argument("--outdir", type=str, default="outputs/outputs/")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.input:
        I = _load_gray(Path(args.input))
    else:
        I = _synthetic()

    if args.model == "gamma":
        I_noisy = apply_speckle_intensity(I, L=args.L, correlate_sigma=args.correlate_sigma)
    else:
        I_noisy = coherent_multilook_intensity(
            I, L=int(round(args.L)), correlate_sigma=args.correlate_sigma
        )

    # Save outputs
    _save_gray(outdir / "clean.png", I)
    _save_gray(outdir / f"noisy_{args.model}_L{args.L}.png", I_noisy)

    # Simple ENL estimate on top-left 80x80 region
    patch = I_noisy[:80, :80]
    enl = estimate_enl(patch)
    with open(outdir / "metrics.txt", "w") as f:
        f.write(f"ENL_est ≈ {enl:.2f}\n")

    # Optional quick-look figure (no specific style/colors)
    plt.figure()
    plt.title(f"Noisy ({args.model}), L={args.L}")
    plt.imshow(I_noisy, cmap="gray")
    plt.axis("off")
    plt.savefig(outdir / "preview.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
