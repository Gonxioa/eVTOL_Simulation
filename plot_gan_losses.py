from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = Path(
    os.getenv(
        "CLASSIFIER_SAVE_DIR_OVERRIDE",
        os.getenv(
            "OUTPUT_DIR_OVERRIDE",
            PROJECT_ROOT / "plots" / "hex" / "classification" / "formal_combined",
        ),
    )
).expanduser().resolve()
SAVE_PATH = EXPERIMENT_ROOT / "gan_loss_curves.png"


def moving_average(values, window=25):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window) / window
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main():
    loss_files = sorted(EXPERIMENT_ROOT.glob("folds/fold_*/wgan_*/models/gan_losses.npz"))
    if not loss_files:
        raise FileNotFoundError(f"No gan_losses.npz files found under {EXPERIMENT_ROOT}")

    loss_g_all = []
    loss_g_adv_all = []
    loss_d_all = []
    loss_phys_all = []
    for path in loss_files:
        data = np.load(path)
        loss_g_all.append(data["loss_G"])
        loss_d_all.append(data["loss_D"])
        if "loss_G_adv" in data and len(data["loss_G_adv"]) > 0:
            loss_g_adv_all.append(data["loss_G_adv"])
        if "loss_phys" in data and len(data["loss_phys"]) > 0:
            loss_phys_all.append(data["loss_phys"])

    groups = [
        ("Generator Total Loss", loss_g_all),
    ]
    if len(loss_g_adv_all) == len(loss_files):
        groups.append(("Generator Adversarial Loss", loss_g_adv_all))
    groups.append(("Discriminator Loss", loss_d_all))
    if len(loss_phys_all) == len(loss_files):
        groups.append(("Physics Residual Loss", loss_phys_all))

    min_len = min(len(values) for _, arrays in groups for values in arrays)
    epochs = np.arange(1, min_len + 1)

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 3.5 * len(groups)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("Fold-local WGAN-GP Training Loss Curves", fontsize=14, fontweight="bold")

    for ax, (title, arrays) in zip(axes, groups):
        stacked = np.array([values[:min_len] for values in arrays])
        for path, values in zip(loss_files, stacked):
            label = f"{path.parents[2].name}/{path.parents[1].name}"
            ax.plot(epochs, moving_average(values), alpha=0.35, lw=1, label=label)
        ax.plot(epochs, moving_average(stacked.mean(axis=0)), color="black", lw=2.2, label="Mean")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(ncol=3, fontsize=8)

    axes[-1].set_xlabel("Epoch")

    plt.tight_layout()
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(SAVE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()
