"""Fold-local conditional WGAN-GP for the hexarotor time-series dataset.

The generator ends in Tanh, therefore its real training data must be scaled to
[-1, 1] with a fold-local MinMaxScaler.  Classifier standardization is a
separate concern handled by classify.py.
"""

from pathlib import Path
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from hex_physical_characteristics import HEX_PHYSICAL_CHARACTERISTICS
from scenario_config import effectiveness_table


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_env(name, default):
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


DATA_DIR = _path_env(
    "DATA_DIR_OVERRIDE", PROJECT_ROOT / "plots" / "hex" / "data_process" / "combined"
)
SAVE_DIR = _path_env("WGAN_SAVE_DIR_OVERRIDE", PROJECT_ROOT / "plots" / "hex" / "wgan")

SEQ_LEN = 100
N_FEATURES = 18
N_CLASSES = 19
NOISE_DIM = int(os.getenv("GAN_NOISE_DIM", "64"))
HIDDEN_DIM = int(os.getenv("GAN_HIDDEN_DIM", "128"))
N_LAYERS = int(os.getenv("GAN_LAYERS", "2"))
EMBED_DIM = int(os.getenv("GAN_EMBED_DIM", "16"))
BATCH_SIZE = max(1, int(os.getenv("GAN_BATCH_SIZE", "64")))
GENERATE_BATCH_SIZE = max(1, int(os.getenv("GAN_GENERATE_BATCH_SIZE", str(BATCH_SIZE))))
LR_G = float(os.getenv("GAN_LR_G", "0.0001"))
LR_D = float(os.getenv("GAN_LR_D", "0.0001"))
N_EPOCHS = max(1, int(os.getenv("GAN_EPOCHS", "600")))
N_CRITIC = max(1, int(os.getenv("GAN_N_CRITIC", "5")))
LAMBDA_GP = float(os.getenv("GAN_LAMBDA_GP", "10"))
LAMBDA_PHYS = float(os.getenv("GAN_LAMBDA_PHYS", "0"))
PHYS_EPSILON = max(0.0, float(os.getenv("GAN_PHYS_EPSILON", "0.15")))
SAVE_EVERY = max(0, int(os.getenv("GAN_SAVE_EVERY", "200")))
EARLY_STOP_PATIENCE = max(1, int(os.getenv("GAN_EARLY_STOP_PATIENCE", "200")))
GAN_EARLY_STOPPING = _env_flag("GAN_EARLY_STOPPING", False)
GAN_RESTORE_BEST = _env_flag("GAN_RESTORE_BEST", False)
TEMPORAL_NOISE = _env_flag("GAN_TEMPORAL_NOISE", True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device():
    print(f"使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"显卡: {torch.cuda.get_device_name(0)}")


def configure_dimensions(n_features, n_classes, seq_len=None):
    global N_FEATURES, N_CLASSES, SEQ_LEN
    N_FEATURES = int(n_features)
    N_CLASSES = int(n_classes)
    if seq_len is not None:
        SEQ_LEN = int(seq_len)
    if min(N_FEATURES, N_CLASSES, SEQ_LEN) <= 0:
        raise ValueError(
            f"非法数据维度: seq_len={SEQ_LEN}, features={N_FEATURES}, classes={N_CLASSES}"
        )


def make_dataloader(X, y, physics_targets=None, batch_size=BATCH_SIZE, shuffle=True):
    tensors = [torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)]
    if physics_targets is not None:
        tensors.append(torch.as_tensor(physics_targets, dtype=torch.float32))
    return DataLoader(
        TensorDataset(*tensors), batch_size=max(1, int(batch_size)), shuffle=shuffle,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )


def sample_noise(batch_size):
    shape = (
        (batch_size, SEQ_LEN, NOISE_DIM)
        if TEMPORAL_NOISE
        else (batch_size, NOISE_DIM)
    )
    return torch.randn(*shape, device=DEVICE)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.lstm = nn.LSTM(
            NOISE_DIM + EMBED_DIM, HIDDEN_DIM, N_LAYERS,
            batch_first=True, dropout=0.2 if N_LAYERS > 1 else 0.0,
        )
        self.out = nn.Sequential(nn.Linear(HIDDEN_DIM, N_FEATURES), nn.Tanh())

    def forward(self, z, labels):
        if z.ndim == 2:
            z_seq = z.unsqueeze(1).expand(-1, SEQ_LEN, -1)
        elif z.ndim == 3 and z.shape[1] == SEQ_LEN:
            z_seq = z
        else:
            raise ValueError(f"z形状必须为(B,{NOISE_DIM})或(B,{SEQ_LEN},{NOISE_DIM})")
        label_seq = self.label_embed(labels).unsqueeze(1).expand(-1, SEQ_LEN, -1)
        sequence, _ = self.lstm(torch.cat([z_seq, label_seq], dim=-1))
        return self.out(sequence)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.conv = nn.Sequential(
            nn.Conv1d(N_FEATURES + EMBED_DIM, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool1d(12)
        self.out = nn.Linear(256 * 12, 1)

    def forward(self, x, labels):
        label_seq = self.label_embed(labels).unsqueeze(1).expand(-1, x.size(1), -1)
        values = torch.cat([x, label_seq], dim=-1).permute(0, 2, 1)
        values = self.pool(self.conv(values)).reshape(values.size(0), -1)
        return self.out(values)


def gradient_penalty(discriminator, real, fake, labels):
    alpha = torch.rand(real.size(0), 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = discriminator(interp, labels)
    gradients = torch.autograd.grad(
        score, interp, torch.ones_like(score), create_graph=True, retain_graph=True
    )[0]
    return ((gradients.reshape(real.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def physics_residual(
    fake_x, labels, feature_names, gan_scaler, physics_targets=None,
    physics_params=None, label_effectiveness=None, epsilon=PHYS_EPSILON,
):
    """Experimental command-thrust consistency regularizer.

    The motor columns are located by name, never by their position.  Fault
    effectiveness is selected from each condition label.  ``physics_targets``
    is the real fold-local ``predicted_plant_thrust`` sequence paired with the
    batch.  Using the SE3-requested ``cmd_thrust`` would be physically wrong
    whenever actuator loss or saturation makes that request unreachable.  If
    the target is absent, hover weight is an explicitly reported fallback.
    Because the target trajectory is not an explicit generator condition,
    this is a distribution-level regularizer, not a full physics-informed
    dynamics residual.
    """
    params = physics_params or HEX_PHYSICAL_CHARACTERISTICS
    n_rotors = int(params["num_rotors"])
    expected_names = [f"cmd_motor{i}" for i in range(n_rotors)]
    index = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in expected_names if name not in index]
    if missing:
        raise ValueError(
            f"推力一致性正则需要全部电机指令特征，当前缺少{missing}；"
            "使用FEATURE_SET=combined/command或设置GAN_LAMBDA_PHYS=0。"
        )
    motor_idx = [index[name] for name in expected_names]
    scaled_cmd = fake_x[:, :, motor_idx]
    scale = torch.as_tensor(gan_scaler.scale_[motor_idx], dtype=fake_x.dtype, device=fake_x.device)
    offset = torch.as_tensor(gan_scaler.min_[motor_idx], dtype=fake_x.dtype, device=fake_x.device)
    cmd = (scaled_cmd - offset) / scale

    table = label_effectiveness or effectiveness_table(n_rotors)
    eta_matrix = torch.as_tensor(
        [table[int(i)] for i in range(N_CLASSES)], dtype=fake_x.dtype, device=fake_x.device
    )
    effectiveness = eta_matrix[labels].unsqueeze(1)
    effective_thrust = (effectiveness * float(params["k_eta"]) * cmd.pow(2)).sum(dim=-1)
    if physics_targets is None:
        target = torch.full_like(effective_thrust, float(params["mass"]) * 9.81)
    else:
        target = physics_targets.to(device=fake_x.device, dtype=fake_x.dtype)
        if target.ndim == 1:
            target = target.unsqueeze(1).expand_as(effective_thrust)
        if target.shape != effective_thrust.shape:
            raise ValueError(
                f"physics_thrust目标形状{tuple(target.shape)}与生成序列{tuple(effective_thrust.shape)}不一致"
            )
    residual = (effective_thrust - target).abs()
    tolerance = max(float(epsilon), 0.0) * target.abs().clamp_min(1e-6)
    return torch.clamp(residual - tolerance, min=0.0).mean()


def train(
    X_train=None, y_train=None, dataloader=None, save_dir=SAVE_DIR,
    save_models=True, n_epochs=N_EPOCHS, save_every=SAVE_EVERY,
    early_stop_patience=EARLY_STOP_PATIENCE, early_stopping=GAN_EARLY_STOPPING,
    restore_best=GAN_RESTORE_BEST, lambda_phys=LAMBDA_PHYS,
    phys_epsilon=PHYS_EPSILON, feature_names=None, gan_scaler=None,
    physics_targets=None, physics_params=None, label_effectiveness=None,
    verbose=True,
):
    if X_train is not None:
        if X_train.ndim != 3:
            raise ValueError(f"X_train必须为3维，实际={X_train.shape}")
        configure_dimensions(X_train.shape[2], int(np.max(y_train)) + 1, X_train.shape[1])
    if dataloader is None:
        if X_train is None or y_train is None:
            raise ValueError("train需要X_train/y_train或dataloader")
        dataloader = make_dataloader(X_train, y_train, physics_targets)
    if lambda_phys > 0 and (feature_names is None or gan_scaler is None):
        raise ValueError("启用推力一致性正则必须传入feature_names和折内gan_scaler")

    save_dir = Path(save_dir)
    if save_models:
        save_dir.mkdir(parents=True, exist_ok=True)
    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=LR_G, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))
    history = {"loss_G": [], "loss_D": [], "loss_G_adv": [], "loss_phys": []}
    best_loss, best_epoch, best_state, patience = float("inf"), -1, None, 0

    if verbose:
        print("\n开始训练六旋翼 WGAN-GP...")
        print(f"G参数={sum(p.numel() for p in G.parameters()):,}, D参数={sum(p.numel() for p in D.parameters()):,}")
        print(f"epochs={n_epochs}, batch={BATCH_SIZE}, temporal_noise={TEMPORAL_NOISE}")
        print(f"early_stopping={early_stopping}, restore_best={restore_best}")
        print(f"lambda_phys={lambda_phys}, target={'predicted_plant_thrust' if physics_targets is not None else 'hover fallback'}")

    for epoch in range(int(n_epochs)):
        totals = dict(g=0.0, d=0.0, adv=0.0, phys=0.0, ng=0, nd=0)
        for batch_index, batch in enumerate(dataloader):
            real_x, real_labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            real_target = batch[2].to(DEVICE) if len(batch) > 2 else None
            batch_size = real_x.size(0)

            opt_D.zero_grad(set_to_none=True)
            fake_x = G(sample_noise(batch_size), real_labels).detach()
            loss_D = (
                D(fake_x, real_labels).mean() - D(real_x, real_labels).mean()
                + LAMBDA_GP * gradient_penalty(D, real_x, fake_x, real_labels)
            )
            loss_D.backward()
            opt_D.step()
            totals["d"] += float(loss_D.item())
            totals["nd"] += 1

            if (batch_index + 1) % N_CRITIC == 0:
                opt_G.zero_grad(set_to_none=True)
                fake_x = G(sample_noise(batch_size), real_labels)
                adv = -D(fake_x, real_labels).mean()
                if lambda_phys > 0:
                    phys = physics_residual(
                        fake_x, real_labels, feature_names, gan_scaler,
                        physics_targets=real_target, physics_params=physics_params,
                        label_effectiveness=label_effectiveness, epsilon=phys_epsilon,
                    )
                else:
                    phys = torch.zeros((), dtype=fake_x.dtype, device=DEVICE)
                loss_G = adv + float(lambda_phys) * phys
                loss_G.backward()
                opt_G.step()
                totals["g"] += float(loss_G.item())
                totals["adv"] += float(adv.item())
                totals["phys"] += float(phys.item())
                totals["ng"] += 1

        avg_g = totals["g"] / max(totals["ng"], 1)
        avg_d = totals["d"] / max(totals["nd"], 1)
        avg_adv = totals["adv"] / max(totals["ng"], 1)
        avg_phys = totals["phys"] / max(totals["ng"], 1)
        for key, value in (("loss_G", avg_g), ("loss_D", avg_d),
                           ("loss_G_adv", avg_adv), ("loss_phys", avg_phys)):
            history[key].append(value)

        if early_stopping or restore_best:
            if np.isfinite(avg_g) and avg_g < best_loss:
                best_loss, best_epoch, patience = avg_g, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
            else:
                patience += 1
        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or epoch + 1 == n_epochs):
            print(
                f"Epoch [{epoch + 1:>4}/{n_epochs}] G={avg_g:>9.4f} "
                f"Adv={avg_adv:>9.4f} Phys={avg_phys:>9.4f} D={avg_d:>9.4f}"
            )
        if save_models and save_every and (epoch + 1) % save_every == 0:
            torch.save(G.state_dict(), save_dir / f"G_epoch{epoch + 1}.pth")
        if early_stopping and patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}; experimental best epoch={best_epoch}")
            break

    if restore_best and best_state is not None:
        G.load_state_dict(best_state)
        selected = "experimental_minimum_loss_G"
    else:
        selected = "final_epoch"
    G.training_history = {
        **{key: np.asarray(value) for key, value in history.items()},
        "lambda_phys": float(lambda_phys), "phys_epsilon": float(phys_epsilon),
        "early_stopping": bool(early_stopping), "restore_best": bool(restore_best),
        "selected_generator": selected, "best_epoch": int(best_epoch),
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        save_dir / "gan_losses.npz",
        loss_G=np.asarray(history["loss_G"]),
        loss_D=np.asarray(history["loss_D"]),
        loss_G_adv=np.asarray(history["loss_G_adv"]),
        loss_phys=np.asarray(history["loss_phys"]),
        lambda_phys=np.asarray([float(lambda_phys)]),
    )
    if save_models:
        torch.save(G.state_dict(), save_dir / "G_final.pth")
        torch.save(D.state_dict(), save_dir / "D_final.pth")
    return G, D, history["loss_G"], history["loss_D"]


def _generation_counts(target_labels, samples_per_label):
    if isinstance(samples_per_label, dict):
        labels = list(samples_per_label) if target_labels is None else list(target_labels)
        return [(int(label), max(0, int(samples_per_label.get(int(label), 0)))) for label in labels]
    labels = list(range(N_CLASSES)) if target_labels is None else list(target_labels)
    return [(int(label), max(0, int(samples_per_label))) for label in labels]


def generate_synthetic(
    G, target_labels=None, samples_per_label=0,
    generate_batch_size=GENERATE_BATCH_SIZE, verbose=True,
):
    counts = _generation_counts(target_labels, samples_per_label)
    generate_batch_size = max(1, int(generate_batch_size))
    G.eval()
    all_X, all_y = [], []
    if verbose:
        print(f"\n小批次生成: counts={dict(counts)}, GPU batch={generate_batch_size}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with torch.inference_mode():
        for label, count in counts:
            generated = 0
            while generated < count:
                size = min(generate_batch_size, count - generated)
                labels = torch.full((size,), label, dtype=torch.long, device=DEVICE)
                fake_gpu = G(sample_noise(size), labels)
                all_X.append(fake_gpu.cpu().numpy().astype(np.float32, copy=False))
                all_y.append(np.full(size, label, dtype=np.int64))
                generated += size
                del fake_gpu, labels
            if verbose and count:
                print(f"  label {label:>2}: {count}个")
    if not all_X:
        return np.empty((0, SEQ_LEN, N_FEATURES), np.float32), np.empty((0,), np.int64)
    return np.concatenate(all_X).astype(np.float32, copy=False), np.concatenate(all_y)


def _standalone():
    describe_device()
    raw_path, y_path = DATA_DIR / "X_raw.npy", DATA_DIR / "y_all.npy"
    if not raw_path.exists() or not y_path.exists():
        raise FileNotFoundError("先运行 data_process.py 生成 X_raw.npy/y_all.npy")
    X_raw, y = np.load(raw_path), np.load(y_path).astype(np.int64)
    configure_dimensions(X_raw.shape[2], int(y.max()) + 1, X_raw.shape[1])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X_raw.reshape(-1, X_raw.shape[-1])).reshape(X_raw.shape).astype(np.float32)
    feature_names = __import__("json").loads((DATA_DIR / "feature_names.json").read_text(encoding="utf-8"))
    targets_path = DATA_DIR / "physics_thrust.npy"
    targets = np.load(targets_path) if targets_path.exists() else None
    G, _D, loss_g, loss_d = train(
        X, y, save_dir=SAVE_DIR, gan_scaler=scaler, feature_names=feature_names,
        physics_targets=targets,
    )
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(loss_g, label="G")
    ax.plot(loss_d, label="D")
    ax.legend(); ax.grid(alpha=0.3); ax.set_title("WGAN-GP training loss")
    fig.savefig(SAVE_DIR / "losses.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("独立运行WGAN.py只训练GAN；正式比较请运行classify.py。")


if __name__ == "__main__":
    _standalone()
