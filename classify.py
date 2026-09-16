"""Formal, leakage-safe evaluation for the six-rotor simulation dataset.

Every fold holds out complete independent simulation sources.  Scalers and
GANs are fitted on the training fold only.  The script compares unweighted,
class-weighted, random-oversampled and fold-local WGAN augmentation baselines,
and reports operational safety metrics in addition to accuracy/F1.
"""

from collections import Counter
import json
import os
from pathlib import Path
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import WGAN
from provenance import code_hashes, runtime_info, sha256_file, stable_hash, write_json
from scenario_config import LABEL_NAMES as CONFIG_LABEL_NAMES, SCENARIO_BY_LABEL

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parent


def _path_env(name, default, aliases=()):
    value = os.getenv(name)
    if value is None:
        for alias in aliases:
            value = os.getenv(alias)
            if value is not None:
                break
    return Path(value).expanduser().resolve() if value else Path(default).resolve()


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


DATA_DIR = _path_env(
    "DATA_DIR_OVERRIDE", PROJECT_ROOT / "plots" / "hex" / "data_process" / "combined"
)
SAVE_DIR = _path_env(
    "CLASSIFIER_SAVE_DIR_OVERRIDE",
    PROJECT_ROOT / "plots" / "hex" / "classification" / "formal_combined",
    aliases=("OUTPUT_DIR_OVERRIDE",),
)
N_FOLDS = max(2, int(os.getenv("N_FOLDS", "4")))
SEED = int(os.getenv("EXPERIMENT_SEED", "42"))
HIDDEN_DIM = int(os.getenv("CLASSIFIER_HIDDEN_DIM", "128"))
N_LAYERS = int(os.getenv("CLASSIFIER_LAYERS", "2"))
BATCH_SIZE = max(1, int(os.getenv("CLASSIFIER_BATCH_SIZE", "64")))
N_EPOCHS = max(1, int(os.getenv("CLASSIFIER_EPOCHS", "100")))
QUALITY_EPOCHS = max(1, int(os.getenv("GAN_QUALITY_CLASSIFIER_EPOCHS", "30")))
LR = float(os.getenv("CLASSIFIER_LR", "0.001"))
GAN_EPOCHS = max(1, int(os.getenv("GAN_EPOCHS", str(WGAN.N_EPOCHS))))
GAN_QUALITY_SAMPLES = max(1, int(os.getenv("GAN_QUALITY_SAMPLES", "32")))
GAN_QUALITY_EVAL = _env_flag("GAN_QUALITY_EVAL", True)
GAN_SAVE_MODELS = _env_flag("FOLD_GAN_SAVE_MODELS", True)
RESUME = _env_flag("CLASSIFIER_RESUME", True)
REQUIRE_GROUPS = _env_flag("REQUIRE_GROUPS", True)
ATTENTION_POOLING = _env_flag("CLASSIFIER_ATTENTION_POOLING", True)
TARGET_POLICY = os.getenv("GAN_TARGET_POLICY", "max").strip().lower()
MAX_GENERATED_PER_CLASS = max(0, int(os.getenv("GAN_MAX_GENERATED_PER_CLASS", "0")))
GAN_LAMBDAS = [
    float(item.strip())
    # Nonzero values enable an experimental command-thrust consistency
    # regularizer. It is disabled by default because the target profile is not
    # an explicit generator condition and therefore is not a full dynamics
    # residual.
    for item in os.getenv("GAN_PHYSICS_LAMBDAS", "0").split(",")
    if item.strip()
]
if not GAN_LAMBDAS:
    GAN_LAMBDAS = [0.0]

SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 0
N_FEATURES = 0
N_CLASSES = 0
FEATURE_NAMES = []
LABEL_NAMES = {}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_data():
    global SEQ_LEN, N_FEATURES, N_CLASSES, FEATURE_NAMES, LABEL_NAMES, N_FOLDS, GAN_LAMBDAS
    required = [
        "X_raw.npy", "y_all.npy", "groups.npy", "feature_names.json",
        "cmd_thrust.npy", "physics_thrust.npy",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing and REQUIRE_GROUPS:
        raise FileNotFoundError(
            f"正式六旋翼评估缺少{missing}。请用新版main.py和data_process.py重新生成，"
            "不能退回随机窗口划分。"
        )
    if not (DATA_DIR / "X_raw.npy").exists() or not (DATA_DIR / "y_all.npy").exists():
        raise FileNotFoundError(f"缺少 {DATA_DIR / 'X_raw.npy'} 或 y_all.npy")

    X = np.load(DATA_DIR / "X_raw.npy", allow_pickle=False).astype(np.float32)
    y = np.load(DATA_DIR / "y_all.npy", allow_pickle=False).astype(np.int64)
    groups = (
        np.load(DATA_DIR / "groups.npy", allow_pickle=False).reshape(-1)
        if (DATA_DIR / "groups.npy").exists() else None
    )
    thrust = (
        np.load(DATA_DIR / "physics_thrust.npy", allow_pickle=False).astype(np.float32)
        if (DATA_DIR / "physics_thrust.npy").exists() else None
    )
    if X.ndim != 3 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"数据形状错误: X={X.shape}, y={y.shape}")
    if groups is None:
        raise RuntimeError("未找到groups.npy；正式实验禁止普通StratifiedKFold")
    if len(groups) != len(y) or thrust is None or thrust.shape != X.shape[:2]:
        raise ValueError(f"X/y/groups/physics_thrust未对齐: {X.shape}, {y.shape}, {groups.shape}, {None if thrust is None else thrust.shape}")
    if not np.isfinite(X).all() or not np.isfinite(thrust).all():
        raise ValueError("输入含NaN/Inf")

    SEQ_LEN, N_FEATURES = int(X.shape[1]), int(X.shape[2])
    N_CLASSES = int(y.max()) + 1
    if N_CLASSES != len(CONFIG_LABEL_NAMES):
        raise ValueError(
            f"正式六旋翼数据应为{len(CONFIG_LABEL_NAMES)}类，实际为{N_CLASSES}类"
        )
    FEATURE_NAMES = [str(x) for x in _load_json(DATA_DIR / "feature_names.json")]
    if len(FEATURE_NAMES) != N_FEATURES:
        raise ValueError("feature_names.json与X特征维度不一致")
    LABEL_NAMES = {int(k): str(v) for k, v in CONFIG_LABEL_NAMES.items() if int(k) < N_CLASSES}
    if set(np.unique(y)) != set(range(N_CLASSES)):
        raise ValueError("正式19类数据必须连续覆盖label 0..18")

    groups_per_class = {
        int(label): int(np.unique(groups[y == label]).size) for label in range(N_CLASSES)
    }
    max_folds = min(groups_per_class.values())
    if max_folds < 2:
        raise RuntimeError(f"每类独立仿真不足2次: {groups_per_class}")
    if N_FOLDS > max_folds:
        raise RuntimeError(
            f"请求{N_FOLDS}折，但每类最少只有{max_folds}个独立来源；"
            "正式实验不自动降低折数，请补充仿真。"
        )
    has_all_commands = all(f"cmd_motor{i}" in FEATURE_NAMES for i in range(6))
    if not has_all_commands:
        skipped = [x for x in GAN_LAMBDAS if x > 0]
        GAN_LAMBDAS = [x for x in GAN_LAMBDAS if x == 0]
        if skipped:
            print(f"当前特征方案不含全部电机指令，跳过推力正则GAN lambda={skipped}")

    print(f"使用设备: {DEVICE}")
    if torch.cuda.is_available():
        print(f"显卡: {torch.cuda.get_device_name(0)}")
    print(f"\n六旋翼数据: X={X.shape}, classes={N_CLASSES}, independent groups={len(np.unique(groups))}")
    print(f"特征: {FEATURE_NAMES}")
    print(f"每类独立来源: {groups_per_class}")
    print(f"评估: StratifiedGroupKFold({N_FOLDS}); 普通随机窗口划分已禁用")
    return X, y, groups, thrust


def fit_scalers(X_train_raw, X_test_raw):
    shape_train, shape_test = X_train_raw.shape, X_test_raw.shape
    cls_scaler = StandardScaler().fit(X_train_raw.reshape(-1, N_FEATURES))
    gan_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_raw.reshape(-1, N_FEATURES))
    X_train_cls = cls_scaler.transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(shape_train)
    X_test_cls = cls_scaler.transform(X_test_raw.reshape(-1, N_FEATURES)).reshape(shape_test)
    X_train_gan = gan_scaler.transform(X_train_raw.reshape(-1, N_FEATURES)).reshape(shape_train)
    return (
        X_train_cls.astype(np.float32), X_test_cls.astype(np.float32),
        X_train_gan.astype(np.float32), cls_scaler, gan_scaler,
    )


class FaultClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            N_FEATURES, HIDDEN_DIM, N_LAYERS, batch_first=True,
            dropout=0.3 if N_LAYERS > 1 else 0.0,
        )
        self.attention = nn.Linear(HIDDEN_DIM, 1)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, N_CLASSES)
        )

    def forward(self, x, return_attention=False):
        sequence, _ = self.lstm(x)
        if ATTENTION_POOLING:
            weights = torch.softmax(self.attention(torch.tanh(sequence)).squeeze(-1), dim=1)
            pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        else:
            weights = torch.full(
                (x.size(0), x.size(1)), 1.0 / x.size(1), device=x.device, dtype=x.dtype
            )
            pooled = sequence.mean(dim=1)
        logits = self.head(pooled)
        return (logits, weights) if return_attention else logits


def make_loader(X, y, shuffle=True):
    return DataLoader(
        TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=shuffle, pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )


def class_weights(y):
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    values = len(y) / (N_CLASSES * np.maximum(counts, 1.0))
    return torch.as_tensor(values, dtype=torch.float32, device=DEVICE)


def train_classifier(X, y, weighted=False, epochs=N_EPOCHS):
    model = FaultClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y) if weighted else None)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loader = make_loader(X, y, shuffle=True)
    model.train()
    for _ in range(int(epochs)):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model


def random_oversample(X, y, seed):
    rng = np.random.default_rng(seed)
    target = max(int(np.sum(y == label)) for label in range(N_CLASSES))
    indices = []
    for label in range(N_CLASSES):
        original = np.flatnonzero(y == label)
        if len(original) == 0:
            raise ValueError(f"训练折缺少label={label}")
        indices.extend(original.tolist())
        indices.extend(rng.choice(original, size=target - len(original), replace=True).tolist())
    indices = np.asarray(indices, dtype=np.int64)
    rng.shuffle(indices)
    return X[indices], y[indices]


def safety_metrics(y_true, y_pred):
    fault_mask, normal_mask = y_true != 0, y_true == 0
    miss = float(np.mean(y_pred[fault_mask] == 0)) if np.any(fault_mask) else np.nan
    false_alarm = float(np.mean(y_pred[normal_mask] != 0)) if np.any(normal_mask) else np.nan
    true_motor = np.array([SCENARIO_BY_LABEL[int(x)].fault_motor for x in y_true])
    pred_motor = np.array([SCENARIO_BY_LABEL[int(x)].fault_motor for x in y_pred])
    true_severity = np.array([SCENARIO_BY_LABEL[int(x)].severity for x in y_true])
    pred_severity = np.array([SCENARIO_BY_LABEL[int(x)].severity for x in y_pred])
    motor_acc = float(np.mean(true_motor[fault_mask] == pred_motor[fault_mask]))
    severity_acc = float(np.mean(true_severity[fault_mask] == pred_severity[fault_mask]))
    per_class_miss = {
        int(label): float(np.mean(y_pred[y_true == label] == 0))
        for label in range(1, N_CLASSES) if np.any(y_true == label)
    }
    return {
        "fault_miss_to_normal": miss, "normal_false_alarm": false_alarm,
        "motor_localization_accuracy": motor_acc, "severity_accuracy": severity_acc,
        "per_class_miss_to_normal": per_class_miss,
    }


def evaluate(model, X, y):
    model.eval()
    predictions, attention = [], []
    with torch.inference_mode():
        for batch_x, _ in make_loader(X, y, shuffle=False):
            logits, weights = model(batch_x.to(DEVICE), return_attention=True)
            predictions.append(logits.argmax(1).cpu().numpy())
            attention.append(weights.cpu().numpy())
    pred = np.concatenate(predictions).astype(np.int64)
    cm = confusion_matrix(y, pred, labels=range(N_CLASSES))
    partial = [item.label for item in SCENARIO_BY_LABEL.values() if item.severity == "partial"]
    partial_mask = np.isin(y, partial)
    result = {
        "acc": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "partial_accuracy": float(np.mean(pred[partial_mask] == y[partial_mask])),
        "per_class_accuracy": {
            int(label): float(np.mean(pred[y == label] == label))
            for label in range(N_CLASSES) if np.any(y == label)
        },
        **safety_metrics(y, pred),
        "mean_attention": np.concatenate(attention).mean(axis=0).astype(float).tolist(),
    }
    return result, cm, pred


def generation_counts(y_train):
    counts = {label: int(np.sum(y_train == label)) for label in range(N_CLASSES)}
    if TARGET_POLICY == "max":
        target = max(counts.values())
    elif TARGET_POLICY == "median":
        target = int(np.median(list(counts.values())))
    else:
        raise ValueError("GAN_TARGET_POLICY只能为max或median")
    deficits = {label: max(0, target - count) for label, count in counts.items()}
    if MAX_GENERATED_PER_CLASS:
        deficits = {label: min(count, MAX_GENERATED_PER_CLASS) for label, count in deficits.items()}
    return deficits


def gan_variant_name(lambda_phys):
    return (
        "wgan_plain"
        if float(lambda_phys) == 0
        else f"wgan_thrust_reg_{lambda_phys:g}"
    )


def distribution_quality(X_real, y_real, X_gen, y_gen):
    wd, ks, spectral = [], [], []
    for label in range(N_CLASSES):
        real = X_real[y_real == label]
        generated = X_gen[y_gen == label]
        if not len(real) or not len(generated):
            continue
        for feature in range(N_FEATURES):
            a = real[:, :, feature].reshape(-1)
            b = generated[:, :, feature].reshape(-1)
            wd.append(wasserstein_distance(a, b))
            ks.append(ks_2samp(a, b).statistic)
            pa = np.mean(np.abs(np.fft.rfft(real[:, :, feature], axis=1)) ** 2, axis=0)
            pb = np.mean(np.abs(np.fft.rfft(generated[:, :, feature], axis=1)) ** 2, axis=0)
            pa, pb = pa / max(pa.sum(), 1e-12), pb / max(pb.sum(), 1e-12)
            spectral.append(float(np.mean(np.abs(pa - pb))))
    return {
        "wasserstein_mean": float(np.mean(wd)),
        "ks_mean": float(np.mean(ks)),
        "normalized_psd_l1_mean": float(np.mean(spectral)),
    }


def atomic_pickle(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle:
        pickle.dump(payload, handle)
    os.replace(temp, path)


def load_checkpoint(path, signature):
    path = Path(path)
    if not RESUME or not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload if payload.get("signature") == signature else None
    except Exception:
        return None


def train_or_load_classifier(name, fold_dir, signature, X_train, y_train, X_test, y_test, test_idx, test_groups, weighted=False, epochs=N_EPOCHS):
    directory = fold_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, model_path = directory / "result.pkl", directory / "classifier.pth"
    saved = load_checkpoint(checkpoint, signature)
    if saved is not None and model_path.exists():
        model = FaultClassifier().to(DEVICE)
        try:
            state = torch.load(model_path, map_location=DEVICE, weights_only=True)
        except TypeError:  # PyTorch < 2.0
            state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"  [续跑] {name}: Acc={saved['result']['acc']:.4f}")
        return model, saved["result"], saved["cm"], saved["pred"]
    model = train_classifier(X_train, y_train, weighted=weighted, epochs=epochs)
    result, cm, pred = evaluate(model, X_test, y_test)
    torch.save(model.state_dict(), model_path)
    np.savez_compressed(
        directory / "test_predictions.npz", y_true=y_test, y_pred=pred,
        test_indices=test_idx, test_groups=test_groups,
    )
    atomic_pickle(checkpoint, {"signature": signature, "result": result, "cm": cm, "pred": pred})
    print(
        f"  {name}: Acc={result['acc']:.4f}, F1={result['f1_macro']:.4f}, "
        f"漏检={result['fault_miss_to_normal']:.4f}, 误报={result['normal_false_alarm']:.4f}"
    )
    return model, result, cm, pred


def train_or_load_gan(fold, fold_dir, signature, lambda_phys, X_train_raw, X_train_gan, y_train, thrust_train, gan_scaler, cls_scaler):
    name = gan_variant_name(lambda_phys)
    directory = fold_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "synthetic_meta.pkl"
    saved = load_checkpoint(meta_path, signature)
    if saved is not None and (directory / "X_gen_raw.npy").exists() and (directory / "y_gen.npy").exists():
        X_gen_raw = np.load(directory / "X_gen_raw.npy", allow_pickle=False)
        y_gen = np.load(directory / "y_gen.npy", allow_pickle=False)
        quality = saved.get("quality", {})
        print(f"  [续跑] {name}: 复用{len(y_gen)}个折内合成样本")
        return X_gen_raw, y_gen, quality

    set_seed(SEED + fold * 1000 + int(round(lambda_phys * 100000)) + 31)
    WGAN.configure_dimensions(N_FEATURES, N_CLASSES, SEQ_LEN)
    model_dir = directory / "models"
    print(f"  训练{name}: epochs={GAN_EPOCHS}, lambda_phys={lambda_phys}")
    G, _D, _lg, _ld = WGAN.train(
        X_train_gan, y_train, save_dir=model_dir, save_models=GAN_SAVE_MODELS,
        n_epochs=GAN_EPOCHS, early_stopping=False, restore_best=False,
        lambda_phys=lambda_phys, feature_names=FEATURE_NAMES, gan_scaler=gan_scaler,
        physics_targets=thrust_train, verbose=True,
    )
    deficits = generation_counts(y_train)
    X_gen_gan, y_gen = WGAN.generate_synthetic(
        G, target_labels=list(range(N_CLASSES)), samples_per_label=deficits,
        generate_batch_size=WGAN.GENERATE_BATCH_SIZE,
    )
    if len(X_gen_gan):
        X_gen_raw = gan_scaler.inverse_transform(
            X_gen_gan.reshape(-1, N_FEATURES)
        ).reshape(X_gen_gan.shape).astype(np.float32)
    else:
        X_gen_raw = np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
    quality = {"augmentation_counts": {int(k): int(v) for k, v in deficits.items()}}

    if GAN_QUALITY_EVAL:
        quality_counts = {label: GAN_QUALITY_SAMPLES for label in range(N_CLASSES)}
        X_q_gan, y_q = WGAN.generate_synthetic(
            G, target_labels=list(range(N_CLASSES)), samples_per_label=quality_counts,
            generate_batch_size=WGAN.GENERATE_BATCH_SIZE, verbose=False,
        )
        X_q_raw = gan_scaler.inverse_transform(
            X_q_gan.reshape(-1, N_FEATURES)
        ).reshape(X_q_gan.shape).astype(np.float32)
        quality["distribution"] = distribution_quality(X_train_raw, y_train, X_q_raw, y_q)
        np.save(directory / "X_quality_raw.npy", X_q_raw)
        np.save(directory / "y_quality.npy", y_q)

    np.save(directory / "X_gen_raw.npy", X_gen_raw)
    np.save(directory / "y_gen.npy", y_gen.astype(np.int64))
    atomic_pickle(meta_path, {"signature": signature, "quality": quality})
    return X_gen_raw, y_gen.astype(np.int64), quality


def quality_classification(base_model, quality_dir, cls_scaler, X_test_cls, y_test):
    if not GAN_QUALITY_EVAL or not (quality_dir / "X_quality_raw.npy").exists():
        return {}
    X_q_raw = np.load(quality_dir / "X_quality_raw.npy")
    y_q = np.load(quality_dir / "y_quality.npy").astype(np.int64)
    X_q_cls = cls_scaler.transform(X_q_raw.reshape(-1, N_FEATURES)).reshape(X_q_raw.shape).astype(np.float32)
    trts, _cm, _pred = evaluate(base_model, X_q_cls, y_q)
    synthetic_model = train_classifier(X_q_cls, y_q, weighted=False, epochs=QUALITY_EPOCHS)
    tstr, _cm, _pred = evaluate(synthetic_model, X_test_cls, y_test)
    return {"TRTS_accuracy": trts["acc"], "TSTR_accuracy": tstr["acc"]}


def run_experiment(X, y, groups, thrust):
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    results, cm_sum = {}, {}
    code_state = code_hashes(
        PROJECT_ROOT,
        [
            "classify.py", "WGAN.py", "scenario_config.py", "data_process.py",
            "hex_physical_characteristics.py", "provenance.py",
        ],
    )
    tracked_data_files = (
        "X_raw.npy", "y_all.npy", "groups.npy", "cmd_thrust.npy",
        "physics_thrust.npy",
        "feature_names.json", "preprocess_metadata.json", "source_manifest.json",
    )
    data_state = {
        name: sha256_file(DATA_DIR / name)
        for name in tracked_data_files if (DATA_DIR / name).exists()
    }
    variants = ["no_gan", "class_weighted", "random_oversampling"] + [gan_variant_name(x) for x in GAN_LAMBDAS]
    for name in variants:
        results[name], cm_sum[name] = [], np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), 1):
        fold_dir = SAVE_DIR / "folds" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_groups, test_groups = set(groups[train_idx]), set(groups[test_idx])
        if train_groups & test_groups:
            raise RuntimeError(f"Fold {fold}发生group泄漏")
        train_labels, test_labels = set(y[train_idx]), set(y[test_idx])
        required_labels = set(range(N_CLASSES))
        if train_labels != required_labels or test_labels != required_labels:
            raise RuntimeError(
                f"Fold {fold}未覆盖全部类别: "
                f"train缺{sorted(required_labels-train_labels)}, "
                f"test缺{sorted(required_labels-test_labels)}"
            )
        print(f"\n[Fold {fold}/{N_FOLDS}] train groups={len(train_groups)}, test groups={len(test_groups)}, overlap=0")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        thrust_train = thrust[train_idx]
        X_train_cls, X_test_cls, X_train_gan, cls_scaler, gan_scaler = fit_scalers(X_train_raw, X_test_raw)
        with (fold_dir / "classifier_scaler.pkl").open("wb") as f:
            pickle.dump(cls_scaler, f)
        with (fold_dir / "gan_minmax_scaler.pkl").open("wb") as f:
            pickle.dump(gan_scaler, f)

        base_signature_payload = {
            "fold": fold, "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(),
            "code_sha256": code_state, "data_sha256": data_state, "seed": SEED,
            "n_folds": N_FOLDS, "classifier_epochs": N_EPOCHS,
            "classifier_batch_size": BATCH_SIZE, "classifier_lr": LR,
            "classifier_hidden_dim": HIDDEN_DIM, "classifier_layers": N_LAYERS,
            "attention_pooling": ATTENTION_POOLING, "feature_names": FEATURE_NAMES,
            "gan_epochs": GAN_EPOCHS, "gan_lambdas": GAN_LAMBDAS,
            "gan_target_policy": TARGET_POLICY, "temporal_noise": WGAN.TEMPORAL_NOISE,
            "gan_batch_size": WGAN.BATCH_SIZE,
            "gan_generate_batch_size": WGAN.GENERATE_BATCH_SIZE,
            "gan_noise_dim": WGAN.NOISE_DIM, "gan_hidden_dim": WGAN.HIDDEN_DIM,
            "gan_layers": WGAN.N_LAYERS, "gan_embed_dim": WGAN.EMBED_DIM,
            "gan_lr_g": WGAN.LR_G, "gan_lr_d": WGAN.LR_D,
            "gan_n_critic": WGAN.N_CRITIC, "gan_lambda_gp": WGAN.LAMBDA_GP,
            "gan_phys_epsilon": WGAN.PHYS_EPSILON,
            "max_generated_per_class": MAX_GENERATED_PER_CLASS,
            "gan_quality_eval": GAN_QUALITY_EVAL,
            "gan_quality_samples": GAN_QUALITY_SAMPLES,
            "gan_quality_classifier_epochs": QUALITY_EPOCHS,
        }
        signature = stable_hash(base_signature_payload)
        write_json(fold_dir / "fold_signature.json", {**base_signature_payload, "signature": signature})

        set_seed(SEED + fold * 100 + 1)
        base_model, base_result, base_cm, _ = train_or_load_classifier(
            "no_gan", fold_dir, signature, X_train_cls, y_train,
            X_test_cls, y_test, test_idx, groups[test_idx], weighted=False,
        )
        results["no_gan"].append(base_result); cm_sum["no_gan"] += base_cm

        set_seed(SEED + fold * 100 + 2)
        _model, result, cm, _ = train_or_load_classifier(
            "class_weighted", fold_dir, signature, X_train_cls, y_train,
            X_test_cls, y_test, test_idx, groups[test_idx], weighted=True,
        )
        results["class_weighted"].append(result); cm_sum["class_weighted"] += cm

        X_ros, y_ros = random_oversample(X_train_cls, y_train, SEED + fold * 100 + 3)
        set_seed(SEED + fold * 100 + 3)
        _model, result, cm, _ = train_or_load_classifier(
            "random_oversampling", fold_dir, signature, X_ros, y_ros,
            X_test_cls, y_test, test_idx, groups[test_idx], weighted=False,
        )
        results["random_oversampling"].append(result); cm_sum["random_oversampling"] += cm

        for order, lambda_phys in enumerate(GAN_LAMBDAS):
            name = gan_variant_name(lambda_phys)
            gan_signature = stable_hash({"base": signature, "lambda_phys": lambda_phys})
            X_gen_raw, y_gen, quality = train_or_load_gan(
                fold, fold_dir, gan_signature, lambda_phys, X_train_raw,
                X_train_gan, y_train, thrust_train, gan_scaler, cls_scaler,
            )
            if len(X_gen_raw):
                X_gen_cls = cls_scaler.transform(
                    X_gen_raw.reshape(-1, N_FEATURES)
                ).reshape(X_gen_raw.shape).astype(np.float32)
            else:
                X_gen_cls = np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
            X_aug = np.concatenate([X_train_cls, X_gen_cls])
            y_aug = np.concatenate([y_train, y_gen])
            set_seed(SEED + fold * 100 + 10 + order)
            _model, result, cm, _ = train_or_load_classifier(
                name, fold_dir, gan_signature, X_aug, y_aug,
                X_test_cls, y_test, test_idx, groups[test_idx], weighted=False,
            )
            if "TRTS_accuracy" not in quality or "TSTR_accuracy" not in quality:
                quality.update(
                    quality_classification(
                        base_model, fold_dir / name, cls_scaler, X_test_cls, y_test
                    )
                )
                atomic_pickle(
                    fold_dir / name / "synthetic_meta.pkl",
                    {"signature": gan_signature, "quality": quality},
                )
            result["gan_quality"] = quality
            results[name].append(result); cm_sum[name] += cm

    return results, cm_sum, code_state, data_state


def summarize(results):
    summary = {}
    print("\n" + "=" * 78)
    print(f"{N_FOLDS}折独立仿真源交叉验证汇总")
    print("=" * 78)
    for name, folds in results.items():
        item = {}
        for metric in (
            "acc", "f1_macro", "partial_accuracy", "fault_miss_to_normal",
            "normal_false_alarm", "motor_localization_accuracy", "severity_accuracy",
        ):
            values = np.asarray([fold[metric] for fold in folds], dtype=float)
            item[f"{metric}_mean"] = float(np.nanmean(values))
            item[f"{metric}_std"] = float(np.nanstd(values))
        summary[name] = item
        print(
            f"{name:<24} Acc={item['acc_mean']:.4f}±{item['acc_std']:.4f}  "
            f"F1={item['f1_macro_mean']:.4f}  漏检={item['fault_miss_to_normal_mean']:.4f}  "
            f"误报={item['normal_false_alarm_mean']:.4f}"
        )
    return summary


def visualize(results, summary, cm_sum):
    names = list(results)
    folds = np.arange(1, N_FOLDS + 1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    for name in names:
        axes[0, 0].plot(folds, [x["acc"] for x in results[name]], marker="o", label=name)
        axes[0, 1].plot(folds, [x["f1_macro"] for x in results[name]], marker="o", label=name)
    axes[0, 0].set_title("Accuracy by held-out source fold")
    axes[0, 1].set_title("Macro F1 by held-out source fold")
    for ax in axes[0]:
        ax.set_ylim(0, 1.05); ax.set_xticks(folds); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    x = np.arange(len(names)); width = 0.35
    axes[1, 0].bar(x - width / 2, [summary[n]["fault_miss_to_normal_mean"] for n in names], width, label="fault miss")
    axes[1, 0].bar(x + width / 2, [summary[n]["normal_false_alarm_mean"] for n in names], width, label="normal false alarm")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names, rotation=35, ha="right")
    axes[1, 0].set_title("Operational safety errors (lower is better)")
    axes[1, 0].legend(); axes[1, 0].grid(axis="y", alpha=0.3)
    axes[1, 1].axis("off")
    text = "\n".join(
        f"{name}: Acc {summary[name]['acc_mean']:.3f}, F1 {summary[name]['f1_macro_mean']:.3f}, "
        f"motor {summary[name]['motor_localization_accuracy_mean']:.3f}, severity {summary[name]['severity_accuracy_mean']:.3f}"
        for name in names
    )
    axes[1, 1].text(0.02, 0.98, text, va="top", family="monospace")
    fig.savefig(SAVE_DIR / "formal_comparison.png", dpi=160)
    plt.close(fig)

    ticks = [LABEL_NAMES[i] for i in range(N_CLASSES)]
    for name, cm in cm_sum.items():
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(ticks, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name}: grouped {N_FOLDS}-fold confusion matrix")
        fig.colorbar(image, ax=ax); fig.tight_layout()
        fig.savefig(SAVE_DIR / f"confusion_{name}.png", dpi=150)
        plt.close(fig)


def main():
    X, y, groups, thrust = load_data()
    results, cm_sum, code_state, data_state = run_experiment(X, y, groups, thrust)
    summary = summarize(results)
    visualize(results, summary, cm_sum)
    np.save(SAVE_DIR / "fold_results.npy", results, allow_pickle=True)
    np.save(SAVE_DIR / "summary.npy", summary, allow_pickle=True)
    write_json(SAVE_DIR / "summary.json", summary)
    manifest = {
        **runtime_info(), "experiment": "hexarotor_grouped_fold_local_gan",
        "data_dir": str(DATA_DIR), "output_dir": str(SAVE_DIR),
        "n_folds": N_FOLDS, "feature_names": FEATURE_NAMES,
        "classifier_epochs": N_EPOCHS, "classifier_batch_size": BATCH_SIZE,
        "classifier_lr": LR, "classifier_hidden_dim": HIDDEN_DIM,
        "classifier_layers": N_LAYERS,
        "attention_pooling": ATTENTION_POOLING, "gan_epochs": GAN_EPOCHS,
        "gan_thrust_regularization_lambdas": GAN_LAMBDAS,
        "gan_target_policy": TARGET_POLICY,
        "gan_batch_size": WGAN.BATCH_SIZE,
        "gan_generate_batch_size": WGAN.GENERATE_BATCH_SIZE,
        "gan_temporal_noise": WGAN.TEMPORAL_NOISE,
        "gan_n_critic": WGAN.N_CRITIC, "gan_lambda_gp": WGAN.LAMBDA_GP,
        "gan_physics_epsilon": WGAN.PHYS_EPSILON,
        "gan_quality_eval": GAN_QUALITY_EVAL,
        "gan_quality_samples_per_label": GAN_QUALITY_SAMPLES,
        "gan_scaler": "fold-local MinMaxScaler(-1,1)",
        "classifier_scaler": "fold-local StandardScaler",
        "splitter": "StratifiedGroupKFold",
        "code_sha256": code_state, "data_sha256": data_state,
    }
    write_json(SAVE_DIR / "run_manifest.json", manifest)
    print(f"\n完成，结果保存到: {SAVE_DIR}")


if __name__ == "__main__":
    main()
