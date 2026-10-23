#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # PHÂN LOẠI TỔN THƯƠNG DA (ISIC 2019): DenseNet121 (pretrained) vs CNN tự thiết kế
# **Học phần:** Chuyên đề xử lý ảnh — Bài tập nhóm
# **Bộ dữ liệu:** [Kaggle JPEG ISIC 2019 512x512](https://www.kaggle.com/datasets/cdeotte/jpeg-isic2019-512x512)
#
# ## Đối chiếu yêu cầu bài tập
# | Yêu cầu | Cách đáp ứng trong code |
# |---|---|
# | Dữ liệu ảnh > 5 lớp, mỗi lớp > 500 ảnh | Tự động **loại các lớp có ≤ 500 ảnh** (VASC = 253, DF = 239) → còn **6 lớp**: NV, MEL, BCC, BKL, AK, SCC |
# | 1 mô hình đơn (pretrained) | `DenseNet121` (trọng số ImageNet) |
# | 1 mô hình tự chế | `Custom_Gated_DenseResNet` (huấn luyện từ đầu) |
# | Transfer learning | Giai đoạn 1: **đóng băng** toàn bộ backbone, chỉ train head |
# | Fine-tuning + rã đông | Giai đoạn 2: **rã đông** từ một khối dense trở đi (BatchNorm vẫn đóng băng) |
# | Mỗi mô hình 3 bộ tham số | `D1/D2/D3` cho DenseNet121, `C1/C2/C3` cho mô hình tự chế |
#
# ## Cách chạy
# - **Server GPU (script):** `python isic2019_densenet_custom.py --data-dir /path/to/jpeg-isic2019-512x512 --output-dir outputs`
# - **Colab (notebook):** sửa danh sách `ARGV` ở cell cuối rồi chạy.
# - Xem tất cả tùy chọn: `python isic2019_densenet_custom.py --help`
#
# > Tập Validation dùng để chọn checkpoint/cấu hình; tập Test chỉ dùng để báo cáo. Đây là bài toán nghiên cứu, không phải công cụ chẩn đoán lâm sàng.

# %% [markdown]
# ## Phần 0: Thư viện và tiện ích

# %%
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.ticker


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


if not in_notebook():
    matplotlib.use("Agg")  # server không có màn hình
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve, auc,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[FIGURE] {path}")
    if in_notebook():
        plt.show()
    else:
        plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    """Bảng Markdown đơn giản (không cần thư viện tabulate)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


# %% [markdown]
# ## Phần 1: Hằng số, 3 bộ siêu tham số cho mỗi mô hình và tham số dòng lệnh
#
# **DenseNet121** (`unfreeze_from` = tên tầng đầu tiên được rã đông; mọi tầng phía sau được huấn luyện, trừ BatchNorm):
# - `D1`: rã đông 8 khối dense cuối của `conv5` (nông nhất)
# - `D2`: rã đông toàn bộ `conv5` (khối dense thứ 4)
# - `D3`: rã đông từ giữa `conv4` đến hết `conv5` (sâu nhất), batch lớn hơn, dropout cao hơn
#
# **Custom_Gated_DenseResNet** (train từ đầu): thay đổi batch size, learning rate đỉnh và dropout.

# %%
DATASET_HANDLE = "cdeotte/jpeg-isic2019-512x512"
ALL_CLASS_CODES = ["NV", "MEL", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]
CLASS_NAMES_VI = {
    "NV": "Melanocytic nevus (Nốt ruồi sắc tố)",
    "MEL": "Melanoma (Ung thư hắc tố)",
    "BCC": "Basal cell carcinoma (UT tế bào đáy)",
    "BKL": "Benign keratosis (Dày sừng lành tính)",
    "AK": "Actinic keratosis (Dày sừng quang hóa)",
    "SCC": "Squamous cell carcinoma (UT tế bào vảy)",
    "VASC": "Vascular lesion (Tổn thương mạch máu)",
    "DF": "Dermatofibroma (U xơ da)",
}
DIAGNOSIS_SYNONYMS = {
    "NV": "NV", "MELANOCYTIC NEVUS": "NV", "NEVUS": "NV",
    "MEL": "MEL", "MELANOMA": "MEL",
    "BCC": "BCC", "BASAL CELL CARCINOMA": "BCC",
    "BKL": "BKL", "BENIGN KERATOSIS": "BKL", "SEBORRHEIC KERATOSIS": "BKL",
    "SOLAR LENTIGO": "BKL", "LICHENOID KERATOSIS": "BKL", "LENTIGO NOS": "BKL",
    "AK": "AK", "ACTINIC KERATOSIS": "AK",
    "SCC": "SCC", "SQUAMOUS CELL CARCINOMA": "SCC",
    "VASC": "VASC", "VASCULAR LESION": "VASC",
    "DF": "DF", "DERMATOFIBROMA": "DF",
}

DENSENET = "DenseNet121"
CUSTOM = "Custom_Gated_DenseResNet"

CONFIGS = {
    "D1": dict(model=DENSENET, name="D1 - Fine-tune nhẹ (8 khối cuối conv5)",
               batch_size=32, head_lr=1e-3, finetune_lr=1e-4, unfreeze_from="conv5_block9_0_bn", dropout=0.30),
    "D2": dict(model=DENSENET, name="D2 - Fine-tune toàn conv5",
               batch_size=32, head_lr=1e-3, finetune_lr=5e-5, unfreeze_from="conv5_block1_0_bn", dropout=0.30),
    "D3": dict(model=DENSENET, name="D3 - Fine-tune sâu (từ giữa conv4)",
               batch_size=64, head_lr=1e-3, finetune_lr=1e-4, unfreeze_from="conv4_block13_0_bn", dropout=0.50),
    "C1": dict(model=CUSTOM, name="C1 - LR thấp, batch nhỏ",
               batch_size=32, lr=5e-4, dropout=0.30),
    "C2": dict(model=CUSTOM, name="C2 - Baseline",
               batch_size=64, lr=1e-3, dropout=0.35),
    "C3": dict(model=CUSTOM, name="C3 - LR cao, dropout cao",
               batch_size=64, lr=2e-3, dropout=0.50),
}
# Tầng dùng cho Grad-CAM (feature map cuối cùng trước phần phân loại)
GRADCAM_LAYER = {DENSENET: "densenet121", CUSTOM: "gated_s4_2_out"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ISIC 2019: DenseNet121 (transfer learning + fine-tuning) vs CNN tự thiết kế",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # dữ liệu
    p.add_argument("--data-dir", default=None, help="Thư mục chứa train.csv + ảnh. Bỏ trống = tải bằng kagglehub")
    p.add_argument("--output-dir", default="outputs", help="Nơi lưu kết quả, model, biểu đồ")
    p.add_argument("--cache-dir", default=None, help="Nơi lưu cache ảnh đã resize (mặc định <output-dir>/cache)")
    p.add_argument("--img-size", type=int, default=256, help="Kích thước ảnh vào mạng (ảnh gốc 512)")
    p.add_argument("--min-images-per-class", type=int, default=500,
                   help="Chỉ giữ lớp có SỐ ẢNH > ngưỡng này (đề bài: > 500). Đặt 0 để giữ cả 8 lớp")
    p.add_argument("--sample-ratio", type=float, default=1.0, help="Lấy mẫu phân tầng để chạy thử nhanh")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--no-group-split", action="store_true",
                   help="Tắt chia theo patient_id (mặc định: ảnh cùng bệnh nhân không bị tách qua train/val/test)")
    p.add_argument("--split-file", default=None,
                   help="CSV chia tập (image_id,split). Có sẵn thì dùng lại; chưa có thì tạo. "
                        "Cả nhóm nên dùng CHUNG một file để so sánh công bằng. Mặc định <output-dir>/split.csv")
    # thí nghiệm
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS),
                   help="Cấu hình cần chạy. Chỉ DenseNet: --configs D1 D2 D3")
    p.add_argument("--seeds", nargs="+", type=int, default=[42], help="Mỗi cấu hình chạy 1 lần cho mỗi seed")
    p.add_argument("--head-epochs", type=int, default=5, help="DenseNet giai đoạn 1 (đóng băng backbone)")
    p.add_argument("--finetune-epochs", type=int, default=25, help="DenseNet giai đoạn 2 (rã đông), tối đa")
    p.add_argument("--custom-epochs", type=int, default=60, help="Mô hình tự chế, tối đa")
    p.add_argument("--warmup-epochs", type=int, default=3, help="Warm-up LR của mô hình tự chế")
    p.add_argument("--patience", type=int, default=6, help="Early stopping (DenseNet giai đoạn 2), theo val Macro-F1")
    p.add_argument("--custom-patience", type=int, default=12, help="Early stopping (mô hình tự chế)")
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weight-beta", type=float, default=0.5, help="0 = không dùng class weight, 1 = nghịch đảo tần suất")
    p.add_argument("--backbone-weights", default="imagenet",
                   help="'imagenet', đường dẫn file .h5 (server không có internet) hoặc 'none'")
    # phần cứng
    p.add_argument("--precision", choices=["auto", "mixed_float16", "mixed_bfloat16", "float32"], default="auto")
    p.add_argument("--workers", type=int, default=8, help="Số luồng đọc ảnh / nạp batch")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2])
    # điều khiển
    p.add_argument("--force", action="store_true", help="Chạy lại cả những run đã có result.json")
    p.add_argument("--report-only", action="store_true", help="Không huấn luyện, chỉ tạo lại bảng/biểu đồ từ kết quả có sẵn")
    p.add_argument("--skip-gradcam", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Chạy thử rất nhanh để kiểm tra môi trường")
    args = p.parse_args(argv)
    if args.smoke_test:
        args.sample_ratio, args.min_images_per_class = 0.05, 0
        args.img_size, args.head_epochs, args.finetune_epochs, args.custom_epochs = 96, 1, 1, 2
        args.warmup_epochs, args.eval_batch_size = 1, 64
    return args


# %% [markdown]
# ## Phần 2: Môi trường chạy (GPU, mixed precision)

# %%
def setup_runtime(args):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    precision = args.precision
    if precision == "auto":
        precision = "mixed_float16" if gpus else "float32"
    keras.mixed_precision.set_global_policy(precision)
    print("=" * 78)
    print(f"TensorFlow {tf.__version__} | Keras {keras.__version__}")
    print(f"GPU: {[g.name for g in gpus] or 'KHÔNG CÓ (sẽ chạy CPU, rất chậm)'}")
    print(f"Precision: {keras.mixed_precision.global_policy().name}")
    print("=" * 78)
    if int(keras.__version__.split(".")[0]) < 3:
        raise RuntimeError("Cần Keras >= 3 (TensorFlow >= 2.16).")


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# %% [markdown]
# ## Phần 3: Đọc metadata, lọc lớp theo yêu cầu (> 500 ảnh/lớp) và chia Train/Val/Test
#
# - Lớp nào có **≤ `--min-images-per-class`** ảnh bị loại **trước** khi chia dữ liệu (ISIC 2019: VASC = 253 và DF = 239 bị loại).
# - Chia **70/15/15 phân tầng theo lớp**, đồng thời **gom theo `patient_id`** để các ảnh của cùng một bệnh nhân không rơi vào cả train lẫn test (tránh rò rỉ dữ liệu làm điểm test cao giả tạo).
# - Kết quả chia được lưu ở `split.csv` — **cả nhóm dùng chung file này** để so sánh các mô hình công bằng.

# %%
def locate_dataset(data_dir):
    if data_dir:
        root = Path(data_dir)
    else:
        import kagglehub
        print(f"[DATA] Tải {DATASET_HANDLE} bằng kagglehub ...")
        root = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csvs = sorted(root.rglob("train.csv")) or sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"Không thấy file CSV trong {root}")
    images = {p.stem: str(p) for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")}
    print(f"[DATA] CSV: {csvs[0]} | {len(images):,} ảnh JPEG")
    return csvs[0], images


def load_metadata(args):
    csv_path, img_map = locate_dataset(args.data_dir)
    raw = pd.read_csv(csv_path)
    img_col = "image_name" if "image_name" in raw.columns else ("image" if "image" in raw.columns else raw.columns[0])
    df = pd.DataFrame({"image_id": raw[img_col].astype(str).str.replace(".jpg", "", regex=False).str.strip()})
    if "diagnosis" in raw.columns:
        df["class_code"] = raw["diagnosis"].astype(str).str.upper().str.strip().map(DIAGNOSIS_SYNONYMS)
    elif all(c in raw.columns for c in ALL_CLASS_CODES):
        df["class_code"] = raw[ALL_CLASS_CODES].idxmax(axis=1)
    else:
        raise ValueError(f"Không nhận ra cột nhãn trong {list(raw.columns)}")
    df["patient_id"] = raw["patient_id"] if "patient_id" in raw.columns else np.nan
    df["image_path"] = df["image_id"].map(img_map)
    df = df[df["class_code"].isin(ALL_CLASS_CODES) & df["image_path"].notna()].reset_index(drop=True)

    # --- Yêu cầu đề bài: mỗi lớp phải có > 500 ảnh ---
    full_counts = df["class_code"].value_counts().reindex(ALL_CLASS_CODES).fillna(0).astype(int)
    keep = [c for c in ALL_CLASS_CODES if full_counts[c] > args.min_images_per_class]
    dropped = [c for c in ALL_CLASS_CODES if c not in keep]
    print(f"[CLASSES] Số ảnh gốc theo lớp: {full_counts.to_dict()}")
    print(f"[CLASSES] Giữ {len(keep)} lớp (> {args.min_images_per_class} ảnh): {keep} | loại: {dropped}")
    if len(keep) <= 5:
        print("[CẢNH BÁO] Đề bài yêu cầu > 5 lớp nhưng chỉ còn", len(keep))
    df = df[df["class_code"].isin(keep)].reset_index(drop=True)
    df["label"] = df["class_code"].map({c: i for i, c in enumerate(keep)}).astype(np.int32)

    if 0.0 < args.sample_ratio < 1.0:
        df, _ = train_test_split(df, train_size=args.sample_ratio, stratify=df["label"], random_state=args.split_seed)
        df = df.sort_index().reset_index(drop=True)
        print(f"[DATA] Lấy mẫu phân tầng {args.sample_ratio:.0%}: {len(df):,} ảnh")

    group = df["patient_id"].astype("string")
    df["group"] = group.where(group.notna(), "img_" + df["image_id"]).astype(str)
    return df, keep, full_counts


def split_dataframe(df: pd.DataFrame, args) -> pd.Series:
    """Trả về Series 'train'/'val'/'test' cùng index với df."""
    use_groups = not args.no_group_split and df["group"].nunique() < len(df)
    if use_groups and df["group"].value_counts().iloc[0] / len(df) > 0.05:
        print("[SPLIT] patient_id có nhóm quá lớn (>5% dữ liệu) → bỏ gom nhóm, chia phân tầng thường.")
        use_groups = False
    split = pd.Series("train", index=df.index)
    y = df["label"].values
    if use_groups:
        print(f"[SPLIT] Chia phân tầng theo lớp + gom theo patient_id ({df['group'].nunique():,} nhóm)")
        k_test = max(2, round(1 / args.test_size))
        trval_i, test_i = next(StratifiedGroupKFold(k_test, shuffle=True, random_state=args.split_seed)
                               .split(df, y, df["group"]))
        rest = df.iloc[trval_i]
        k_val = max(2, round(len(rest) / (args.val_size * len(df))))
        _, val_rel = next(StratifiedGroupKFold(k_val, shuffle=True, random_state=args.split_seed)
                          .split(rest, rest["label"], rest["group"]))
        val_i = trval_i[val_rel]
    else:
        print("[SPLIT] Chia phân tầng theo lớp (không gom nhóm)")
        trval_i, test_i = train_test_split(np.arange(len(df)), test_size=args.test_size, stratify=y,
                                           random_state=args.split_seed)
        _, val_i = train_test_split(trval_i, test_size=args.val_size / (1 - args.test_size),
                                    stratify=y[trval_i], random_state=args.split_seed)
    split.iloc[test_i] = "test"
    split.iloc[val_i] = "val"
    return split


def prepare_split(df, class_codes, args, out: Path):
    split_path = Path(args.split_file) if args.split_file else out / "split.csv"
    if split_path.exists():
        sp = pd.read_csv(split_path)[["image_id", "split"]]
        n0 = len(df)
        df = df.merge(sp, on="image_id", how="inner").reset_index(drop=True)
        print(f"[SPLIT] Dùng lại {split_path} ({len(df):,}/{n0:,} ảnh khớp)")
    else:
        df["split"] = split_dataframe(df, args)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        df[["image_id", "class_code", "label", "split", "image_path"]].to_csv(split_path, index=False)
        print(f"[SPLIT] Đã lưu {split_path}")
    if (out / "split.csv") != split_path and not (out / "split.csv").exists():
        df[["image_id", "class_code", "label", "split", "image_path"]].to_csv(out / "split.csv", index=False)

    tbl = (df.groupby(["class_code", "split"]).size().unstack(fill_value=0)
           .reindex(class_codes)[["train", "val", "test"]])
    tbl["total"] = tbl.sum(axis=1)
    print(tbl.to_string())
    print(f"Train {tbl['train'].sum():,} | Val {tbl['val'].sum():,} | Test {tbl['test'].sum():,}")
    tbl.reset_index().to_csv(out / "reports" / "dataset_split_summary.csv", index=False, encoding="utf-8-sig")
    return df


# %% [markdown]
# ## Phần 4: Cache ảnh đã resize + nạp batch
#
# Ảnh JPEG 512×512 được **giải mã và resize đúng 1 lần**, lưu thành mảng `uint8` (`.npy`) rồi giữ trong RAM.
# Nhờ đó mỗi epoch không còn tốn CPU giải mã JPEG (nút thắt chính khi chạy Colab) và GPU được cấp dữ liệu liên tục.
# Data augmentation chạy ngay trên GPU, là các tầng đầu của mô hình.

# %%
def build_image_cache(df: pd.DataFrame, size: int, cache_dir: Path, workers: int) -> np.ndarray:
    key = hashlib.md5(("|".join(df["image_id"]) + f"|{size}").encode()).hexdigest()[:10]
    path = cache_dir / f"images_{size}px_{len(df)}_{key}.npy"
    if path.exists():
        print(f"[CACHE] Nạp {path}")
        return np.load(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    paths = df["image_path"].tolist()

    def load(i):
        img = cv2.imread(paths[i], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Không đọc được {paths[i]}")
        arr[i] = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    print(f"[CACHE] Resize {len(df):,} ảnh về {size}x{size} ...")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(load, range(len(df))))
    arr.flush()
    del arr
    tmp.rename(path)
    print(f"[CACHE] Xong sau {time.time() - t0:.0f}s → {path} ({path.stat().st_size / 1e9:.1f} GB)")
    return np.load(path)


class ArrayDataset(keras.utils.PyDataset):
    """Cấp batch từ mảng ảnh uint8 trong RAM. Train: xáo trộn mỗi epoch, bỏ batch lẻ, kèm sample_weight."""

    def __init__(self, images, labels, indices, batch_size, num_classes, shuffle=False,
                 sample_weights=None, seed=0, workers=4):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=workers * 2)
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)
        self.batch_size, self.num_classes = int(batch_size), num_classes
        self.shuffle, self.sample_weights = shuffle, sample_weights
        self.rng = np.random.default_rng(seed)
        self.order = self.indices.copy()
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        n = len(self.order)
        return n // self.batch_size if self.shuffle else math.ceil(n / self.batch_size)

    def __getitem__(self, i):
        idx = self.order[i * self.batch_size:(i + 1) * self.batch_size]
        idx = np.sort(idx)
        x = self.images[idx]
        y = keras.utils.to_categorical(self.labels[idx], self.num_classes).astype("float32")
        if self.sample_weights is not None:
            return x, y, self.sample_weights[self.labels[idx]].astype("float32")
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def compute_class_weights(train_labels, num_classes, beta):
    """w_c = (N / (K * N_c)) ** beta, chuẩn hóa để trọng số trung bình theo MẪU bằng 1."""
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    w = (len(train_labels) / (num_classes * counts)) ** beta
    w = w / np.sum(counts / counts.sum() * w)
    return w.astype(np.float32)


# %% [markdown]
# ## Phần 5: Khám phá dữ liệu (EDA)

# %%
def plot_eda(df, class_codes, full_counts, args, images, out: Path):
    fig_dir = out / "figures"
    # 01 — phân bố lớp + ngưỡng 500 ảnh của đề bài
    fig, ax = plt.subplots(figsize=(10, 4.8))
    codes = ALL_CLASS_CODES if args.min_images_per_class > 0 else class_codes
    cols = ["#2b6cb0" if c in class_codes else "#a0aec0" for c in codes]
    bars = ax.barh(codes[::-1], [full_counts[c] for c in codes][::-1], color=cols[::-1], edgecolor="black")
    ax.axvline(args.min_images_per_class, color="red", linestyle="--", label=f"Ngưỡng đề bài: {args.min_images_per_class} ảnh")
    for b, c in zip(bars, codes[::-1]):
        ax.text(b.get_width() + 0.01 * full_counts.max(), b.get_y() + b.get_height() / 2,
                f"{full_counts[c]:,}" + ("" if c in class_codes else "  (bị loại)"), va="center", fontsize=9)
    ax.set_xlabel("Số ảnh")
    ax.set_title("Phân bố số ảnh theo lớp — xanh: được giữ, xám: bị loại do ≤ ngưỡng", fontweight="bold")
    ax.legend(loc="lower right")
    save_fig(fig, fig_dir / "01_class_distribution.png")

    # 02 — ảnh mẫu
    n = len(class_codes)
    fig, axes = plt.subplots(4, n, figsize=(2.6 * n, 10.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for j, code in enumerate(class_codes):
        sub = df[df["class_code"] == code]
        sub = sub.sample(n=min(4, len(sub)), random_state=0)
        for i, idx in enumerate(sub.index):
            axes[i, j].imshow(images[idx])
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(f"{code}\n(n={int((df['class_code'] == code).sum()):,})", fontweight="bold")
    fig.suptitle("Ảnh mẫu dermoscopy theo lớp", fontweight="bold")
    save_fig(fig, fig_dir / "02_sample_grid.png")

    # 03 — augmentation
    aug = build_augmentation(seed=0, name="aug_preview")
    sample = images[df.index[0]][None].astype("float32")
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for k, ax in enumerate(axes.ravel()):
        img = sample if k == 0 else np.asarray(aug(sample, training=True))
        ax.imshow(np.clip(img[0] / 255.0, 0, 1))
        ax.set_title("Ảnh gốc" if k == 0 else f"Augment #{k}")
        ax.axis("off")
    fig.suptitle("Minh họa data augmentation (chỉ áp dụng khi train)", fontweight="bold")
    save_fig(fig, fig_dir / "03_augmentation_preview.png")


# %% [markdown]
# ## Phần 6: Kiến trúc mô hình
#
# ### 6.1 DenseNet121 pretrained
# `Input → Augmentation → Chuẩn hóa ImageNet → DenseNet121 → GAP → BN → Dense(256) → Dropout → Softmax`
#
# ### 6.2 Custom_Gated_DenseResNet (tự thiết kế)
# - **Stage 1-2:** các khối *dense* (nối đặc trưng, tái sử dụng đặc trưng mức thấp: màu, vân da).
# - **Stage 3-4:** conv stride 2 + **Gated Bottleneck Adapter** (nhánh residual có cổng):
#   $h' = h + g(h) \odot W_{up}\,\mathrm{DW}\,\mathrm{Swish}(W_{down}\,\mathrm{BN}(h))$, với $g(h)=\sigma(W_g\,\mathrm{GAP}(h))$; $W_{up}$ khởi tạo bằng 0 nên mỗi adapter bắt đầu như phép đồng nhất.
# - **Dual-pooling head:** $\alpha=\sigma(W[\mathrm{GAP};\mathrm{GMP}])$, đặc trưng $=\alpha\odot\mathrm{GAP}+(1-\alpha)\odot\mathrm{GMP}$.
#
# Tất cả đều ghép từ tầng Keras có sẵn nên lưu/nạp `.keras` không cần đăng ký lớp tùy chỉnh.

# %%
def build_augmentation(seed, name="augmentation"):
    """Ảnh dermoscopy không có hướng cố định → lật/xoay thoải mái; chỉ đổi tương phản nhẹ."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.25, fill_mode="reflect", seed=seed),
        layers.RandomZoom((-0.10, 0.10), (-0.10, 0.10), fill_mode="reflect", seed=seed),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect", seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name=name)


def _stem(img_size, seed):
    inputs = keras.Input((img_size, img_size, 3), dtype="uint8", name="input_image")
    x = layers.Rescaling(1.0, name="to_float")(inputs)      # uint8 → float, giữ thang 0..255
    x = build_augmentation(seed)(x)
    return inputs, x


def _classifier_head(x, num_classes, dropout, seed):
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dense(256, activation="swish", kernel_initializer="he_normal", name="head_fc")(x)
    x = layers.Dropout(dropout, seed=seed, name="head_drop")(x)
    return layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)


def build_densenet121(num_classes, img_size, dropout, seed, weights):
    inputs, x = _stem(img_size, seed)
    x = layers.Normalization(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                             variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
                             name="imagenet_norm")(x)
    backbone = keras.applications.DenseNet121(include_top=False, weights=weights,
                                              input_shape=(img_size, img_size, 3))
    backbone.trainable = False                               # Giai đoạn 1: transfer learning
    x = backbone(x)
    x = layers.GlobalAveragePooling2D(name="head_gap")(x)
    outputs = _classifier_head(x, num_classes, dropout, seed)
    return keras.Model(inputs, outputs, name=DENSENET), backbone


def unfreeze_backbone_from(backbone, start_layer: str):
    """Rã đông từ tầng `start_layer` đến hết backbone; BatchNorm luôn đóng băng."""
    names = [l.name for l in backbone.layers]
    if start_layer not in names:
        raise ValueError(f"Backbone không có tầng '{start_layer}'")
    start = names.index(start_layer)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)


def conv_bn_swish(x, filters, kernel_size=3, strides=1, name=None):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False,
                      kernel_initializer="he_normal", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("swish", name=f"{name}_swish")(x)


def dense_micro_layer(x, growth_rate, name):
    y = conv_bn_swish(x, growth_rate * 2, 1, name=f"{name}_bottleneck")
    y = conv_bn_swish(y, growth_rate, 3, name=f"{name}_conv3x3")
    return layers.Concatenate(name=f"{name}_concat")([x, y])


def transition_down(x, out_channels, name):
    x = conv_bn_swish(x, out_channels, 1, name=f"{name}_1x1")
    return layers.AveragePooling2D(2, strides=2, padding="same", name=f"{name}_pool")(x)


def gated_bottleneck_adapter(x, name, ratio=4):
    c = int(x.shape[-1])
    b = max(8, c // ratio)
    z = layers.BatchNormalization(name=f"{name}_bn")(x)
    z = layers.Conv2D(b, 1, activation="swish", kernel_initializer="he_normal", name=f"{name}_down")(z)
    z = layers.DepthwiseConv2D(3, padding="same", activation="swish", name=f"{name}_dw")(z)
    z = layers.Conv2D(c, 1, kernel_initializer="zeros", name=f"{name}_up")(z)
    g = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gate_gap")(x)
    g = layers.Conv2D(c, 1, activation="sigmoid", name=f"{name}_gate")(g)
    return layers.Add(name=f"{name}_out")([x, layers.Multiply(name=f"{name}_scale")([g, z])])


def dual_pooling_head(x, name="dual_pool"):
    c = int(x.shape[-1])
    avg = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    mx = layers.GlobalMaxPooling2D(name=f"{name}_gmp")(x)
    gate = layers.Dense(c, activation="sigmoid", name=f"{name}_gate")(layers.Concatenate(name=f"{name}_cat")([avg, mx]))
    inv = layers.Rescaling(-1.0, offset=1.0, name=f"{name}_inv")(gate)           # 1 - gate
    return layers.Add(name=f"{name}_mix")([layers.Multiply(name=f"{name}_a")([gate, avg]),
                                           layers.Multiply(name=f"{name}_m")([inv, mx])])


def build_custom_gated_denseresnet(num_classes, img_size, dropout, seed):
    inputs, x = _stem(img_size, seed)
    x = layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)
    x = conv_bn_swish(x, 32, 3, strides=2, name="stem")
    for i in range(3):
        x = dense_micro_layer(x, 24, f"dense_s1_{i}")
    x = transition_down(x, 96, "trans_s1")
    for i in range(3):
        x = dense_micro_layer(x, 32, f"dense_s2_{i}")
    x = transition_down(x, 160, "trans_s2")
    x = conv_bn_swish(x, 256, 3, strides=2, name="down_s3")
    for i in (1, 2):
        x = gated_bottleneck_adapter(x, f"gated_s3_{i}")
    x = conv_bn_swish(x, 384, 3, strides=2, name="down_s4")
    for i in (1, 2):
        x = gated_bottleneck_adapter(x, f"gated_s4_{i}")
    x = dual_pooling_head(x)
    outputs = _classifier_head(x, num_classes, dropout, seed)
    return keras.Model(inputs, outputs, name=CUSTOM)


def count_params(weights):
    return int(sum(np.prod(w.shape) for w in weights))


# %% [markdown]
# ## Phần 7: Huấn luyện một run (một cấu hình × một seed) và đánh giá
#
# **DenseNet121**
# 1. *Transfer learning:* backbone đóng băng, train head `head_epochs` epoch với `head_lr`.
# 2. *Fine-tuning / rã đông:* rã đông từ `unfreeze_from` (BatchNorm vẫn đóng băng), LR nhỏ `finetune_lr`, giảm LR khi val Macro-F1 chững, dừng sớm.
#
# **Custom:** train từ đầu, warm-up + cosine decay, dừng sớm.
#
# Cả hai: AdamW, label smoothing, class weight làm mượt. Checkpoint tốt nhất được chọn theo **Macro-F1 trên Validation** (không dùng Test).
# Sau giai đoạn 1 của DenseNet, điểm được ghi lại (`TL_*`) để chứng minh fine-tuning có cải thiện hay không so với chỉ transfer learning.

# %%
class ValMacroF1(keras.callbacks.Callback):
    """Tính Macro-F1 trên Validation mỗi epoch, ghi vào logs['val_macro_f1'] và giữ lại trọng số tốt nhất."""

    def __init__(self, val_ds, y_val, restore_best=True):
        super().__init__()
        self.val_ds, self.y_val, self.restore_best = val_ds, y_val, restore_best
        self.best, self.best_weights, self.best_epoch = -1.0, None, -1

    def on_epoch_end(self, epoch, logs=None):
        pred = np.argmax(self.model.predict(self.val_ds, verbose=0), axis=1)
        f1 = f1_score(self.y_val, pred, average="macro", zero_division=0)
        logs["val_macro_f1"] = float(f1)
        if f1 > self.best:
            self.best, self.best_epoch = f1, epoch
            self.best_weights = self.model.get_weights() if self.restore_best else None

    def on_train_end(self, logs=None):
        if self.restore_best and self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"[BEST] Khôi phục trọng số epoch {self.best_epoch + 1} (val Macro-F1 = {self.best:.4f})")


def compile_model(model, lr, args):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=args.weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def evaluate_predictions(y_true, prob, num_classes, class_codes):
    pred = prob.argmax(1)
    labels = list(range(num_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=labels, zero_division=0)
    try:
        auc_macro = roc_auc_score(label_binarize(y_true, classes=labels), prob, average="macro", multi_class="ovr")
    except Exception:
        auc_macro = float("nan")
    rec = {
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Macro_Precision": p.mean(), "Macro_Recall": r.mean(), "Macro_F1": f.mean(),
        "Weighted_F1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "Macro_ROC_AUC": auc_macro,
    }
    for i, c in enumerate(class_codes):
        rec[f"F1_{c}"], rec[f"Recall_{c}"] = float(f[i]), float(r[i])
    return {k: float(v) for k, v in rec.items()}


def run_one(cfg_id, seed, data, args, out: Path):
    cfg = CONFIGS[cfg_id]
    run_dir = out / "runs" / f"{cfg_id}_seed{seed}"
    if (run_dir / "result.json").exists() and not args.force:
        print(f"[SKIP] {cfg_id} seed {seed} đã có kết quả (dùng --force để chạy lại)")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    k = len(data.class_codes)
    print("\n" + "=" * 90)
    print(f"RUN {cfg_id} | {cfg['name']} | seed={seed}")
    print({a: b for a, b in cfg.items() if a not in ("model", "name")})
    print("=" * 90)

    keras.backend.clear_session()
    gc.collect()
    set_seed(seed)
    mk = lambda idx, bs, train: ArrayDataset(
        data.images, data.labels, idx, bs, k, shuffle=train,
        sample_weights=data.class_weights if train else None, seed=seed, workers=args.workers if train else 2)
    train_ds = mk(data.idx["train"], min(cfg["batch_size"], len(data.idx["train"])), True)
    val_ds = mk(data.idx["val"], args.eval_batch_size, False)
    test_ds = mk(data.idx["test"], args.eval_batch_size, False)
    y_val, y_test = data.labels[data.idx["val"]], data.labels[data.idx["test"]]
    val_cb = lambda restore: ValMacroF1(val_ds, y_val, restore_best=restore)
    fit = lambda m, epochs, cbs: m.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs,
                                       verbose=args.verbose).history
    info = {}
    t0 = time.time()

    if cfg["model"] == DENSENET:
        weights = None if str(args.backbone_weights).lower() == "none" else args.backbone_weights
        model, backbone = build_densenet121(k, args.img_size, cfg["dropout"], seed, weights)
        total_params = model.count_params()
        # --- Giai đoạn 1: transfer learning (đóng băng backbone) ---
        stage1_params = count_params(model.trainable_weights)
        print(f"[Giai đoạn 1 — Transfer learning] backbone đóng băng, tham số huấn luyện: {stage1_params:,}")
        compile_model(model, cfg["head_lr"], args)
        h1 = fit(model, args.head_epochs, [val_cb(True), keras.callbacks.TerminateOnNaN()])
        tl_val = f1_score(y_val, model.predict(val_ds, verbose=0).argmax(1), average="macro", zero_division=0)
        tl_prob = model.predict(test_ds, verbose=0)
        tl = evaluate_predictions(y_test, tl_prob, k, data.class_codes)
        info.update(TL_Val_Macro_F1=float(tl_val), TL_Test_Macro_F1=tl["Macro_F1"],
                    TL_Test_Balanced_Acc=tl["Balanced_Accuracy"])
        # --- Giai đoạn 2: fine-tuning (rã đông) ---
        unfreeze_backbone_from(backbone, cfg["unfreeze_from"])
        n_bb_train = count_params(backbone.trainable_weights)
        print(f"[Giai đoạn 2 — Fine-tuning] rã đông từ '{cfg['unfreeze_from']}': "
              f"{n_bb_train:,}/{count_params(backbone.weights):,} tham số backbone được huấn luyện (BN đóng băng)")
        compile_model(model, cfg["finetune_lr"], args)
        h2 = fit(model, args.finetune_epochs, [
            val_cb(True),
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.patience),
            keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.3, patience=3, min_lr=1e-7, verbose=1),
            keras.callbacks.TerminateOnNaN()])
        history = {key: list(h1.get(key, [])) + list(h2.get(key, [])) for key in set(h1) | set(h2)}
        history["stage1_epochs"] = len(h1["loss"])
        info.update(Unfreeze_From=cfg["unfreeze_from"], Backbone_Trainable_Params=n_bb_train)
    else:
        model = build_custom_gated_denseresnet(k, args.img_size, cfg["dropout"], seed)
        total_params = model.count_params()
        steps = len(train_ds)
        warm = args.warmup_epochs * steps
        schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=cfg["lr"] * 0.01, decay_steps=max(1, args.custom_epochs * steps - warm),
            alpha=0.01, warmup_target=cfg["lr"], warmup_steps=warm)
        compile_model(model, schedule, args)
        print(f"[Train from scratch] ≤{args.custom_epochs} epoch, LR đỉnh {cfg['lr']}, warm-up {args.warmup_epochs} epoch + cosine")
        history = fit(model, args.custom_epochs, [
            val_cb(True),
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=args.custom_patience),
            keras.callbacks.TerminateOnNaN()])
        history["stage1_epochs"] = 0
    train_time = time.time() - t0

    val_prob = model.predict(val_ds, verbose=0)
    model.predict(ArrayDataset(data.images, data.labels, data.idx["test"][:args.eval_batch_size],
                               args.eval_batch_size, k, workers=1), verbose=0)   # khởi động trước khi đo tốc độ
    t1 = time.time()
    test_prob = model.predict(test_ds, verbose=0)
    infer_ms = (time.time() - t1) / len(y_test) * 1000

    rec = {
        "Run_ID": f"{cfg_id}_seed{seed}", "Model": cfg["model"], "Config_ID": cfg_id, "Config_Name": cfg["name"],
        "Seed": seed, "Batch_Size": cfg["batch_size"], "Dropout": cfg["dropout"],
        "Head_LR": cfg.get("head_lr"), "FineTune_LR": cfg.get("finetune_lr"), "Custom_LR": cfg.get("lr"),
        "Epochs_Run": len(history["loss"]), "Stage1_Epochs": history["stage1_epochs"],
        "Val_Macro_F1": float(f1_score(y_val, val_prob.argmax(1), average="macro", zero_division=0)),
        "Val_Balanced_Acc": float(balanced_accuracy_score(y_val, val_prob.argmax(1))),
        **info,
        **evaluate_predictions(y_test, test_prob, k, data.class_codes),
        "Training_Time_s": train_time, "Inference_ms_per_img": infer_ms,
        "Total_Params": total_params, "Trainable_Params": count_params(model.trainable_weights),
    }
    rec["Trainable_Ratio_Pct"] = round(rec["Trainable_Params"] / total_params * 100, 2)
    print(f"[KẾT QUẢ] Val Macro-F1 {rec['Val_Macro_F1']:.4f} | Test Acc {rec['Accuracy']:.4f} | "
          f"Bal-Acc {rec['Balanced_Accuracy']:.4f} | Macro-F1 {rec['Macro_F1']:.4f} | AUC {rec['Macro_ROC_AUC']:.4f} | "
          f"{train_time / 60:.1f} phút")

    model.save(run_dir / "model.keras")
    np.save(run_dir / "test_probs.npy", test_prob.astype(np.float32))
    with open(run_dir / "history.json", "w") as f:
        json.dump({k_: [float(v) for v in vs] if isinstance(vs, list) else vs for k_, vs in history.items()}, f)
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:   # ghi cuối cùng = đánh dấu run hoàn tất
        json.dump(rec, f, ensure_ascii=False, indent=1)
    del model
    gc.collect()


# %% [markdown]
# ## Phần 8: Tổng hợp kết quả (bảng) từ thư mục `runs/`
#
# Mọi bảng đều được dựng lại từ các file `result.json`, nên có thể chạy lại phần báo cáo bất cứ lúc nào bằng `--report-only`.

# %%
MAIN_METRICS = [
    ("Val_Macro_F1", "Val Macro-F1"), ("Accuracy", "Accuracy"), ("Balanced_Accuracy", "Balanced Acc"),
    ("Macro_Precision", "Macro Precision"), ("Macro_Recall", "Macro Recall"), ("Macro_F1", "Test Macro-F1"),
    ("Weighted_F1", "Weighted F1"), ("Macro_ROC_AUC", "Macro ROC-AUC"),
]


def load_runs(out: Path):
    rows, hist, probs = [], {}, {}
    for d in sorted((out / "runs").glob("*_seed*")):
        if not (d / "result.json").exists():
            continue
        rec = json.loads((d / "result.json").read_text(encoding="utf-8"))
        rows.append(rec)
        hist[rec["Run_ID"]] = json.loads((d / "history.json").read_text())
        probs[rec["Run_ID"]] = np.load(d / "test_probs.npy")
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {c: i for i, c in enumerate(CONFIGS)}
        df = df.sort_values(["Config_ID", "Seed"], key=lambda s: s.map(order) if s.name == "Config_ID" else s).reset_index(drop=True)
    return df, hist, probs


def mean_std_str(s):
    s = s.astype(float)
    return f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if len(s) > 1 else f"{s.mean():.4f}"


def build_tables(runs: pd.DataFrame, class_codes, out: Path):
    rep = out / "reports"
    g = runs.groupby("Config_ID", sort=False)
    num = {}
    for key, _ in MAIN_METRICS + [("Training_Time_s", ""), ("Inference_ms_per_img", "")]:
        num[f"{key}_mean"], num[f"{key}_std"] = g[key].mean(), g[key].std(ddof=1)
    first = g.first()
    summary = pd.concat([first[["Model", "Config_Name", "Batch_Size", "Dropout", "Head_LR", "FineTune_LR", "Custom_LR",
                                "Unfreeze_From", "Total_Params", "Trainable_Params", "Backbone_Trainable_Params"]
                               if "Unfreeze_From" in first else
                               ["Model", "Config_Name", "Batch_Size", "Dropout", "Head_LR", "FineTune_LR", "Custom_LR",
                                "Total_Params", "Trainable_Params"]],
                         g.size().rename("N_Runs"), pd.DataFrame(num)], axis=1).reset_index()
    summary["Best_By_Val"] = summary.groupby("Model")["Val_Macro_F1_mean"].transform(lambda s: s == s.max())
    summary.to_csv(rep / "summary_by_config.csv", index=False, encoding="utf-8-sig")
    runs.to_csv(rep / "runs_raw.csv", index=False, encoding="utf-8-sig")

    per_f1 = g[[f"F1_{c}" for c in class_codes]].mean().round(4)
    per_rec = g[[f"Recall_{c}" for c in class_codes]].mean().round(4)
    per_f1.to_csv(rep / "per_class_f1.csv", encoding="utf-8-sig")
    per_rec.to_csv(rep / "per_class_recall.csv", encoding="utf-8-sig")

    pretty = pd.DataFrame({"Model": first["Model"], "Config": g["Config_Name"].first()})
    for key, label in MAIN_METRICS:
        pretty[label] = g[key].apply(mean_std_str)
    pretty["Train (phút)"] = (g["Training_Time_s"].mean() / 60).round(1)
    pretty["Params (M)"] = (first["Total_Params"] / 1e6).round(2)
    pretty["Train params (M)"] = (first["Trainable_Params"] / 1e6).round(2)
    pretty = pretty.reset_index()

    tl = None
    if "TL_Test_Macro_F1" in runs and runs["TL_Test_Macro_F1"].notna().any():
        d = runs[runs["Model"] == DENSENET].groupby("Config_ID", sort=False)
        tl = pd.DataFrame({
            "TL: Val Macro-F1": d["TL_Val_Macro_F1"].mean(), "TL+FT: Val Macro-F1": d["Val_Macro_F1"].mean(),
            "TL: Test Macro-F1": d["TL_Test_Macro_F1"].mean(), "TL+FT: Test Macro-F1": d["Macro_F1"].mean(),
            "TL: Test Bal-Acc": d["TL_Test_Balanced_Acc"].mean(), "TL+FT: Test Bal-Acc": d["Balanced_Accuracy"].mean(),
        }).round(4)
        tl["Δ Test Macro-F1"] = (tl["TL+FT: Test Macro-F1"] - tl["TL: Test Macro-F1"]).round(4)
        tl = tl.reset_index()
        tl.to_csv(rep / "transfer_vs_finetune.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(rep / "results.xlsx", engine="openpyxl") as w:
            pretty.to_excel(w, sheet_name="1_Summary", index=False)
            summary.to_excel(w, sheet_name="2_Summary_numeric", index=False)
            if tl is not None:
                tl.to_excel(w, sheet_name="3_Transfer_vs_Finetune", index=False)
            per_f1.to_excel(w, sheet_name="4_F1_per_class")
            per_rec.to_excel(w, sheet_name="5_Recall_per_class")
            runs.to_excel(w, sheet_name="6_Raw_runs", index=False)
            pd.read_csv(rep / "dataset_split_summary.csv").to_excel(w, sheet_name="7_Dataset_split", index=False)
    except ImportError:
        print("[INFO] Chưa cài openpyxl → bỏ qua results.xlsx (đã có CSV).")

    with open(rep / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Kết quả thực nghiệm (tập Test, trung bình ± độ lệch chuẩn qua các seed)\n\n")
        f.write(md_table(pretty) + "\n\n")
        if tl is not None:
            f.write("## Transfer learning (đóng băng) so với Fine-tuning (rã đông) — DenseNet121\n\n" + md_table(tl) + "\n\n")
        f.write("## F1 theo lớp\n\n" + md_table(per_f1.reset_index()) + "\n\n")
        f.write("## Recall theo lớp\n\n" + md_table(per_rec.reset_index()) + "\n")
    print(f"[REPORT] {rep}/summary.md , summary_by_config.csv , runs_raw.csv , results.xlsx")
    return summary, pretty, tl


# %% [markdown]
# ## Phần 9: Biểu đồ so sánh
# Learning curves, confusion matrix, ROC, so sánh chỉ số, F1 theo lớp. Nếu chạy nhiều seed, confusion matrix/ROC dùng trung bình các seed.

# %%
def plot_results(runs, hist, probs, class_codes, y_test, out: Path):
    fig_dir = out / "figures"
    cfgs = list(dict.fromkeys(runs["Config_ID"]))
    k = len(class_codes)

    # 04 — learning curves (loss | accuracy | val Macro-F1)
    fig, axes = plt.subplots(len(cfgs), 3, figsize=(16, 3.1 * len(cfgs)), squeeze=False)
    for r, cid in enumerate(cfgs):
        for _, run in runs[runs["Config_ID"] == cid].iterrows():
            h = hist[run["Run_ID"]]
            ep = range(1, len(h["loss"]) + 1)
            lab = f"seed {run['Seed']}"
            axes[r, 0].plot(ep, h["loss"], alpha=.8, label=f"train {lab}")
            axes[r, 0].plot(ep, h["val_loss"], "--", alpha=.8, label=f"val {lab}")
            axes[r, 1].plot(ep, h["accuracy"], alpha=.8, label=f"train {lab}")
            axes[r, 1].plot(ep, h["val_accuracy"], "--", alpha=.8, label=f"val {lab}")
            axes[r, 2].plot(ep, h["val_macro_f1"], alpha=.9, label=lab)
            if h.get("stage1_epochs"):
                for c in range(3):
                    axes[r, c].axvline(h["stage1_epochs"] + .5, color="purple", ls=":", lw=1.3,
                                       label="bắt đầu fine-tuning" if (c == 0 and run["Seed"] == runs["Seed"].iloc[0]) else None)
        name = runs[runs["Config_ID"] == cid]["Config_Name"].iloc[0]
        for c, t in enumerate(["Loss", "Accuracy", "Val Macro-F1"]):
            axes[r, c].set_title(f"{name} — {t}", fontsize=9, fontweight="bold")
            axes[r, c].set_xlabel("Epoch")
            axes[r, c].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            axes[r, c].legend(fontsize=6, ncol=2)
    fig.suptitle("Learning curves (train có augmentation + dropout nên thường thấp hơn val ở các epoch đầu)", fontweight="bold", y=1.0)
    fig.tight_layout()
    save_fig(fig, fig_dir / "04_learning_curves.png")

    def mean_probs(cid):
        return np.mean([probs[r] for r in runs[runs["Config_ID"] == cid]["Run_ID"]], axis=0)

    # 05 — confusion matrix (chuẩn hóa theo hàng = recall từng lớp)
    ncols = min(3, len(cfgs))
    nrows = math.ceil(len(cfgs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.7 * nrows), squeeze=False)
    for i, cid in enumerate(cfgs):
        ax = axes[i // ncols, i % ncols]
        cm = confusion_matrix(y_test, mean_probs(cid).argmax(1), labels=list(range(k)))
        sns.heatmap(cm / cm.sum(1, keepdims=True) * 100, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                    xticklabels=class_codes, yticklabels=class_codes, ax=ax)
        ax.set_title(runs[runs["Config_ID"] == cid]["Config_Name"].iloc[0], fontsize=9, fontweight="bold")
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
    for j in range(len(cfgs), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Confusion matrix trên Test (% theo hàng; đường chéo = recall từng lớp)", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, fig_dir / "05_confusion_matrices.png")

    # 06 — ROC macro-average
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y_bin = label_binarize(y_test, classes=list(range(k)))
    grid = np.linspace(0, 1, 200)
    for cid in cfgs:
        p = mean_probs(cid)
        tpr = np.mean([np.interp(grid, *roc_curve(y_bin[:, c], p[:, c])[:2]) for c in range(k)], axis=0)
        tpr[0], tpr[-1] = 0, 1
        ax.plot(grid, tpr, lw=2, label=f"{runs[runs['Config_ID'] == cid]['Config_Name'].iloc[0]} (AUC={auc(grid, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", label="Ngẫu nhiên")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC macro-average (one-vs-rest) trên Test", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    save_fig(fig, fig_dir / "06_roc_curves.png")

    # 07 — so sánh 5 chỉ số
    metrics = [("Accuracy", "Accuracy"), ("Macro_Precision", "Macro Precision"), ("Macro_Recall", "Macro Recall"),
               ("Macro_F1", "Macro F1"), ("Macro_ROC_AUC", "Macro ROC-AUC")]
    g = runs.groupby("Config_ID", sort=False)
    fig, ax = plt.subplots(figsize=(1.9 * len(cfgs) + 5, 6))
    x, w = np.arange(len(cfgs)), 0.16
    for m, (key, label) in enumerate(metrics):
        mean, std = g[key].mean().reindex(cfgs), g[key].std(ddof=1).reindex(cfgs).fillna(0)
        ax.bar(x + (m - 2) * w, mean, w, yerr=std, capsize=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(cfgs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Điểm (trung bình ± std qua seed)")
    ax.set_title("So sánh các chỉ số theo cấu hình (D = DenseNet121, C = CNN tự thiết kế)", fontweight="bold")
    ax.legend(ncol=3, fontsize=8)
    save_fig(fig, fig_dir / "07_metrics_comparison.png")

    # 08 — F1 theo lớp
    f1m = g[[f"F1_{c}" for c in class_codes]].mean().reindex(cfgs)
    fig, ax = plt.subplots(figsize=(1.3 * k + 3, 0.7 * len(cfgs) + 2.5))
    sns.heatmap(f1m.values, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
                xticklabels=class_codes, yticklabels=cfgs, cbar_kws={"label": "F1"})
    ax.set_title("F1-score theo lớp và cấu hình (Test)", fontweight="bold")
    save_fig(fig, fig_dir / "08_per_class_f1.png")


# %% [markdown]
# ## Phần 10: Grad-CAM
# Bản đồ nhiệt Grad-CAM thật (gradient của xác suất lớp dự đoán theo feature map cuối), cho cấu hình tốt nhất (theo **Validation Macro-F1**) của từng mô hình.
# Màu đỏ = vùng ảnh mô hình dựa vào nhiều nhất. Mục đích: kiểm tra mô hình nhìn vào tổn thương hay vào nền/thước đo/lông.

# %%
def make_cam_forward(model, model_name):
    """Trả về hàm x → (feature map cuối, xác suất). DenseNet có backbone lồng nhau nên chạy tay từng khối."""
    if model_name == DENSENET:
        pre = keras.Model(model.inputs, model.get_layer("imagenet_norm").output)
        backbone = model.get_layer(GRADCAM_LAYER[DENSENET])
        head = [model.get_layer(n) for n in ("head_gap", "head_bn", "head_fc", "head_drop", "predictions")]

        def forward(x):
            fmap = backbone(pre(x, training=False), training=False)
            h = fmap
            for layer in head:
                h = layer(h, training=False)
            return fmap, h
        return forward
    gm = keras.Model(model.inputs, [model.get_layer(GRADCAM_LAYER[model_name]).output, model.output])
    return lambda x: gm(x, training=False)


def grad_cam(forward, img_uint8):
    x = tf.convert_to_tensor(img_uint8[None])
    with tf.GradientTape() as tape:
        fmap, prob = forward(x)
        cls = int(tf.argmax(prob[0]))
        score = prob[:, cls]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(tf.cast(grads, tf.float32), axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * tf.cast(fmap, tf.float32), axis=-1))[0].numpy()
    cam = cv2.resize(cam, img_uint8.shape[:2][::-1])
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8), prob[0].numpy(), cls


def plot_gradcam(runs, class_codes, split_df, img_size, out: Path):
    test_df = split_df[split_df["split"] == "test"]
    if not test_df["image_path"].map(os.path.exists).all():
        print("[GRADCAM] Không tìm thấy ảnh gốc (đường dẫn trong split.csv) → bỏ qua Grad-CAM.")
        return
    for model_name in runs["Model"].unique():
        sub = runs[runs["Model"] == model_name]
        best_cfg = sub.groupby("Config_ID")["Val_Macro_F1"].mean().idxmax()
        best_run = sub[sub["Config_ID"] == best_cfg].sort_values("Val_Macro_F1").iloc[-1]
        model = keras.models.load_model(out / "runs" / best_run["Run_ID"] / "model.keras", compile=False)
        forward = make_cam_forward(model, model_name)
        k = len(class_codes)
        rows = math.ceil(k / 2)
        fig, axes = plt.subplots(rows, 4, figsize=(15, 3.7 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ci, code in enumerate(class_codes):
            cls_rows = test_df[test_df["class_code"] == code]
            if cls_rows.empty:
                continue
            row = cls_rows.iloc[0]
            img = cv2.resize(cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB), (img_size, img_size),
                             interpolation=cv2.INTER_AREA)
            cam, prob, pred = grad_cam(forward, img)
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            r, c = ci // 2, (ci % 2) * 2
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"Thực tế: {code}", fontweight="bold")
            axes[r, c + 1].imshow(np.uint8(0.6 * img + 0.4 * heat))
            axes[r, c + 1].set_title(f"Dự đoán: {class_codes[pred]} ({prob[pred] * 100:.0f}%)", fontweight="bold",
                                     color="#15803d" if pred == ci else "#dc2626")
        fig.suptitle(f"Grad-CAM — {model_name}, cấu hình {best_cfg} (chọn theo Val Macro-F1)", fontweight="bold")
        fig.tight_layout()
        save_fig(fig, out / "figures" / f"09_gradcam_{model_name}.png")
        del model
        keras.backend.clear_session()


# %% [markdown]
# ## Phần 11: Hàm `main` — chạy toàn bộ quy trình

# %%
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    for sub in ("reports", "figures", "runs"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        setup_runtime(args)
        df, class_codes, full_counts = load_metadata(args)
        df = prepare_split(df, class_codes, args, out)
        images = build_image_cache(df, args.img_size, Path(args.cache_dir or out / "cache"), args.workers)
        # (df đã được merge/sắp lại theo split → nạp lại ảnh theo đúng thứ tự df)
        k = len(class_codes)
        labels = df["label"].values.astype(np.int32)
        idx = {s: np.where(df["split"].values == s)[0] for s in ("train", "val", "test")}
        data = SimpleNamespace(
            images=images, labels=labels, idx=idx, class_codes=class_codes,
            class_weights=compute_class_weights(labels[idx["train"]], k, args.class_weight_beta))
        print("Trọng số lớp:", {c: round(float(w), 3) for c, w in zip(class_codes, data.class_weights)})
        (out / "reports" / "run_config.json").write_text(json.dumps({
            "args": vars(args), "class_codes": class_codes, "configs": {c: CONFIGS[c] for c in args.configs},
            "class_weights": data.class_weights.tolist()}, indent=1, ensure_ascii=False), encoding="utf-8")
        plot_eda(df, class_codes, full_counts, args, images, out)
        for cfg_id in args.configs:
            for seed in args.seeds:
                run_one(cfg_id, seed, data, args, out)

    cfg_path = out / "reports" / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError("Chưa có kết quả nào trong output-dir. Hãy chạy huấn luyện trước.")
    class_codes = json.loads(cfg_path.read_text(encoding="utf-8"))["class_codes"]
    runs, hist, probs = load_runs(out)
    if runs.empty:
        raise RuntimeError("Chưa có run nào hoàn tất.")
    split_df = pd.read_csv(out / "split.csv")
    y_test = split_df[split_df["split"] == "test"]["label"].values
    summary, pretty, _ = build_tables(runs, class_codes, out)
    print("\n" + pretty.drop(columns=["Model"]).to_string(index=False))
    plot_results(runs, hist, probs, class_codes, y_test, out)
    if not args.skip_gradcam:
        img_size = json.loads(cfg_path.read_text(encoding="utf-8"))["args"]["img_size"]
        plot_gradcam(runs, class_codes, split_df, img_size, out)
    print(f"\nHOÀN TẤT. Kết quả: {out.resolve()}  (reports/, figures/, runs/)")


# %% [script-only]
if __name__ == "__main__":
    main()

# %% [notebook-only]
# # ====== CHỈNH Ở ĐÂY RỒI CHẠY CELL ======
# ARGV = [
#     "--output-dir", "/content/drive/MyDrive/ISIC2019_outputs",   # lưu lên Drive để không mất khi Colab ngắt
#     "--configs", "D1", "D2", "D3", "C1", "C2", "C3",             # có thể chạy từng phần, run đã xong sẽ tự được bỏ qua
#     "--seeds", "42",
#     "--img-size", "224",                                         # Colab T4: 224 cho nhanh; server GPU nên dùng 256 trở lên
#     # "--data-dir", "/duong/dan/jpeg-isic2019-512x512",          # bỏ comment nếu đã có dữ liệu; mặc định tải bằng kagglehub
# ]
# main(ARGV)
