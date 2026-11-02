"""
videomae_dataset.py
================================================================
4 classes: Fighting, Robbery, Vandalism, Normal  (Assault removed)

AUGMENTATION PIPELINE (train split only, 7 steps):
  Step 1: Speed perturbation (30%)
  Step 2: Temporal boundary jitter (70%)
  Step 3: Random spatial crop 192→224 (60%)
  Step 4: Horizontal flip (50%)
  Step 5: Color jitter (50%)
  Step 6: Gaussian noise (35%)
  Step 7: Random grayscale (20%)
"""

import re
import random
import platform
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torchvision.transforms as T

ROOT      = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / 'data' / 'processed' / 'clips'

CLASSES      = ['Fighting', 'Robbery', 'Vandalism', 'Normal']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD  = [0.229, 0.224, 0.225]

CLIP_LEN_DISK  = 16
CLIP_LEN_MODEL = 16
CLIP_SIZE      = 224
CROP_SIZE      = 192

VALID_FOLDS = [1, 2, 3, 4]
IS_WINDOWS  = platform.system() == 'Windows'


# ═══════════════════════════════════════════════════════════════════════════════
# Clip Collector
# ═══════════════════════════════════════════════════════════════════════════════

def _video_stem(clip_stem: str) -> str:
    return re.sub(r'_c\d{4}(_fh|_tr)?$', '', clip_stem)


def collect_clips(clips_dir: Path, split: str, fold: int,
                  verify: bool = True) -> tuple:
    assert fold in VALID_FOLDS, f"fold must be 1–4, got {fold}"

    fold_dir  = clips_dir / f'fold_{fold}'
    split_dir = fold_dir / split

    def get_stems(sp: str) -> set:
        stems = set()
        for cls in CLASSES:
            d = fold_dir / sp / cls / 'scene'
            if d.exists():
                for p in d.glob('*.npy'):
                    stems.add(_video_stem(p.stem))
        return stems

    print(f"\n{'='*65}")
    print(f"  SceneClipDataset  fold={fold}  split={split}")
    print(f"{'='*65}")

    if verify:
        train_stems = get_stems('train')
        test_stems  = get_stems('test')
        overlap     = train_stems & test_stems
        if overlap:
            raise RuntimeError(
                f"Leakage in fold {fold}: {len(overlap)} video stems in BOTH "
                f"train and test. Re-extract clips/fold_{fold}/."
            )
        print(f"[Dataset] ✅ Leakage check passed (fold {fold})")

    all_clips, all_labels = [], []

    for cls in CLASSES:
        label     = CLASS_TO_IDX[cls]
        scene_dir = split_dir / cls / 'scene'
        if not scene_dir.exists():
            print(f"[Dataset] ⚠  {scene_dir} not found — skipping {cls}")
            continue
        cls_clips = sorted(scene_dir.glob('*.npy'))
        n         = len(cls_clips)
        all_clips.extend(cls_clips)
        all_labels.extend([label] * n)
        print(f"[Dataset] fold_{fold}/{split}/{cls:10s}/scene : {n:5d} clips")

    print(f"[Dataset] fold_{fold}/{split} total : {len(all_clips)} clips\n")
    return all_clips, all_labels


# ═══════════════════════════════════════════════════════════════════════════════
# Class Weights
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(labels: list) -> dict:
    counts = Counter(labels)
    total  = len(labels)
    n_cls  = len(CLASSES)
    print("[Dataset] Class weights (inverse-frequency, clamped [0.5, 2.0]):")
    w = {}
    for i, cls in enumerate(CLASSES):
        cnt    = counts.get(i, 1)
        weight = total / (n_cls * cnt)
        weight = max(0.5, min(2.0, weight))
        w[cls] = round(weight, 4)
        print(f"  {cls:10s}: {cnt:5d} clips  weight={weight:.4f}")
    return w


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class SceneClipDataset(Dataset):
    def __init__(self, clips: list, labels: list, split: str = 'train'):
        self.clips   = clips
        self.labels  = labels
        self.augment = (split == 'train')
        self.norm    = T.Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD)
        self.jitter  = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.gray   = T.Grayscale(num_output_channels=3)
        self.resize = T.Resize(
            (CLIP_SIZE, CLIP_SIZE),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        raw   = np.load(self.clips[idx])
        label = self.labels[idx]
        clip  = torch.from_numpy(raw).float().permute(0, 3, 1, 2)
        if self.augment:
            clip = self._augment(clip)
        clip = torch.stack([self.norm(clip[t]) for t in range(CLIP_LEN_MODEL)])
        return clip, label

    def _augment(self, clip: torch.Tensor) -> torch.Tensor:
        T_len = clip.shape[0]

        # Step 1: Speed perturbation (30%)
        if random.random() > 0.7:
            indices = torch.arange(0, T_len, 2).repeat_interleave(2)
            clip    = clip[indices[:T_len]]

        # Step 2: Temporal boundary jitter (70%)
        if random.random() > 0.3:
            trim_s = random.randint(0, 2)
            trim_e = random.randint(0, 2)
            trim_e = max(0, min(trim_e, T_len - trim_s - 12))
            end     = T_len - trim_e if trim_e > 0 else T_len
            trimmed = clip[trim_s:end]
            while trimmed.shape[0] < CLIP_LEN_MODEL:
                trimmed = torch.cat(
                    [clip[:1], trimmed] if random.random() > 0.5
                    else [trimmed, clip[-1:]], dim=0)
            clip = trimmed[:CLIP_LEN_MODEL]

        # Step 3: Random spatial crop (60%)
        if random.random() > 0.4:
            top     = random.randint(0, CLIP_SIZE - CROP_SIZE)
            left    = random.randint(0, CLIP_SIZE - CROP_SIZE)
            cropped = clip[:, :, top:top + CROP_SIZE, left:left + CROP_SIZE]
            clip    = torch.stack([self.resize(cropped[t])
                                   for t in range(CLIP_LEN_MODEL)])

        # Step 4: Horizontal flip (50%)
        if random.random() > 0.5:
            clip = torch.flip(clip, dims=[3])

        # Step 5: Color jitter (50%)
        if random.random() > 0.5:
            clip = torch.stack([self.jitter(clip[t])
                                for t in range(CLIP_LEN_MODEL)])

        # Step 6: Gaussian noise (35%)
        if random.random() > 0.65:
            sigma = random.uniform(0.005, 0.025)
            clip  = (clip + torch.randn_like(clip) * sigma).clamp(0.0, 1.0)

        # Step 7: Random grayscale (20%)
        if random.random() > 0.8:
            clip = torch.stack([self.gray(clip[t])
                                for t in range(CLIP_LEN_MODEL)])
        return clip

    def get_sample_weights(self, class_weights: dict) -> list:
        return [class_weights[CLASSES[l]] for l in self.labels]


# ═══════════════════════════════════════════════════════════════════════════════
# Dataloader Factory
# ═══════════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    fold        : int   = 1,
    clips_dir   : str   = str(CLIPS_DIR),
    batch_size  : int   = 8,
    num_workers : int   = 4,
    use_sampler : bool  = True,
    verify      : bool  = True,
    sampler_mult: float = 2.0,
) -> tuple:
    assert fold in VALID_FOLDS, f"fold must be 1–4, got {fold}"
    clips_dir = Path(clips_dir)

    train_clips, train_labels = collect_clips(clips_dir, 'train', fold, verify)
    test_clips,  test_labels  = collect_clips(clips_dir, 'test',  fold, verify)

    tr_c = Counter(train_labels)
    te_c = Counter(test_labels)
    print("[Dataset] Class balance (train / test):")
    for i, cls in enumerate(CLASSES):
        tr = tr_c.get(i, 0)
        te = te_c.get(i, 0)
        r  = f"{tr/te:.1f}x" if te > 0 else "inf"
        print(f"  {cls:10s}: train={tr:5d}  test={te:4d}  ratio={r}")
    print()

    class_weights = compute_class_weights(train_labels)
    train_ds = SceneClipDataset(train_clips, train_labels, split='train')
    test_ds  = SceneClipDataset(test_clips,  test_labels,  split='test')

    # Windows crash prevention
    train_workers = min(num_workers, 2) if IS_WINDOWS else num_workers
    test_workers  = 0   # ALWAYS 0 — eliminates shared-memory crash

    if IS_WINDOWS:
        print(f"[Dataset] Windows: train_workers={train_workers}, "
              f"test_workers=0 (crash prevention)")

    if use_sampler:
        sw          = train_ds.get_sample_weights(class_weights)
        counts      = Counter(train_labels)
        min_count   = min(counts.values())
        n_classes   = len(CLASSES)
        sampler_len = min(len(sw), int(sampler_mult * min_count * n_classes))
        print(f"[Dataset] WeightedSampler: {sampler_len} samples/epoch "
              f"({sampler_mult}× mult, min_class={min_count})")
        print(f"          Steps/epoch @ batch={batch_size}: "
              f"{sampler_len // batch_size}\n")
        sampler      = WeightedRandomSampler(sw, sampler_len, replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size         = batch_size,
            sampler            = sampler,
            num_workers        = train_workers,
            pin_memory         = True,
            drop_last          = True,
            persistent_workers = False,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size         = batch_size,
            shuffle            = True,
            num_workers        = train_workers,
            pin_memory         = True,
            drop_last          = True,
            persistent_workers = False,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size         = batch_size,
        shuffle            = False,
        num_workers        = test_workers,
        pin_memory         = False,
        persistent_workers = False,
    )

    return train_loader, test_loader, class_weights


CLASS_WEIGHTS = {c: 1.0 for c in CLASSES}