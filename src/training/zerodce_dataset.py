"""
zerodce_dataset.py
==================
Dataset loaders for Zero-DCE++ training:
  - ExDARKDataset  : Stage 1 — all 7363 images, 256×256, no labels
  - LLVIPDataset   : Stage 2 — 15488 images + XML annotations, 256×256
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random


# ── ExDARK ─────────────────────────────────────────────────────────────────────

class ExDARKDataset(Dataset):
    """
    Stage 1 dataset. No labels — unsupervised enhancement training.
    Handles mixed .jpg / .jpeg / .png across 12 class folders.
    Training resolution: 256×256 (spec).
    """

    VALID_EXT = {'.jpg', '.jpeg', '.png'}

    def __init__(self, root_dir, img_size=256, augment=True):
        self.img_size = img_size
        self.augment  = augment
        self.paths    = self._collect(Path(root_dir))

        self.resize   = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()

        print(f"[ExDARK] {len(self.paths)} images | size={img_size}×{img_size} | augment={augment}")

    def _collect(self, root):
        paths = []
        for cls in sorted(root.iterdir()):
            if cls.is_dir():
                for f in cls.iterdir():
                    if f.suffix.lower() in self.VALID_EXT:
                        paths.append(f)
        return paths

    def _augment(self, img):
        if random.random() > 0.5:
            img = TF.hflip(img)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            img = TF.rotate(img, angle)
        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.resize(img)
        if self.augment:
            img = self._augment(img)
        return self.to_tensor(img)


# ── LLVIP ──────────────────────────────────────────────────────────────────────

class LLVIPDataset(Dataset):
    """
    Stage 2 dataset. Returns images + YOLO-format bounding boxes.
    Reads Pascal VOC XML annotations, converts to normalized [cx, cy, w, h].
    All images are person class only (class_id = 0).
    Training resolution: 256×256 (spec).
    """

    def __init__(self, visible_dir, ann_dir, split='train', img_size=256, augment=True):
        self.img_dir  = Path(visible_dir) / split
        self.ann_dir  = Path(ann_dir)
        self.img_size = img_size
        self.augment  = augment

        self.samples  = self._collect()
        self.resize   = T.Resize((img_size, img_size))
        self.to_tensor = T.ToTensor()

        print(f"[LLVIP {split}] {len(self.samples)} images | "
              f"size={img_size}×{img_size} | augment={augment}")

    def _collect(self):
        samples = []
        for img_path in sorted(self.img_dir.glob('*.jpg')):
            xml_path = self.ann_dir / (img_path.stem + '.xml')
            if xml_path.exists():
                samples.append((img_path, xml_path))
        return samples

    def _parse_xml(self, xml_path, orig_w, orig_h):
        """Parse VOC XML → list of [0, cx, cy, w, h] normalized."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.findall('object'):
            name = obj.find('name').text.lower()
            if name != 'person':
                continue
            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            cx = ((xmin + xmax) / 2) / orig_w
            cy = ((ymin + ymax) / 2) / orig_h
            bw = (xmax - xmin) / orig_w
            bh = (ymax - ymin) / orig_h
            boxes.append([0, cx, cy, bw, bh])   # class_id = 0 (person)

        return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))

    def _augment(self, img, boxes):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if len(boxes):
                boxes[:, 1] = 1.0 - boxes[:, 1]   # flip cx
        return img, boxes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        boxes = self._parse_xml(xml_path, orig_w, orig_h)

        img = self.resize(img)
        if self.augment:
            img, boxes = self._augment(img, boxes)

        return self.to_tensor(img), boxes


def collate_llvip(batch):
    """Custom collate: images → stacked tensor, boxes → list (variable length)."""
    images, boxes = zip(*batch)
    return torch.stack(images, 0), list(boxes)


# ── Loaders ────────────────────────────────────────────────────────────────────

def get_exdark_loaders(root_dir, batch_size=16, img_size=256, num_workers=4):
    """Returns train_loader, val_loader (90/10 split)."""
    from torch.utils.data import random_split

    full = ExDARKDataset(root_dir, img_size=img_size, augment=True)
    val_n = int(0.1 * len(full))
    trn_n = len(full) - val_n
    trn_ds, val_ds = random_split(full, [trn_n, val_n],
                                  generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trn_loader, val_loader


def get_llvip_loaders(visible_dir, ann_dir, batch_size=8, img_size=256, num_workers=4):
    """Returns train_loader, val_loader using LLVIP train/test splits."""
    trn_ds = LLVIPDataset(visible_dir, ann_dir, split='train',
                          img_size=img_size, augment=True)
    val_ds = LLVIPDataset(visible_dir, ann_dir, split='test',
                          img_size=img_size, augment=False)

    trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True,
                            drop_last=True, collate_fn=collate_llvip)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_llvip)
    return trn_loader, val_loader