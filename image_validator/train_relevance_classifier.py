"""
train_relevance_classifier.py
==============================

Fine-tunes a MobileNetV3-Small (ImageNet-pretrained) into a binary
"is this a photo of packaging/a product label?" classifier, to replace
the edge-density heuristic in DocumentRelevanceChecker for the
irrelevant-image case (e.g. a random document, a selfie, a screenshot
that happens to contain text but isn't packaging at all).

Why a classifier instead of more heuristics: edge density / OCR-keyword
presence can't reliably tell "packaging" apart from "any other photo
with text in it" — that's a visual/semantic distinction (shape of a
bottle/box/pouch, printed-label texture, etc.), which is exactly what a
CNN backbone already pretrained on ImageNet is good at, with only a
light fine-tune needed on top.

Requires: torch, torchvision (pip install torch torchvision).
Needs internet access on first run to download ImageNet weights.

--------------------------------------------------------------------
Expected data layout (you provide this — collect your own examples):

    data/
      train/
        packaging/        <- product photos: bottles, boxes, pouches,
                              wrappers, labels — front AND back panels,
                              various lighting/angles. Aim for 150-300+.
        not_packaging/     <- everything else: random documents, forms,
                              screenshots, selfies, receipts, random
                              objects, blank paper, etc. Similar count.
      val/
        packaging/
        not_packaging/

A rough 80/20 train/val split of ~300-600 total images is enough to get
a useful signal for a binary problem like this with transfer learning.
Include some of your OWN rejected/edge-case uploads if you have logs
from testing — real production-shaped mistakes teach the model more
than generic stock photos.

Run:
    python train_relevance_classifier.py --data-dir data --epochs 8

Output:
    relevance_classifier.pt   (state dict, loaded by relevance_classifier.py)
--------------------------------------------------------------------
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
CLASS_NAMES = ["not_packaging", "packaging"]  # alphabetical -> ImageFolder order


def build_model() -> nn.Module:
    """
    MobileNetV3-Small: chosen for speed (this runs as a pre-filter on
    every upload, so inference cost matters) over accuracy-maximizing
    but heavier backbones. Swap for mobilenet_v3_large or
    efficientnet_b0 if accuracy matters more than latency for your
    deployment target.
    """
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)  # binary: not_packaging / packaging
    return model


def build_dataloaders(data_dir: str, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(data_dir) / "val", transform=val_tf)

    assert train_ds.classes == CLASS_NAMES, (
        f"Expected class folders {CLASS_NAMES}, found {train_ds.classes}. "
        "Check your data/train subfolder names."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train(data_dir: str, epochs: int, batch_size: int, lr: float, output_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(data_dir, batch_size)
    model = build_model().to(device)

    # Freeze the backbone for the first few epochs, then unfreeze —
    # cheap way to get stable fine-tuning on a small dataset without
    # wrecking the pretrained features early on.
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    unfreeze_at = max(1, epochs // 3)

    for epoch in range(epochs):
        if epoch == unfreeze_at:
            print("Unfreezing backbone for full fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=lr / 10)

        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        val_acc = val_correct / max(1, val_total)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_acc:.3f} ({time.time() - t0:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    print(f"Best val accuracy: {best_acc:.3f}")
    torch.save(best_state, output_path)
    print(f"Saved weights to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Path containing train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="relevance_classifier.pt")
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.batch_size, args.lr, args.output)
