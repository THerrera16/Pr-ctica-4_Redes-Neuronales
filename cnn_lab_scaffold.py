#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laboratorio: Red Neuronal Convolucional (CNN) en CPU o GPU modesta
===================================================================
Dataset : Fashion-MNIST (60,000 imágenes de 28x28 en escala de grises, 10 clases)
Framework: PyTorch (versión CPU es suficiente)

Uso (desde la terminal):
    python cnn_lab_scaffold.py --part data
    python cnn_lab_scaffold.py --part baselines --epochs 3
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Tenis", "Bolso", "Botín"]
MEAN, STD = 0.2860, 0.3530          # estadísticos de Fashion-MNIST (entrenamiento)
N_VAL = 5000                        # imágenes reservadas para validación

ABLATIONS = {
    "width":   [dict(c1=8, c2=16), dict(c1=16, c2=32), dict(c1=32, c2=64)],
    "kernel":  [dict(k=3), dict(k=5)],
    "bn":      [dict(use_bn=False), dict(use_bn=True)],
    "dropout": [dict(p_drop=0.0), dict(p_drop=0.3), dict(p_drop=0.5)],
    "augment": [dict(augment=False), dict(augment=True)],
}


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades generales (proporcionadas)
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    """Fija las semillas para que los experimentos sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Elige GPU (CUDA o Apple MPS) si existe; si no, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Datos (proporcionados)
# ─────────────────────────────────────────────────────────────────────────────
def make_fake(n: int, seed: int):
    """Datos sintéticos: cada clase es un patrón suave fijo más ruido gaussiano."""
    g_proto = torch.Generator().manual_seed(123)
    protos = torch.randn(10, 1, 28, 28, generator=g_proto)
    protos = F.avg_pool2d(protos, 5, stride=1, padding=2) * 4
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    x = protos[y] + 6.0 * torch.randn(n, 1, 28, 28, generator=g)
    return x, y


def load_tensors(root: str, train: bool):
    """Descarga (si hace falta) Fashion-MNIST y lo devuelve como tensores normalizados."""
    from torchvision import datasets
    ds = datasets.FashionMNIST(root, train=train, download=True)
    x = ds.data.float().div(255.0).unsqueeze(1)           # (N, 1, 28, 28) en [0, 1]
    x = (x - MEAN) / STD
    return x, ds.targets.clone()


def build_data(args) -> dict:
    """Carga todo en memoria y separa entrenamiento / validación / prueba."""
    if args.fake:
        x_all, y_all = make_fake(60000, seed=1)
        x_te, y_te = make_fake(10000, seed=2)
    else:
        x_all, y_all = load_tensors(args.data_dir, train=True)
        x_te, y_te = load_tensors(args.data_dir, train=False)

    perm = torch.randperm(len(x_all), generator=torch.Generator().manual_seed(0))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    if args.subset > 0:
        tr_idx = tr_idx[:args.subset]
    return {
        "train": (x_all[tr_idx], y_all[tr_idx]),
        "val": (x_all[val_idx], y_all[val_idx]),
        "test": (x_te, y_te),
    }


def make_loaders(data: dict, batch_size: int, seed: int) -> dict:
    """Crea los DataLoader. El orden de mezcla depende de `seed` (comparaciones justas)."""
    g = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(TensorDataset(*data["train"]), batch_size=batch_size,
                            shuffle=True, generator=g),
        "val": DataLoader(TensorDataset(*data["val"]), batch_size=512),
        "test": DataLoader(TensorDataset(*data["test"]), batch_size=512),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aumento de datos  — TODO del estudiante (Se usa en Parte 3)
# ─────────────────────────────────────────────────────────────────────────────
def augment_batch(x: torch.Tensor, max_shift: int = 2) -> torch.Tensor:
    raise NotImplementedError("Completa el bloque TODO en la Parte 3")


# ─────────────────────────────────────────────────────────────────────────────
# Modelos (Completado para Parte 2)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Línea base SIN convoluciones: 784 → 128 → 64 → 10 (proporcionada)."""

    def __init__(self, hidden=(128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
            nn.Linear(hidden[1], 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """CNN pequeña para imágenes 1x28x28."""

    def __init__(self, c1=16, c2=32, k=3, use_bn=False, p_drop=0.0, hidden=64):
        super().__init__()
        pad = k // 2

        # Bloque 1: 28x28 -> 14x14
        layers1 = [nn.Conv2d(1, c1, kernel_size=k, padding=pad, bias=not use_bn)]
        if use_bn:
            layers1.append(nn.BatchNorm2d(c1))
        layers1.extend([nn.ReLU(), nn.MaxPool2d(2)])

        # Bloque 2: 14x14 -> 7x7
        layers2 = [nn.Conv2d(c1, c2, kernel_size=k, padding=pad, bias=not use_bn)]
        if use_bn:
            layers2.append(nn.BatchNorm2d(c2))
        layers2.extend([nn.ReLU(), nn.MaxPool2d(2)])

        self.features = nn.Sequential(*(layers1 + layers2))

        # Clasificador denso
        clf_layers = [
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, hidden),
            nn.ReLU(),
        ]
        if p_drop > 0.0:
            clf_layers.append(nn.Dropout(p_drop))
        clf_layers.append(nn.Linear(hidden, 10))

        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def count_params(model: nn.Module) -> int:
    """Número de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento y evaluación (Completado para Parte 2)
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer=None, augment=False):
    """Una pasada completa por `loader`. Si hay `optimizer`, entrena; si no, evalúa."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            if training and augment:
                x = augment_batch(x)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            seen += y.size(0)
    return loss_sum / seen, correct / seen


def fit(model, loaders, device, epochs, lr=1e-3, augment=False, weight_decay=0.0, verbose=True):
    """Entrena `epochs` épocas con Adam y registra el historial (incluye segundos por época)."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tl, ta = run_epoch(model, loaders["train"], device, opt, augment)
        vl, va = run_epoch(model, loaders["val"], device)
        dt = time.perf_counter() - t0
        for k, v in zip(hist, (tl, ta, vl, va, dt)):
            hist[k].append(v)
        if verbose:
            print(f"  época {ep:02d}/{epochs} | train {tl:.3f} / {ta:.3f} | "
                  f"val {vl:.3f} / {va:.3f} | {dt:5.1f} s")
    return hist


@torch.no_grad()
def predict(model, loader, device):
    """Devuelve (predicciones, etiquetas verdaderas) como tensores en CPU."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int = 10) -> np.ndarray:
    raise NotImplementedError("Completa el bloque TODO en la Parte 4")


def top_confusions(cm: np.ndarray, k: int = 5):
    """Los k pares (real → predicha) con más errores (proporcionada)."""
    pairs = [(int(cm[i, j]), i, j) for i in range(len(cm)) for j in range(len(cm)) if i != j]
    return sorted(pairs, reverse=True)[:k]


# ─────────────────────────────────────────────────────────────────────────────
# Visualización (proporcionada)
# ─────────────────────────────────────────────────────────────────────────────
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _denorm(x):
    return (x * STD + MEAN).clamp(0, 1)


def plot_samples(x, y, path, n=16):
    plt = _plt()
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 0.7, 3))
    for ax, img, lab in zip(axes.ravel(), x[:n], y[:n]):
        ax.imshow(_denorm(img[0]), cmap="gray")
        ax.set_title(CLASSES[int(lab)], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(histories: dict, path):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (name, h) in enumerate(histories.items()):
        c = f"C{i}"
        ep = range(1, len(h["val_loss"]) + 1)
        a1.plot(ep, h["train_loss"], "--", color=c, alpha=0.6)
        a1.plot(ep, h["val_loss"], "-", color=c, label=name)
        a2.plot(ep, h["train_acc"], "--", color=c, alpha=0.6)
        a2.plot(ep, h["val_acc"], "-", color=c, label=name)
    a1.set(title="Pérdida (--- train, — val)", xlabel="época")
    a2.set(title="Exactitud (--- train, — val)", xlabel="época")
    a2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASSES, fontsize=7)
    ax.set(xlabel="Predicha", ylabel="Real", title="Matriz de confusión (test)")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_filters(model, path):
    """Filtros aprendidos por la primera capa convolucional."""
    plt = _plt()
    w = model.features[0].weight.detach().cpu()
    n = w.size(0)
    cols = min(n, 8)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.2), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n):
        axes.ravel()[i].imshow(w[i, 0], cmap="RdBu_r")
        axes.ravel()[i].set_title(f"f{i}", fontsize=7)
    fig.suptitle("Filtros de la 1ª capa convolucional", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def plot_feature_maps(model, x, path, device, max_ch=8):
    """Mapas de activación (tras cada ReLU) para UNA imagen x de forma (1, 1, 28, 28)."""
    plt = _plt()
    model.eval()
    out, maps = x.to(device), []
    for layer in model.features:
        out = layer(out)
        if isinstance(layer, nn.ReLU):
            maps.append(out.cpu()[0])
    fig, axes = plt.subplots(len(maps) + 1, max_ch, figsize=(max_ch * 1.2, (len(maps) + 1) * 1.3),
                            squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    axes[0, 0].imshow(_denorm(x[0, 0]), cmap="gray")
    axes[0, 0].set_title("entrada", fontsize=7)
    for r, m in enumerate(maps, start=1):
        for c in range(min(max_ch, m.size(0))):
            axes[r, c].imshow(m[c], cmap="viridis")
            axes[r, c].set_title(f"capa {r} · c{c}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_misclassified(x, y_true, y_pred, path, n=12):
    plt = _plt()
    wrong = (y_true != y_pred).nonzero().flatten()[:n]
    fig, axes = plt.subplots(2, max(1, n // 2), figsize=(n * 0.9, 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), wrong):
        ax.imshow(_denorm(x[i, 0]), cmap="gray")
        ax.set_title(f"real: {CLASSES[int(y_true[i])]}\npred: {CLASSES[int(y_pred[i])]}", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Partes del laboratorio
# ─────────────────────────────────────────────────────────────────────────────
def part_data(args, data, out):
    print("\n=== PARTE 1 · Exploración de datos ===")
    for split, (x, y) in data.items():
        counts = torch.bincount(y, minlength=10).tolist()
        print(f"  {split:5s}: x={tuple(x.shape)}  y={tuple(y.shape)}  "
              f"media={x.mean():.3f} std={x.std():.3f}  clases={counts}")
    plot_samples(*data["train"], out / "01_muestras.png")
    print(f"  → {out / '01_muestras.png'}")


def part_baselines(args, data, out, device):
    print("\n=== PARTE 2 · Línea base (MLP) vs. CNN ===")
    results, histories = {}, {}
    for name, factory in (("MLP", MLP), ("CNN", SimpleCNN)):
        set_seed(args.seed)
        loaders = make_loaders(data, args.batch_size, args.seed)
        model = factory()
        n_par = count_params(model)
        print(f"\n[{name}] parámetros: {n_par:,}")
        h = fit(model, loaders, device, args.epochs, lr=args.lr)
        histories[name] = h
        results[name] = {"params": n_par, "val_acc": max(h["val_acc"]),
                         "train_acc": h["train_acc"][-1],
                         "s_per_epoch": float(np.mean(h["epoch_time"]))}
    plot_curves(histories, out / "02_baselines_curvas.png")
    print("\n  Modelo | Parámetros | Val acc | Train acc | s/época")
    for k, r in results.items():
        print(f"  {k:6s} | {r['params']:>10,} | {r['val_acc']:.4f}  | {r['train_acc']:.4f}   | {r['s_per_epoch']:.1f}")
    (out / "02_baselines.json").write_text(json.dumps(results, indent=2))


def part_ablation(args, data, out, device):
    print(f"\n=== PARTE 3 · Experimento controlado: '{args.ablation}' ===")
    results, histories = [], {}
    for cfg in ABLATIONS[args.ablation]:
        cfg = dict(cfg)
        augment = cfg.pop("augment", False)
        name = ", ".join(f"{k}={v}" for k, v in {**cfg, **({"augment": augment} if args.ablation == "augment" else {})}.items())
        set_seed(args.seed)
        loaders = make_loaders(data, args.batch_size, args.seed)
        model = SimpleCNN(**cfg)
        n_par = count_params(model)
        print(f"\n[{name}] parámetros: {n_par:,}")
        h = fit(model, loaders, device, args.epochs, lr=args.lr, augment=augment)
        histories[name] = h
        results.append({"config": name, "params": n_par,
                        "val_acc": max(h["val_acc"]), "train_acc": h["train_acc"][-1],
                        "gap": h["train_acc"][-1] - h["val_acc"][-1],
                        "s_per_epoch": float(np.mean(h["epoch_time"]))})
    plot_curves(histories, out / f"03_ablacion_{args.ablation}.png")
    print("\n  Configuración | Parámetros | Val acc | Train acc | Brecha | s/época")
    for r in results:
        print(f"  {r['config']:<22s}| {r['params']:>10,} | {r['val_acc']:.4f}  | "
              f"{r['train_acc']:.4f}   | {r['gap']:+.4f} | {r['s_per_epoch']:.1f}")
    (out / f"03_ablacion_{args.ablation}.json").write_text(json.dumps(results, indent=2))


def part_interpret(args, data, out, device):
    print("\n=== PARTE 4 · Evaluación final e interpretación ===")
    set_seed(args.seed)
    loaders = make_loaders(data, args.batch_size, args.seed)
    model = SimpleCNN()
    print(f"[CNN final] parámetros: {count_params(model):,}")
    fit(model, loaders, device, args.epochs, lr=args.lr)

    preds, targets = predict(model, loaders["test"], device)
    acc = (preds == targets).float().mean().item()
    print(f"\n  EXACTITUD EN TEST: {acc:.4f}")

    cm = confusion_matrix(targets, preds)
    print("  Pares más confundidos (real → predicha):")
    for n_err, i, j in top_confusions(cm):
        print(f"    {CLASSES[i]:>9s} → {CLASSES[j]:<9s}: {n_err}")

    x_te = data["test"][0]
    plot_confusion(cm, out / "04_confusion.png")
    plot_filters(model, out / "04_filtros.png")
    plot_feature_maps(model, x_te[:1], out / "04_mapas_activacion.png", device)
    plot_misclassified(x_te, targets, preds, out / "04_errores.png")
    print(f"  → figuras guardadas en {out}/")
    torch.save(model.state_dict(), out / "cnn_final.pt")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Laboratorio de CNN en CPU / GPU modesta")
    ap.add_argument("--part", choices=["data", "baselines", "ablation", "interpret"], required=True)
    ap.add_argument("--ablation", choices=list(ABLATIONS), default="width")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--subset", type=int, default=0, help="usar solo N imágenes de entrenamiento")
    ap.add_argument("--threads", type=int, default=0, help="hilos de CPU (0 = valor por defecto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fake", action="store_true", help="datos sintéticos, sin descarga")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Dispositivo: {device} | hilos CPU: {torch.get_num_threads()} | torch {torch.__version__}")
    if args.fake:
        print("AVISO: usando datos SINTÉTICOS (--fake). Los resultados no son de Fashion-MNIST.")

    data = build_data(args)
    if args.part == "data":
        part_data(args, data, out)
    elif args.part == "baselines":
        part_baselines(args, data, out, device)
    elif args.part == "ablation":
        part_ablation(args, data, out, device)
    else:
        part_interpret(args, data, out, device)


if __name__ == "__main__":
    main()